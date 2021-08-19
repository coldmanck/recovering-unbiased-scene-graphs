# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # sanity checks
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL: # predcls
        assert 'precls' in cfg.OUTPUT_DIR or 'predcls' in cfg.OUTPUT_DIR
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL: # sgcls
        assert 'sgcls' in cfg.OUTPUT_DIR
    elif not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL: # sgdet
        assert 'sgdet' in cfg.OUTPUT_DIR
    else:
        assert False, "Not defined testing mode"
    if 'precls' in cfg.OUTPUT_DIR or 'predcls' in cfg.OUTPUT_DIR:
        assert cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
    elif 'sgcls' in cfg.OUTPUT_DIR:
        assert cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
    elif 'sgdet' in cfg.OUTPUT_DIR:
        assert not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
    else:
        assert False, "Not defined testing mode"

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    labeling_prob = None
    if cfg.MODEL.STL_EVAL_ZERO_OUT_NAN:
        train_labeling_prob_fname = 'train_labeling_prob_raw.pt'
    else:
        train_labeling_prob_fname = 'train_labeling_prob.pt'
    train_labeling_prob_path = os.path.join(output_folder, train_labeling_prob_fname)
    if cfg.MODEL.STL_EVAL:
        import pdb; pdb.set_trace()
        if not cfg.TEST.ALLOW_LOAD_LABEL_PROB_FROM_CACHE or not os.path.exists(train_labeling_prob_path):
            # inference on training set to compute labeling probability
            mode = 'train'
            data_loaders_val = make_data_loader(cfg, mode=mode, is_distributed=distributed, is_train=False)
            for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
                inference(
                    cfg,
                    model,
                    data_loader_val,
                    dataset_name,
                    mode=mode,
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                    labeling_prob_path=train_labeling_prob_path,
                )
                synchronize()

        labeling_prob = torch.load(train_labeling_prob_path)
        assert labeling_prob is not None
        factor_labeling_prob = (1 - labeling_prob) / labeling_prob
        if cfg.MODEL.STL_EVAL_ZERO_OUT_NAN:
            nan_idxs = torch.isnan(labeling_prob)
            factor_labeling_prob[nan_idxs] = 0.0

        mode = 'train'
        data_loaders_val = make_data_loader(cfg, mode=mode, is_distributed=distributed, is_train=False)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            inference(
                cfg,
                model,
                data_loader_val,
                dataset_name,
                mode=mode,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                labeling_prob=factor_labeling_prob,
            )
            synchronize()
    else:
        # factor_labeling_prob = None
        # if cfg.TEST.STL_MODE:
        #     assert os.path.exists(train_labeling_prob_path)
        #     labeling_prob = torch.load(train_labeling_prob_path)
        #     assert labeling_prob is not None
        #     factor_labeling_prob = (1 - labeling_prob) / labeling_prob
            
        data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            # DO NOT ADD mode='test' as it will change cache_name in inference.py
            inference(
                cfg,
                model,
                data_loader_val,
                dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                # labeling_prob=factor_labeling_prob,
            )
            synchronize()



if __name__ == "__main__":
    main()
