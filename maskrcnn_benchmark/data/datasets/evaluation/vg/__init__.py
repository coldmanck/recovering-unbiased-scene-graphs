from .vg_eval import do_vg_evaluation


def vg_evaluation(
    cfg,
    dataset,
    predictions,
    output_folder,
    logger,
    iou_types,
    labeling_prob_path,
    comp_soft_label,
    cache_name,
    labeling_prob,
    stl_eval_general,
    monitor_mean_recall,
    **_
):
    return do_vg_evaluation(
        cfg=cfg,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
        labeling_prob_path=labeling_prob_path,
        comp_soft_label=comp_soft_label,
        cache_name=cache_name,
        labeling_prob=labeling_prob,
        stl_eval_general=stl_eval_general,
        monitor_mean_recall=monitor_mean_recall,
    )
