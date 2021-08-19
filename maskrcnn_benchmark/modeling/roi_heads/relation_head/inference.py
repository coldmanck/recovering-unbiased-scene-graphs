# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import get_constr_out
from .utils_relation import obj_prediction_nms

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        attribute_on,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
        stl_eval=False,
        test_stl_mode=False,
        test_stl_general=False,
        labeling_prob=None,
        use_balanced_norm=False,
        bal_reweight_only=False,
        balanced_norm_test_only=False,
        balanced_norm_as_soft_label=False,
        test_with_label_prob_only=False,
        c_hmc_test=False,
    ):
        """
        Arguments:

        """
        super(PostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        
        self.stl_eval = stl_eval

        self.test_stl_mode = test_stl_mode
        self.labeling_prob = labeling_prob
        self.test_stl_general = test_stl_general
        if self.test_stl_mode: # inference using STL method 
            assert self.labeling_prob is not None
            if self.test_stl_general: # use p(y=r|x) = p(s=r|x) / p(s=r|y=r)
                self.labeling_prob[0] = 1 # p(s=0|y=0) always equals 1
                # Correct the largest outlier, possibly "flying in" predicate
                minimum, second_minimum = sorted(self.labeling_prob)[:2]
                min_idx = torch.argmin(self.labeling_prob).item()
                self.labeling_prob[min_idx] = second_minimum

        self.balanced_norm_test_only = balanced_norm_test_only
        self.balanced_norm_as_soft_label = balanced_norm_as_soft_label
        self.test_with_label_prob_only = test_with_label_prob_only
        self.use_balanced_norm = False if bal_reweight_only or balanced_norm_test_only or balanced_norm_as_soft_label or test_with_label_prob_only else use_balanced_norm

        self.c_hmc_test = c_hmc_test
        

    def forward(self, x, rel_pair_idxs, boxes, relation_probs_norm=None, labeling_prob=None, matrix_of_ancestor=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []
        if self.use_balanced_norm:
            assert relation_probs_norm is not None
            start_idx = 0
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
            relation_logits, finetune_obj_logits, rel_pair_idxs, boxes
        )):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres)
                obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]
            
            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                boxlist = BoxList(box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs], box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class) # (#obj, )
            boxlist.add_field('pred_scores', obj_scores) # (#obj, )

            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)
            
            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            if self.test_with_label_prob_only:
                assert labeling_prob is not None
                # import pdb; pdb.set_trace()
                rel_class_prob = torch.cat([labeling_prob.clone().detach().view(1, -1)] * rel_logit.shape[0])
            elif self.use_balanced_norm:
                end_idx = start_idx + rel_logit.shape[0]
                rel_class_prob = relation_probs_norm[start_idx:end_idx]
                start_idx = end_idx
            elif self.c_hmc_test:
                # import pdb; pdb.set_trace()
                rel_logit = torch.sigmoid(rel_logit)
                # rel_class_prob = torch.cat([rel_logit[:, 0].view(-1, 1), get_constr_out(rel_logit[:, 1:], matrix_of_ancestor)], dim=1).float()
                rel_class_prob = get_constr_out(rel_logit, matrix_of_ancestor).float()
            else:
                # if self.balanced_norm_test_only:
                #     import pdb; pdb.set_trace()
                rel_class_prob = F.softmax(rel_logit, -1)
                if self.balanced_norm_as_soft_label:
                    # import pdb; pdb.set_trace()
                    assert labeling_prob is not None
                    rel_class_prob /= labeling_prob
            
            if self.test_stl_mode:
                if self.test_stl_general: # use p(y=r|x) = p(s=r|x) / p(s=r|y=r)
                    pred_soft_labels = rel_class_prob / self.labeling_prob.float()
                else: # use p(y=r|x, s=0) = (1 - p(s=r|y=r) / p(s=r|y=r)) * (p(s=r|x)/p(s=0|x))
                    factor_labeling_prob = (1 - self.labeling_prob) / self.labeling_prob
                    pred_soft_labels = factor_labeling_prob.float() * rel_class_prob / rel_class_prob[:, 0].view(-1, 1)
                    pred_soft_labels[:, 0] = rel_class_prob[:, 0] # keep bg prob unchanged
                pred_soft_labels /= torch.sum(pred_soft_labels, dim=-1).view(-1, 1)
                rel_class_prob = pred_soft_labels

            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1
            # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            try:
                if self.use_gt_box:
                    assert rel_pair_idx.shape[0] == (obj_class.shape[0] * (obj_class.shape[0] - 1))
            except:
                # import pdb; pdb.set_trace()
                print('[Warning] assertion failed: rel_pair_idx.shape[0] == (obj_class.shape[0] * (obj_class.shape[0] - 1))')

            boxlist.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels) # (#rel, )
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            # Note
            # TODO Kaihua: add a new type of element, which can have different length with boxlist (similar to field, except that once 
            # the boxlist has such an element, the slicing operation should be forbidden.)
            # it is not safe to add fields about relation into boxlist!
            results.append(boxlist)
        return results


def make_roi_relation_post_processor(cfg, labeling_prob=None):
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
    stl_eval = cfg.MODEL.STL_EVAL
    test_stl_mode = cfg.TEST.STL_MODE
    test_stl_general = cfg.TEST.STL_MODE_GENERAL
    use_balanced_norm = cfg.MODEL.BALANCED_NORM
    bal_reweight_only = cfg.TRAIN.BAL_REWEIGHT_ONLY
    balanced_norm_test_only = cfg.MODEL.BALANCED_NORM_TEST_ONLY
    balanced_norm_as_soft_label = cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL
    test_with_label_prob_only = cfg.TEST.WITH_LABEL_PROB_ONLY
    c_hmc_test = cfg.HMC.C_HMC_TEST

    postprocessor = PostProcessor(
        attribute_on,
        use_gt_box,
        later_nms_pred_thres,
        stl_eval,
        test_stl_mode,
        test_stl_general,
        labeling_prob,
        use_balanced_norm,
        bal_reweight_only,
        balanced_norm_test_only,
        balanced_norm_as_soft_label,
        test_with_label_prob_only,
        c_hmc_test,
    )
    return postprocessor
