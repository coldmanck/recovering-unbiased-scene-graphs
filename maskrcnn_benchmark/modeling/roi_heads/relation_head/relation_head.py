# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.layers import BalancedNorm1d, LearnableBalancedNorm1d
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor

import os
from maskrcnn_benchmark.modeling.utils import cat
import torch.nn.functional as F
import pickle


class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        
        if cfg.TEST.STL_MODE:
            assert cfg.TEST.STL_TRAINING_SET_LABELING_PROB != ''
            labeling_prob = torch.load(cfg.TEST.STL_TRAINING_SET_LABELING_PROB).cuda()
            assert labeling_prob is not None
        else:
            labeling_prob = None

        self.post_processor = make_roi_relation_post_processor(cfg, labeling_prob)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        # stl
        self.stl_train = self.cfg.MODEL.STL_TRAIN
        self.stl_eval = self.cfg.MODEL.STL_EVAL

        # testing with label prob directly
        if self.cfg.TEST.WITH_LABEL_PROB_ONLY:
            assert self.cfg.MODEL.PRECOMPUTED_BALANCED_NORM_PATH != ''
            self.precomputed_labeling_prob = torch.load(self.cfg.MODEL.PRECOMPUTED_BALANCED_NORM_PATH).cuda().float()
            # if self.cfg.MODEL.BALANCED_NORM_FIXED_NORMALIZED:
            #     self.precomputed_labeling_prob /= sum(self.precomputed_labeling_prob)
        
        # training time balanced_norm
        if self.cfg.MODEL.BALANCED_NORM:
            if self.cfg.MODEL.BALANCED_NORM_FIXED:
                assert self.cfg.MODEL.PRECOMPUTED_BALANCED_NORM_PATH != ''
                self.precomputed_labeling_prob = torch.load(self.cfg.MODEL.PRECOMPUTED_BALANCED_NORM_PATH).cuda().float()
                if self.cfg.MODEL.BALANCED_NORM_FIXED_NORMALIZED:
                    self.precomputed_labeling_prob /= sum(self.precomputed_labeling_prob)
            elif self.cfg.MODEL.BALANCED_NORM_LEARNABLE:
                self.balanced_norm = LearnableBalancedNorm1d(51, normalized_probs=self.cfg.MODEL.BALANCED_NORM_NORMALIZED_PROBS) # VG has 50 (fg) + 1 (bg) classes
            else:
                self.balanced_norm = BalancedNorm1d(51, normalized_probs=self.cfg.MODEL.BALANCED_NORM_NORMALIZED_PROBS, with_gradient=self.cfg.MODEL.BALANCED_NORM_WITH_GRADIENT) # VG has 50 (fg) + 1 (bg) classes
        elif self.cfg.MODEL.BALANCED_NORM_FIXED and self.cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL:
            assert self.cfg.MODEL.PRECOMPUTED_BALANCED_NORM_PATH != ''
            self.precomputed_labeling_prob = torch.load(self.cfg.MODEL.PRECOMPUTED_BALANCED_NORM_PATH).cuda().float()
            if self.cfg.MODEL.BALANCED_NORM_FIXED_NORMALIZED:
                self.precomputed_labeling_prob /= sum(self.precomputed_labeling_prob)

        if self.cfg.MODEL.PCPL_CENTER_LOSS:
            # Currently only implemented for Motifs and VCTree models
            assert (self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ['MotifPredictor', 'VCTreePredictor']) or (self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == 'CausalAnalysisPredictor' and self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER in ['motifs', 'vctree'])

        # Multi-label training 
        self.multi_label_training = cfg.TRAIN.MULTI_LABEL_TRAINING or cfg.HMC.C_HMC_TEST
        self.matrix_of_ancestor = None
        if self.multi_label_training:
            if cfg.HMC.C_HMC_TRAIN or cfg.HMC.C_HMC_TEST:
                assert cfg.HMC.ANCESTOR_MAT_PATH != '', 'Error: ancestor matrix\'s path cannot be empty!'
                with open(cfg.HMC.ANCESTOR_MAT_PATH, 'rb') as f:
                    self.matrix_of_ancestor = pickle.load(f) # f

                self.matrix_of_ancestor = torch.tensor(self.matrix_of_ancestor)
                #Transpose to get the ancestors for each node 
                self.matrix_of_ancestor = self.matrix_of_ancestor.transpose(1, 0)
                self.matrix_of_ancestor = self.matrix_of_ancestor.unsqueeze(0).to(torch.device('cuda'))

    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys, stl_labels = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys, stl_labels = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_binarys, stl_labels = None, None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)

        # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, proposals)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None
        
        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)

        relation_logits_raw = None
        if self.cfg.MODEL.PCPL_CENTER_LOSS:
            # relation_logits_raw of shape MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
            relation_logits, relation_logits_raw = relation_logits

        # training time balanced_norm
        rel_labels_one_hot_count = None
        if self.cfg.MODEL.BALANCED_NORM_FIXED and self.cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL:
            relation_probs_norm = None
            labeling_prob = self.precomputed_labeling_prob
        elif self.cfg.MODEL.BALANCED_NORM:
            if self.cfg.MODEL.BALANCED_NORM_FIXED:
                labeling_prob = self.precomputed_labeling_prob
                if not self.cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL:
                    relation_logits_temp = cat(relation_logits) if isinstance(relation_logits, tuple) else relation_logits
                    relation_probs_norm = F.softmax(relation_logits_temp, dim=-1) / labeling_prob
            else:
                # relation_logits = self.balanced_norm(
                #     cat(relation_logits, dim=0), 
                #     cat(rel_pair_idxs, dim=0),
                #     cat(rel_labels, dim=0)
                # )
                # if self.cfg.DEBUG:
                #     import pdb; pdb.set_trace()
                relation_probs_norm, labeling_prob, rel_labels_one_hot_count = self.balanced_norm(relation_logits, rel_labels)
                # if rel_labels_one_hot_count is not None:
                #     import pdb; pdb.set_trace()
                
        elif self.cfg.TEST.WITH_LABEL_PROB_ONLY:
            relation_probs_norm = None
            labeling_prob = self.precomputed_labeling_prob
        else:
            relation_probs_norm = labeling_prob = None
        
        if self.cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL and self.cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL_NORMALIZE_LB:
            assert not self.cfg.MODEL.BALANCED_NORM_FIXED # DO NOT USE TOGETHER!
            labeling_prob = labeling_prob.detach().clone()
            labeling_prob /= sum(labeling_prob)

        # for test
        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals, relation_probs_norm, labeling_prob, self.matrix_of_ancestor)
            return roi_features, result, {}, None, None

        loss_relation, loss_refine, loss_relation_stl, loss_center, loss_gx, loss_avg_belief, rel_features, rel_targets = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits, stl_labels, relation_probs_norm, relation_logits_raw, rel_pair_idxs, labeling_prob, self.matrix_of_ancestor)

        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        if loss_relation_stl is not None:
            output_losses['loss_relation_stl'] = loss_relation_stl

        if loss_center is not None:
            output_losses['loss_center'] = loss_center

        if loss_gx is not None:
            output_losses['loss_gx'] = loss_gx

        if loss_avg_belief is not None:
            output_losses['loss_avg_belief'] = loss_avg_belief

        if rel_labels_one_hot_count is not None:
            output_losses['rel_labels_one_hot_count'] = rel_labels_one_hot_count

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses, rel_features, rel_targets


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
