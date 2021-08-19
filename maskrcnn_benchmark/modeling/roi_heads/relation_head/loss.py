# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression, CenterLoss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat, get_constr_out

import itertools
import pickle
import scipy, math
from scipy import optimize

def soft_cross_entropy_loss(x, target, reduction='mean'):
    assert reduction in ['sum', 'mean']
    logprobs = torch.nn.functional.log_softmax(x, dim = 1)
    loss = -(target * logprobs).sum()
    return loss if reduction == 'sum' else (loss / x.shape[0])

# for center loss
# https://github.com/louis-she/center-loss.pytorch/blob/master/loss.py
def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss

def weighted_mse_loss(input, target, weight=None, reduction='mean'):
    if weight is None:
        loss = torch.sum((input - target) ** 2)
    else:
        loss = torch.sum(weight * (input - target) ** 2)
    return loss if reduction == 'sum' else (loss / input.shape[0])

def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim = 1)
    return -(target * logprobs).sum() / input.shape[0]

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        stl_train,
        loss_relation_stl_alpha,
        bg_soft_loss,
        use_balanced_norm,
        pcpl_center_loss,
        center_loss_lambda,
        num_classes,
        feat_dim,
        unbiased_training,
        predicate_weights_path,
        weight_factor,
        use_gt_box,
        balnorm_reweight_label_distrib,
        balnorm_reweight_label_prob,
        balnorm_reweight_inv_label_prob,
        norm_predicate_counts_path,
        bal_reweight_only,
        balanced_norm_train_gx,
        balanced_norm_learnable,
        balanced_norm_normalized_probs,
        balanced_norm_test_only,
        balanced_norm_fixed,
        balanced_norm_fixed_mseloss,
        balanced_norm_as_soft_label,
        balanced_norm_as_soft_label_sll,
        balanced_norm_as_soft_label_mll,
        train_avg_belief_to_one,
        train_avg_belief_to_one_only_this_loss,
        balanced_norm_use_bceloss,
        balanced_norm_use_mseloss,
        multi_label_training,
        multi_label_norm_loss,
        c_hmc_train,
        debug,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.stl_train = stl_train
        self.bg_soft_loss = bg_soft_loss
        self.debug = debug

        # BalNorm
        self.use_balanced_norm = False if balanced_norm_test_only else use_balanced_norm

        # PCPL & Center Loss
        self.pcpl_center_loss = pcpl_center_loss
        self.num_classes = num_classes

        self.bal_reweight_only = bal_reweight_only

        # BalNorm + Reweighting by labeling probability
        self.balnorm_reweight_label_prob = balnorm_reweight_label_prob
        if self.balnorm_reweight_label_prob:
            assert unbiased_training == 'reweight' # make sure this is enabled!

        # BalNorm + Reweighting by inverse labeling probability
        self.balnorm_reweight_inv_label_prob = balnorm_reweight_inv_label_prob
        if self.balnorm_reweight_inv_label_prob:
            assert unbiased_training == 'reweight' # make sure this is enabled!

        # BalNorm + Reweighting by estimated label distrib.
        self.balnorm_reweight_label_distrib = balnorm_reweight_label_distrib
        if self.balnorm_reweight_label_distrib:
            assert unbiased_training == 'reweight' # make sure this is enabled!
        self.norm_predicate_counts_path = norm_predicate_counts_path

        # Whether to train against g(x) in addition to f(x), against the training set p(s|x)
        if balanced_norm_train_gx:
            assert use_balanced_norm
        self.balanced_norm_train_gx = balanced_norm_train_gx

        # whether to use learnable BalNorm
        if balanced_norm_learnable:
            assert use_balanced_norm
        self.balanced_norm_learnable = balanced_norm_learnable

        if balanced_norm_normalized_probs:
            assert use_balanced_norm
        self.balanced_norm_normalized_probs = balanced_norm_normalized_probs

        self.balanced_norm_fixed = balanced_norm_fixed
        if balanced_norm_fixed_mseloss:
            assert use_balanced_norm
        self.balanced_norm_fixed_mseloss = balanced_norm_fixed_mseloss

        self.balanced_norm_use_bceloss = balanced_norm_use_bceloss
        self.balanced_norm_use_mseloss = balanced_norm_use_mseloss

        # multi-label training
        self.multi_label_training = multi_label_training
        if self.multi_label_training:
            self.multi_label_norm_loss = multi_label_norm_loss
            self.c_hmc_train = c_hmc_train

        if balanced_norm_as_soft_label:
            if not self.balanced_norm_fixed:
                assert use_balanced_norm
            assert balanced_norm_as_soft_label_sll ^ balanced_norm_as_soft_label_mll
        self.balanced_norm_as_soft_label = balanced_norm_as_soft_label
        self.balanced_norm_as_soft_label_sll = balanced_norm_as_soft_label_sll
        self.balanced_norm_as_soft_label_mll = balanced_norm_as_soft_label_mll

        if train_avg_belief_to_one:
            assert use_balanced_norm
        self.train_avg_belief_to_one = train_avg_belief_to_one
        self.train_avg_belief_to_one_only_this_loss = train_avg_belief_to_one_only_this_loss

        self.weight = None
        self.unbiased_training = unbiased_training
        self.use_gt_box = use_gt_box
        if unbiased_training != '':
            assert not self.pcpl_center_loss
            if unbiased_training == 'reweight':
                if self.multi_label_training:
                    with open(predicate_weights_path, 'rb') as f:
                        self.weight = pickle.load(f)
                    # import pdb; pdb.set_trace()
                    assert len(self.weight) == 51
                elif balnorm_reweight_label_prob or balnorm_reweight_inv_label_prob: # Type 0: reweighting by estimated labeling prob (only used with Balnorm)
                    pass # do nothing here (labeling prob. is estimated & passed dynamically)
                elif self.balnorm_reweight_label_distrib: # Type I: reweighting by estimated label distrib. (only used with Balnorm)
                    with open(self.norm_predicate_counts_path, 'rb') as f:
                        self.norm_predicate_counts = pickle.load(f) # f
                    assert len(self.norm_predicate_counts) == 50
                elif self.bal_reweight_only: # Type II: reweight by estimated labeling probability (1/p)
                    pass # do nothing here
                else: # Type III: normal reweighting
                    assert predicate_weights_path.split('.')[-1] == 'pkl'
                    with open(predicate_weights_path, 'rb') as f:
                        self.weight = pickle.load(f)
                    assert len(self.weight) == 51
            elif unbiased_training == 'reweight_vrd':
                assert predicate_weights_path.split('.')[-1] == 'npy'
                with open(predicate_weights_path, 'rb') as f:
                    self.weight = np.load(f)
                assert self.weight.shape[0] == 151 and self.weight.shape[1] == 151 and self.weight.shape[2] == 51
            elif unbiased_training == 'resample':
                raise NotImplementedError
            elif unbiased_training == 'focal_loss':
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            if not self.balnorm_reweight_label_distrib and not self.bal_reweight_only and not self.balnorm_reweight_label_prob and not self.balnorm_reweight_inv_label_prob:
                self.weight = torch.tensor(self.weight).cuda() * weight_factor
        elif self.pcpl_center_loss:
            assert not self.use_label_smoothing # not implemented for use with label smoothing
            self.center_loss_lambda = center_loss_lambda
            
            # Center Loss version 1: https://github.com/KaiyangZhou/pytorch-center-loss
            # self.center_loss = CenterLoss(num_classes - 1, feat_dim)

            # Center loss version 2: https://github.com/louis-she/center-loss.pytorch
            self.centers = nn.Parameter(torch.Tensor(num_classes, feat_dim).normal_(), requires_grad=False).cuda()

            self.corr_order = torch.tensor([(i, j) for i in range(num_classes) for j in range(num_classes)]) 

        # For PCPL we put the weight in F.cross_entropy(weight=weight) as the weight
        # is dynamically changing
        if self.stl_train:
            if self.weight is not None:
                raise NotImplementedError # haven't implemented
            self.criterion_loss = nn.CrossEntropyLoss()
            self.loss_relation_stl_alpha = loss_relation_stl_alpha
        elif self.use_label_smoothing:
            assert self.weight is None # not to be used with other reweighting methods
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        # elif self.use_balanced_norm:
        #     if self.unbiased_training != '':
        #         assert self.weight is not None
        #     self.loss_relation_balanced_norm = nn.NLLLoss(weight=self.weight)
        #     self.criterion_loss = nn.CrossEntropyLoss()
        else:
            self.criterion_loss = nn.CrossEntropyLoss()
            if self.multi_label_training:
                # import pdb; pdb.set_trace()
                if self.c_hmc_train:
                    self.criterion_loss_relation = nn.BCELoss(weight=self.weight)
                else:
                    self.criterion_loss_relation = nn.BCEWithLogitsLoss(weight=self.weight)
            else:
                self.criterion_loss_relation = nn.CrossEntropyLoss(weight=self.weight)
        
        if self.use_balanced_norm and not self.balnorm_reweight_label_distrib and not self.bal_reweight_only and not self.balnorm_reweight_label_prob and not self.balanced_norm_normalized_probs and not self.balnorm_reweight_inv_label_prob and not self.balanced_norm_as_soft_label:
            if self.unbiased_training != '':
                assert self.weight is not None
            if self.balanced_norm_use_bceloss:
                assert self.weight is None # NOT IMPLEMENTED TO RUN TOGETHER YET
                self.loss_relation_balanced_norm = nn.BCELoss()
            elif self.balanced_norm_use_mseloss:
                assert self.weight is None # NOT IMPLEMENTED TO RUN TOGETHER YET
                self.loss_relation_balanced_norm = nn.MSELoss()
            else:
                self.loss_relation_balanced_norm = nn.NLLLoss(weight=self.weight)

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, stl_labels=None, relation_probs_norm=None, relation_logits_raw=None, rel_pair_idxs=None, labeling_prob=None, matrix_of_ancestor=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        weight = self.weight
        if self.unbiased_training == 'reweight_vrd': 
            raise NotImplementedError # not completed yet
            assert not self.pcpl_center_loss # cannot be used together
            weights = []
            for refine_obj_logit, rel_pair_idx in zip(refine_obj_logits, rel_pair_idxs):
                obj_class_prob = F.softmax(refine_obj_logit, -1)

                if self.use_gt_box:
                    _, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                    obj_pred = obj_pred + 1
                else:
                    raise NotImplementedError # haven't implemented for SGDet
                    # NOTE: by kaihua, apply late nms for object prediction
                    obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres)
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                
                sub_cls = obj_pred[rel_pair_idx[:, 0]]
                obj_cls = obj_pred[rel_pair_idx[:, 1]]
                weight = self.weight[sub_cls, obj_cls]
                weights.append(weight)
            weights = torch.cat(weights)

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)
        rel_labels = cat(rel_labels, dim=0)
        fg_idxs, bg_idxs = (rel_labels != 0), (rel_labels == 0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        
        assert not (self.stl_train and self.bg_soft_loss), 'Simultaneous usage of stl_train and bg_soft_loss are not supported!'
        assert not (self.use_balanced_norm and self.bg_soft_loss), 'Simultaneous usage of use_balanced_norm and bg_soft_loss are not supported!'

        loss_center = loss_gx = rel_features = rel_targets = loss_avg_belief = None
        if self.pcpl_center_loss:
            assert relation_logits_raw is not None
            # compute center loss
            # Way 1
            # loss_center = self.center_loss(relation_logits_raw[fg_idxs].detach().clone(), rel_labels.long()[fg_idxs]) # only compute loss for non-bg classes
            # loss_center = self.center_loss(relation_logits_raw.detach().clone(), rel_labels.long()) # also compute loss for bg class (class 0)
            # Way 2
            rel_features = relation_logits_raw.clone().detach()
            loss_center = compute_center_loss(rel_features, self.centers, rel_labels.long()) * self.center_loss_lambda
            rel_targets = rel_labels.long()

            # (eq. 2) compute e_{kj}
            corr = torch.norm((self.centers[self.corr_order[:,0]] - self.centers[self.corr_order[:,1]]), dim=1)
            # (eq. 3 compute u_i)
            global_corr = torch.cat([(torch.sum(corr_class, dim=0) / self.num_classes).reshape(-1) for corr_class in torch.split(corr, self.num_classes)])
            # (eq. 4 compute correlation factor tao_i as weight)
            eps = 0.09
            max_corr, min_corr = max(global_corr), min(global_corr)
            corr_factor = (global_corr - min_corr + eps) / (max_corr - min_corr)
            weight = corr_factor.detach()

        if self.stl_train:
            assert stl_labels is not None
            stl_labels = cat(stl_labels, dim=0)

            if self.balanced_norm_train_gx:
                loss_gx = F.cross_entropy(relation_logits, rel_labels.long())
            
            if self.pcpl_center_loss:
                loss_relation = F.cross_entropy(relation_logits[fg_idxs], rel_labels.long()[fg_idxs], weight=weight)
            elif self.bal_reweight_only:
                # use estimated labeling probabilitiy for reweighting
                assert labeling_prob is not None
                import pdb; pdb.set_trace()
            elif self.balnorm_reweight_label_prob:
                assert labeling_prob is not None
                
                with torch.no_grad():
                    weight = 1 / labeling_prob.detach()
                loss_relation = F.nll_loss(torch.log(relation_probs_norm), rel_labels.long(), weight=weight)
            elif self.balnorm_reweight_label_distrib:
                # estimate label distribution
                assert labeling_prob is not None
                
                with torch.no_grad():
                    f = self.norm_predicate_counts
                    q = np.array(labeling_prob[1:].detach().cpu())
                    a = np.array([[qq * (ff - 1) if i == j else qq * ff for i, qq in enumerate(q)] for j, ff in enumerate(f)])
                    a[-1] = np.array([1] * len(a))
                    b = np.array([0] * 49 + [1])
                    p = scipy.optimize.lsq_linear(a, b, (0, float('inf')))
                    p = p.x

                    # counts = np.array([0.0] * 51)
                    # counts[0] = 3.0
                    # counts[1:] = p
                    weight = torch.tensor([math.sqrt(1/p[idx - 1]) if idx != 0 else math.sqrt(1/(3*sum(p))) for idx in range(len(p) + 1)]).cuda()
                
                loss_relation = F.nll_loss(torch.log(relation_probs_norm[fg_idxs]), rel_labels.long()[fg_idxs], weight=weight)
            elif self.use_balanced_norm:
                loss_relation = self.loss_relation_balanced_norm(torch.log(relation_probs_norm[fg_idxs]), rel_labels.long()[fg_idxs])
                # loss_relation = self.criterion_loss(relation_logits[fg_idxs], rel_labels.long()[fg_idxs])
            else:
                loss_relation = self.criterion_loss(relation_logits[fg_idxs], rel_labels.long()[fg_idxs])
            assert relation_logits[bg_idxs].shape == stl_labels.shape
            loss_relation_stl = soft_cross_entropy_loss(relation_logits[bg_idxs], stl_labels, reduction='mean') * self.loss_relation_stl_alpha
            
            # loss_relation = (loss_relation + loss_relation_stl * self.loss_relation_stl_alpha) / len(relation_logits)
            # loss_relation_stl = None # set to None (for now) as it should be combined into loss_relation. Delete this variable if it's correct to do so.
        else:
            if self.use_balanced_norm:
                assert relation_probs_norm is not None
                if self.balanced_norm_train_gx:
                    loss_gx = F.cross_entropy(relation_logits, rel_labels.long())

                if self.balanced_norm_as_soft_label:
                    # import pdb; pdb.set_trace()
                    if self.balanced_norm_as_soft_label_sll:
                        logprobs = F.log_softmax(relation_logits, dim=1)[range(relation_logits.shape[0]), rel_labels]
                        loss_relation = -(labeling_prob.detach()[rel_labels] * logprobs).sum() / logprobs.shape[0]
                    elif self.balanced_norm_as_soft_label_mll:
                        loss_relation = softXEnt(relation_logits, labeling_prob.detach())
                    else:
                        raise NotImplementedError
                elif self.pcpl_center_loss:
                    raise NotImplementedError
                    loss_relation = F.nll_loss(torch.log(relation_probs_norm), rel_labels.long())
                elif self.bal_reweight_only:
                    # use estimated labeling probabilitiy for reweighting
                    assert labeling_prob is not None
                    
                    loss_relation = F.cross_entropy(relation_logits, rel_labels.long(), weight=(1 / labeling_prob))
                elif self.balnorm_reweight_inv_label_prob:
                    assert labeling_prob is not None

                    if self.balanced_norm_normalized_probs:
                        weight = labeling_prob
                        
                        loss_relation = weighted_mse_loss(relation_probs_norm[range(relation_probs_norm.shape[0]), rel_labels], torch.ones(len(rel_labels)).cuda(), weight=weight[rel_labels])
                    else:
                        loss_relation = F.nll_loss(torch.log(relation_probs_norm), rel_labels.long(), weight=labeling_prob.detach())
                elif self.balnorm_reweight_label_prob:
                    assert labeling_prob is not None

                    if self.balanced_norm_normalized_probs:
                        # import pdb; pdb.set_trace()
                        weight = 1 / labeling_prob
                        
                        loss_relation = weighted_mse_loss(relation_probs_norm[range(relation_probs_norm.shape[0]), rel_labels], torch.ones(len(rel_labels)).cuda(), weight=weight[rel_labels])
                    else:
                        # if self.balanced_norm_learnable: # use label prob as weight with gradient
                        #     weight = 1 / labeling_prob
                        # else:
                        with torch.no_grad():
                            weight = 1 / labeling_prob.detach()
                        loss_relation = F.nll_loss(torch.log(relation_probs_norm), rel_labels.long(), weight=weight)
                elif self.balnorm_reweight_label_distrib:
                    # estimate label distribution
                    assert labeling_prob is not None
                    
                    with torch.no_grad():
                        f = self.norm_predicate_counts
                        q = np.array(labeling_prob[1:].detach().cpu())
                        a = np.array([[qq * (ff - 1) if i == j else qq * ff for i, qq in enumerate(q)] for j, ff in enumerate(f)])
                        a[-1] = np.array([1] * len(a))
                        b = np.array([0] * 49 + [1])
                        p = scipy.optimize.lsq_linear(a, b, (0, float('inf')))
                        p = p.x

                        # counts = np.array([0.0] * 51)
                        # counts[0] = 3.0
                        # counts[1:] = p
                        weight = torch.tensor([math.sqrt(1/p[idx - 1]) if idx != 0 else math.sqrt(1/(3*sum(p))) for idx in range(len(p) + 1)]).cuda()
                    loss_relation = F.nll_loss(torch.log(relation_probs_norm), rel_labels.long(), weight=weight)
                else: # use balanced norm
                    if self.balanced_norm_normalized_probs or self.balanced_norm_fixed_mseloss:
                        # import pdb; pdb.set_trace()
                        loss_relation = weighted_mse_loss(relation_probs_norm[range(relation_probs_norm.shape[0]), rel_labels], torch.ones(len(rel_labels)).cuda())
                    else:
                        if self.train_avg_belief_to_one:
                            nonzero_idxs = torch.nonzero(rel_labels)
                            # n_unique_cls = len(torch.unique(rel_labels[nonzero_idxs]))
                            
                            unique_class_to_example_idxs = {obj_cls:[] for obj_cls in torch.unique(rel_labels[nonzero_idxs]).detach().cpu().numpy().tolist()}
                            for i in range(relation_probs_norm.shape[0]):
                                obj_cls = rel_labels[i].detach().cpu().item()
                                if obj_cls != 0:
                                    unique_class_to_example_idxs[obj_cls].append(i)
                            
                            avg_beliefs = []
                            for obj_cls in sorted(unique_class_to_example_idxs.keys()):
                                avg_belief = relation_probs_norm[unique_class_to_example_idxs[obj_cls]].sum(0) / len(relation_probs_norm[unique_class_to_example_idxs[obj_cls]])
                                avg_beliefs.append(avg_belief.view(1, -1))
                            avg_beliefs = torch.cat(avg_beliefs)
                            
                            # import pdb; pdb.set_trace()
                            loss_avg_belief = F.nll_loss(torch.log(avg_beliefs), torch.unique(rel_labels[nonzero_idxs]))
                            if self.train_avg_belief_to_one_only_this_loss:
                                loss_relation = loss_avg_belief
                                loss_avg_belief = None
                        if not self.train_avg_belief_to_one_only_this_loss:
                            if self.balanced_norm_use_bceloss:
                                import pdb; pdb.set_trace()
                                loss_relation = self.loss_relation_balanced_norm(relation_probs_norm[range(relation_probs_norm.shape[0]), rel_labels], torch.ones(len(rel_labels)).cuda())
                            elif self.balanced_norm_use_mseloss:
                                # loss_relation = self.loss_relation_balanced_norm(relation_probs_norm[range(relation_probs_norm.shape[0]), rel_labels], torch.ones(len(rel_labels)).cuda())
                                import pdb; pdb.set_trace()
                                loss_relation = self.loss_relation_balanced_norm(relation_probs_norm, rel_labels)
                            else:
                                loss_relation = self.loss_relation_balanced_norm(torch.log(relation_probs_norm), rel_labels.long())
            elif self.balanced_norm_fixed and self.balanced_norm_as_soft_label:
                if self.balanced_norm_as_soft_label_sll:
                    logprobs = F.log_softmax(relation_logits, dim=1)[range(relation_logits.shape[0]), rel_labels]
                    loss_relation = -(labeling_prob.detach()[rel_labels] * logprobs).sum() / logprobs.shape[0]
                elif self.balanced_norm_as_soft_label_mll:
                    loss_relation = softXEnt(relation_logits, labeling_prob.detach())
                else:
                    raise NotImplementedError
            elif self.bg_soft_loss:
                raise NotImplementedError
                hard_loss_fg_relation = self.criterion_loss(relation_logits[:len(fg_labels)], rel_labels.long()[:len(fg_labels)])
                soft_loss_bg_relation = soft_cross_entropy_loss(relation_logits[len(fg_labels):], rel_labels.float()[len(fg_labels):, None], reduction='mean')
                # Note that this section is not implemented for self.pcpl_center_loss
                loss_relation = hard_loss_fg_relation + soft_loss_bg_relation
            elif self.pcpl_center_loss:
                loss_relation = F.cross_entropy(relation_logits, rel_labels.long(), weight=weight)
            elif self.unbiased_training == 'reweight_vrd':
                raise NotImplementedError
            else:
                if self.multi_label_training:
                    if self.c_hmc_train:
                        # import pdb; pdb.set_trace()
                        assert isinstance(self.criterion_loss_relation, nn.BCELoss) # sanity check
                        relation_logits = torch.sigmoid(relation_logits)
                        constr_output = get_constr_out(relation_logits, matrix_of_ancestor)
                        train_output = rel_labels * relation_logits.double()
                        train_output = get_constr_out(train_output, matrix_of_ancestor)
                        relation_logits = (1 - rel_labels) * constr_output.double() + rel_labels * train_output
                    else:
                        assert isinstance(self.criterion_loss_relation, nn.BCEWithLogitsLoss) # sanity check
                    
                    if self.multi_label_norm_loss:
                        rel_labels = rel_labels / rel_labels.sum(dim=1).view(-1, 1)
                    loss_relation = self.criterion_loss_relation(relation_logits, rel_labels.double()).float()
                    
                    # if self.multi_label_norm_loss:
                    #     if rel_labels.sum() < rel_labels.shape[0]:
                    #         import pdb; pdb.set_trace()
                    #         nonzero_idxs = rel_labels.sum(dim=1).nonzero()
                    #         rel_labels = rel_labels[nonzero_idxs].squeeze()
                    #         loss_relation = loss_relation[nonzero_idxs].squeeze()
                    #     # elif rel_labels.sum() > rel_labels.shape[0]:
                    #     import pdb; pdb.set_trace()
                    #     loss_relation = (loss_relation / rel_labels.sum(dim=1).view(-1, 1)).sum() / loss_relation.shape[0]
                else:
                    loss_relation = self.criterion_loss_relation(relation_logits, rel_labels.long())
            loss_relation_stl = None
        
        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att), loss_relation_stl, loss_center, loss_gx, loss_avg_belief, rel_features, rel_targets
        else:
            return loss_relation, loss_refine_obj, loss_relation_stl, loss_center, loss_gx, loss_avg_belief, rel_features, rel_targets

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cfg.MODEL.STL_TRAIN,
        cfg.MODEL.STL_TRAIN_LOSS_ALPHA,
        cfg.MODEL.BG_SOFT_LOSS,
        cfg.MODEL.BALANCED_NORM,
        cfg.MODEL.PCPL_CENTER_LOSS,
        cfg.MODEL.CENTER_LOSS_LAMBDA,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES,
        cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM,
        cfg.TRAIN.UNBIASED_TRAINING,
        cfg.TRAIN.PREDICATE_WEIGHTS_PATH,
        cfg.TRAIN.WEIGHT_FACTOR,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        cfg.TRAIN.BALNORM_REWEIGHT_LABEL_DISTRIB,
        cfg.TRAIN.BALNORM_REWEIGHT_LABEL_PROB,
        cfg.TRAIN.BALNORM_REWEIGHT_INV_LABEL_PROB,
        cfg.TRAIN.NORM_PREDICATE_COUNTS_PATH,
        cfg.TRAIN.BAL_REWEIGHT_ONLY,
        cfg.MODEL.BALANCED_NORM_TRAIN_GX,
        cfg.MODEL.BALANCED_NORM_LEARNABLE,
        cfg.MODEL.BALANCED_NORM_NORMALIZED_PROBS,
        cfg.MODEL.BALANCED_NORM_TEST_ONLY,
        cfg.MODEL.BALANCED_NORM_FIXED,
        cfg.MODEL.BALANCED_NORM_FIXED_MSELOSS,
        cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL,
        cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL_SLL,
        cfg.MODEL.BALANCED_NORM_AS_SOFT_LABEL_MLL,
        cfg.MODEL.TRAIN_AVG_BELIEF_TO_ONE,
        cfg.MODEL.TRAIN_AVG_BELIEF_TO_ONE_ONLY_THIS_LOSS,
        cfg.MODEL.BALANCED_NORM_USE_BCELOSS,
        cfg.MODEL.BALANCED_NORM_USE_MSELOSS,
        cfg.TRAIN.MULTI_LABEL_TRAINING,
        cfg.TRAIN.MULTI_LABEL_NORM_LOSS,
        cfg.HMC.C_HMC_TRAIN,
        cfg.DEBUG,
    )

    return loss_evaluator
