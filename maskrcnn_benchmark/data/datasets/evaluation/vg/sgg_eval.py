import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps

from abc import ABC, abstractmethod
from collections import defaultdict
import heapq

def lca_to_gt_dist_ratio(g, r_k, hier, hier_paths):
    assert hier is not None
    dist_ratios = []
    for r in r_k:
        i = 0
        while i < len(hier_paths[g]) and i < len(hier_paths[r]) and hier_paths[g][i] == hier_paths[r][i]:
            i += 1
        dist_ratio = i / len(hier_paths[g])
        assert dist_ratio <= 1 # sanity check
        dist_ratios.append(dist_ratio)
    return np.array(dist_ratios)

def semantic_similarity(g, r_k, rel_hier, rel_hier_paths, use_obj_hier, obj_hier, obj_hier_paths):
    '''
    Return the semantic similarities between a GT triplet, g_i, and a set of predicted triplets, r_k
    '''
    phi_p = lca_to_gt_dist_ratio(g, [r['rel_idx'] for r in r_k], rel_hier, rel_hier_paths)
    if use_obj_hier:
        raise NotImplementedError
        phi_s = lca_to_gt_dist_ratio(g, [r['sub_idx'] for r in r_k], obj_hier, obj_hier_paths)
        phi_o = lca_to_gt_dist_ratio(g, [r['obj_idx'] for r in r_k], obj_hier, obj_hier_paths)
        return (phi_s + phi_p + phi_o) / 3
    else:
        return phi_p

def max_semantic_similarity(g_i, r_k, rel_hier, rel_hier_paths, use_obj_hier, obj_hier, obj_hier_paths):
    '''
    Return the maximum semantic similarity between a GT triplet, g_i, and a set of predicted triplets, r_k
    '''
    hier_recalls = []
    for g in g_i:
        sem_sims = semantic_similarity(g, r_k, rel_hier, rel_hier_paths, use_obj_hier, obj_hier, obj_hier_paths)
        hier_recalls.append(max(sem_sims))
    return hier_recalls


class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass
    
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, comp_labeling_prob, comp_label_prob_for_sgdet=False, comp_cogtree=False, test_with_cogtree=False, cogtree_file=None):
        super(SGRecall, self).__init__(result_dict)
        self.comp_labeling_prob = comp_labeling_prob
        self.comp_label_prob_for_sgdet = comp_label_prob_for_sgdet
        self.comp_cogtree = comp_cogtree
        self.test_with_cogtree = test_with_cogtree
        if test_with_cogtree: 
            assert cogtree_file is not None
            self.cogtree_file = cogtree_file

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        # Compute recall. It's most efficient to match once and then do recall after
        # TODO:
        # duplicate gt_triplets with concept predicates in _compute_pred_matches()
        # but keep the denominator unchanged when computing recall@X
        if self.test_with_cogtree: 
            assert not self.comp_label_prob_for_sgdet # not compatible for now

            gt_triplet_idxs_to_concept_idxs = defaultdict(list)
            n_gt_triplets = concept_idx = len(gt_triplets)
            concept_gt_triplets, concept_gt_boxes, concept_gt_rels = [], [], []
            for idx, (gt_triplet, gt_rel) in enumerate(zip(gt_triplets, gt_rels)):
                for concept in self.cogtree_file[gt_triplet[1]]:
                    if concept != gt_triplet[1]:
                        gt_triplet_idxs_to_concept_idxs[idx].append(concept_idx)
                        concept_idx += 1

                        concept_gt_rels.append([[gt_rel[0], gt_rel[1], concept]])
                        concept_gt_triplets.append([[gt_triplet[0], concept, gt_triplet[2]]])
                        concept_gt_boxes.append([gt_triplet_boxes[idx]])
                        
            if len(concept_gt_triplets) > 0: # at least one concept triplet generated
                concept_gt_triplets = np.concatenate(concept_gt_triplets)
                concept_gt_boxes = np.concatenate(concept_gt_boxes)
                concept_gt_rels = np.concatenate(concept_gt_rels)

                gt_triplets = np.concatenate([gt_triplets, concept_gt_triplets])
                gt_triplet_boxes = np.concatenate([gt_triplet_boxes, concept_gt_boxes])
                gt_rels = np.concatenate([gt_rels, concept_gt_rels])
            
            local_container['n_gt_triplets'] = n_gt_triplets
            local_container['gt_rels'] = gt_rels
            local_container['gt_triplet_idxs_to_concept_idxs'] = gt_triplet_idxs_to_concept_idxs
        else:
            local_container['n_gt_triplets'] = None
            local_container['gt_triplet_idxs_to_concept_idxs'] = None

        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
            test_with_cogtree=self.test_with_cogtree,
            n_gt_triplets=local_container['n_gt_triplets'], 
            gt_triplet_idxs_to_concept_idxs=local_container['gt_triplet_idxs_to_concept_idxs'],
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            if self.test_with_cogtree:
                rec_i = float(len(match)) / float(local_container['n_gt_triplets'])
            else:
                rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)
        
        pred_predicates = None
        if self.comp_cogtree:
            pred_rel_inds_to_idx = {tuple(pred_rel_ind):idx for idx, pred_rel_ind in enumerate(pred_rel_inds)}
            gt_rel_inds = [tuple(gt_rel[:2]) for gt_rel in gt_rels]
            try:
                idxs = [pred_rel_inds_to_idx[gt_rel_ind] for gt_rel_ind in gt_rel_inds]
                pred_predicates = [(gt_rels[jdx, 2], pred_triplets[idx, 1]) for jdx, idx in enumerate(idxs)] # a list of tuples of (gt_rel, pred_rel)
            except:
                print('Inconsistent indices happened! Skip this img...')

        if self.comp_label_prob_for_sgdet:
            # import pdb; pdb.set_trace()
            # pred_idx_to_gt_idxs = [(idx, i[0]) for idx, i in enumerate(pred_to_gt) if i]
            # pred_labels = [gt_rels[gt_idx][2] for _, gt_idx in pred_idx_to_gt_idxs]
            # pred_scores = rel_scores[[pred_idx for pred_idx, _ in pred_idx_to_gt_idxs], pred_labels]
            
            # Check subject & object IOU >= 0.5
            sub_iou = bbox_overlaps(gt_triplet_boxes[:, :4], pred_triplet_boxes[:, :4])
            obj_iou = bbox_overlaps(gt_triplet_boxes[:, 4:], pred_triplet_boxes[:, 4:])
            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)
            # (Pdb) inds.nonzero()
            # (array([0, 0, 0, 2, 2, 2, 3, 3, 3, 4, 4]), array([   0,  137, 1739,    5,  171,  427,   29,  496,  883,    7,   14]))

            # Check subject & object classes
            gt_sub_obj = np.concatenate([gt_triplets[:, 0].reshape(-1, 1), gt_triplets[:, 1].reshape(-1, 1)], axis=1)
            pred_sub_obj = np.concatenate([pred_triplets[:, 0].reshape(-1, 1), pred_triplets[:, 1].reshape(-1, 1)], axis=1)
            inds = inds & intersect_2d(gt_sub_obj, pred_sub_obj)
            # (Pdb) (inds & intersect_2d(gt_sub_obj, pred_sub_obj)).nonzero()
            # (array([0, 0, 0, 2, 3]), array([   0,  137, 1739,    5,   29]))
            
            # Remove repeated predicted pairs
            # import pdb; pdb.set_trace()
            # _, non_repeat_idxs = np.unique(inds.nonzero()[0], return_index=True)
            # gt_idxs, pred_idxs = inds.nonzero()[0][non_repeat_idxs], inds.nonzero()[1][non_repeat_idxs]
            gt_idxs, pred_idxs = inds.nonzero()[0], inds.nonzero()[1]

            pred_labels = [gt_rel[2] for gt_rel in gt_rels[gt_idxs]]
            pred_scores = rel_scores[pred_idxs, pred_labels]
            
            # prepare (gt, pred) idxs for computing soft pseudal labels
            inds = bbox_overlaps(gt_boxes, pred_boxes) >= iou_thres
            _, non_repeat_idxs = np.unique(inds.nonzero()[0], return_index=True)
            gt_idxs, pred_idxs = inds.nonzero()[0][non_repeat_idxs], inds.nonzero()[1][non_repeat_idxs]

            # remove pred boxes that matched to more than one gt box
            pred_idxs_set = set(pred_idxs)
            gt_pred_idxs = []
            for gt_idx, pred_idx in zip(gt_idxs, pred_idxs):
                if pred_idx in pred_idxs_set:
                    pred_idxs_set.remove(pred_idx)
                    gt_pred_idxs.append((gt_idx.item(), pred_idx.item()))
            
            return local_container, pred_scores, pred_labels, gt_pred_idxs, pred_predicates
            
            # return local_container, pred_scores, pred_labels, None, pred_predicates
        elif self.comp_labeling_prob:
            pred_rel_inds_to_idx = {tuple(pred_rel_ind):idx for idx, pred_rel_ind in enumerate(pred_rel_inds)}
            gt_rel_inds = [tuple(gt_rel[:2]) for gt_rel in gt_rels]
            try:
                idxs = [pred_rel_inds_to_idx[gt_rel_ind] for gt_rel_ind in gt_rel_inds]
            except:
                print('Inconsistent indices happened! Skip this img...')
                return local_container, None, None, None, pred_predicates
            pred_labels = [gt_rel[2] for gt_rel in gt_rels]
            pred_scores = rel_scores[idxs, pred_labels]

            # idxs = [i for i in range(len(pred_to_gt)) if pred_to_gt[i]]
            # pred_scores = pred_triplet_scores[idxs, 1]
            # pred_labels = pred_triplets[idxs, 1]

            return local_container, pred_scores, pred_labels, None, pred_predicates
        else:
            return local_container, None, None, None, pred_predicates

class SGHierRecall(SceneGraphEvaluation):
    '''
    Compute Instance-wise Recall. Only work under PredCls mode.
    '''
    def __init__(self, result_dict, comp_obj_hier_recall, matrix_of_ancestor_rel, rel_hierarchy_paths, matrix_of_ancestor_obj, obj_hierarchy_paths):
        super(SGHierRecall, self).__init__(result_dict)
        self.matrix_of_ancestor_rel = matrix_of_ancestor_rel
        self.rel_hierarchy_paths = rel_hierarchy_paths
        
        self.comp_obj_hier_recall = comp_obj_hier_recall
        self.matrix_of_ancestor_obj = matrix_of_ancestor_obj
        self.obj_hierarchy_paths = obj_hierarchy_paths

    def register_container(self, mode):
        self.result_dict[mode + '_hier_recall'] = []

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        result_str += '    HR @ alpha: %.4f; ' % (np.mean(self.result_dict[mode + '_hier_recall']))
        result_str += ' for mode=%s, type=Hierarchical Recall(Main).' % mode
        result_str += '\n'

        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        pred_class_prob = local_container['pred_class_prob']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        obj_scores = local_container['obj_scores']

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['n_gt_triplets'] = None
        local_container['gt_triplet_idxs_to_concept_idxs'] = None
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_rel_to_inds_dic = {tuple(pred_rel_ind): idx for idx, pred_rel_ind in enumerate(pred_rel_inds)}
        unique_rel_pairs = list(set(tuple(gt_rel[:2]) for gt_rel in gt_rels))
        unique_rel_pair_gt_diclist = {rel: [] for rel in unique_rel_pairs}
        for gt_rel in gt_rels:
            unique_rel_pair_gt_diclist[tuple(gt_rel[:2])].append(gt_rel[2])

        # obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        # nogc_overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        # # pred_class_prob_per_rel = pred_class_prob[pred_rel_inds].prod(1) # (6320, 151)
        # # nogc_overall_scores = pred_class_prob_per_rel[:, 1:, None] * rel_scores[:, None, 1:] # (6320, 150, 50)
        # nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        # nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:,0]], nogc_score_inds[:,1]+1))
        # nogc_pred_scores = rel_scores[nogc_score_inds[:,0], nogc_score_inds[:,1]+1]

        pred_triplets = []
        if mode != 'predcls' and mode != 'predcls': # SGCls/SGDet
            iou_thres = global_container['iou_thres']
            
            # Check subject & object IOU >= 0.5
            inds = bbox_overlaps(gt_boxes, pred_boxes) >= iou_thres
            # _, non_repeat_idxs = np.unique(inds.nonzero()[0], return_index=True)
            # gt_idxs, pred_idxs = inds.nonzero()[0][non_repeat_idxs], inds.nonzero()[1][non_repeat_idxs]
            gt_idxs, pred_idxs = inds.nonzero()[0], inds.nonzero()[1]

            gt_to_pred_box_idxs_diclist = defaultdict(list)
            for i, gt_idx in enumerate(gt_idxs):
                gt_to_pred_box_idxs_diclist[gt_idx].append(pred_idxs[i])

            for gt_rel in unique_rel_pair_gt_diclist.keys():
                temp_pred_triplets = []
                sub, obj = gt_rel[0], gt_rel[1]
                for pred_sub_box in gt_to_pred_box_idxs_diclist[sub]:
                    for pred_obj_box in gt_to_pred_box_idxs_diclist[obj]:
                        if pred_sub_box == pred_obj_box:
                            print('A self-pair found! skipped...')
                            continue
                        pred_idx = pred_rel_to_inds_dic[(pred_sub_box, pred_obj_box)]
                        triplet_rel_scores = np.log(rel_scores[pred_idx, 1:])
                        # we supposed to sort 150*50*150 candidate triplets; however,
                        # given the nature of small fraction of 
                        temp_sub_scores = pred_class_prob[pred_sub_box][1:] # (150,)
                        temp_obj_scores = pred_class_prob[pred_obj_box][1:] # (150,)
                        temp_obj_pair_scores = np.outer(temp_sub_scores, temp_obj_scores) # (150, 150)
                        triplet_scores = temp_obj_pair_scores[:, :, None] * rel_scores[None, None, pred_idx, 1:] # (150, 150, 50)
                        
                        # Use heapq to get only the top (6) candidate triplets
                        triplet_scores_list = triplet_scores.reshape(-1).tolist()
                        triplet_scores_list = [(-score, idx) for idx, score in enumerate(triplet_scores_list)]
                        heapq.heapify(triplet_scores_list)
                        top_triplet_scores_list = []
                        while len(top_triplet_scores_list) < 6:
                            top_triplet_scores_list.append(heapq.heappop(triplet_scores_list))

                        box_pair_scores = np.log(obj_scores[[pred_sub_box, pred_obj_box]]).sum()
                        # overall_triplet_scores = triplet_rel_scores + box_pair_scores
                        for neg_score, idx in top_triplet_scores_list:
                            rel_idx = idx // (150 * 150)
                            sub_idx = (idx % (150 * 150)) // 150
                            obj_idx = (idx % (150 * 150)) % 150
                            assert idx == rel_idx * 150 * 150 + sub_idx * 150 + obj_idx # sanity check
                            temp_pred_triplets.append({
                                'overall_triplet_score': -neg_score,
                                'sub_score': pred_class_prob[pred_sub_box, sub_idx + 1],
                                'rel_score': rel_scores[pred_idx, rel_idx + 1],
                                'obj_score': pred_class_prob[pred_obj_box, obj_idx + 1],
                                'sub_idx': sub_idx,
                                'rel_idx': rel_idx,
                                'obj_idx': obj_idx,
                            })
                pred_triplets.append(temp_pred_triplets)
        else: # PredCls
            for rel_pair in unique_rel_pairs:
                try:
                    pred_idx = pred_rel_to_inds_dic[rel_pair]
                    sorted_rel_idxs = np.argsort(rel_scores[pred_idx, 1:])[::-1] + 1
                except:
                    print('Inconsistent pred_rel_pairs happens! Example skipped...')
                    pred_triplets.append([])
                    continue
                
                temp_pred_triplets = []
                for rel_idx in sorted_rel_idxs:
                    rel_score = rel_scores[pred_idx, rel_idx]
                    temp_pred_triplets.append({
                        'overall_triplet_score': rel_score,
                        'rel_score': rel_score,
                        'rel_idx': rel_idx,
                    })
                pred_triplets.append(temp_pred_triplets)

        local_container['unique_rel_pair_gt_diclist'] = unique_rel_pair_gt_diclist

        # gt_rels are relation (subj, obj, pred) triplets; note that can be more than one pair of the same (subj, obj) exists!
        for idx, gt_rel_list in enumerate(unique_rel_pair_gt_diclist.values()):
            # rec_i = len(np.intersect1d(sorting_idx[idx, :alpha_i], gt_rel_list)) / len(gt_rel_list)
            if len(pred_triplets[idx]) == 0: # Not even bounding boxes overlapped; only happens in SGDet!
                # assert mode == 'sgdet' # sanity check
                hier_recalls = [0.0] * len(gt_rel_list)
            else:
                alpha_i = len(gt_rel_list)
                # import pdb; pdb.set_trace()
                top_alpha_i_pred_triplets = sorted(pred_triplets[idx], key=lambda x: x['overall_triplet_score'], reverse=True)[:alpha_i]

                # pred_triplets_idx = [pred_triplets[idx][jdx]['overall_triplet_scores'] for jdx in range(len(pred_triplets[idx]))]
                # sorting_idx = np.argsort(np.concatenate(pred_triplets_idx))[::-1][:alpha_i]
                # r_k = []
                # for jdx in sorting_idx:
                #     pred_triplet = pred_triplets[idx][jdx // 50]
                #     for k, v in pred_triplet:
                #         pass
                #     r_k.append()
                # r_k = pred_triplets[idx][:alpha_i]

                hier_recalls = max_semantic_similarity(gt_rel_list, top_alpha_i_pred_triplets, self.matrix_of_ancestor_rel, self.rel_hierarchy_paths, self.comp_obj_hier_recall, self.matrix_of_ancestor_obj, self.obj_hierarchy_paths)
            self.result_dict[mode + '_hier_recall'].extend(hier_recalls)

        '''
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        obj_scores = local_container['obj_scores']

        pred_rel_to_inds_dic = {tuple(pred_rel_ind): idx for idx, pred_rel_ind in enumerate(pred_rel_inds)}
        unique_rel_pairs = list(set(tuple(gt_rel[:2]) for gt_rel in gt_rels))
        unique_rel_pair_gt_diclist = {rel: [] for rel in unique_rel_pairs}
        for gt_rel in gt_rels:
            unique_rel_pair_gt_diclist[tuple(gt_rel[:2])].append(gt_rel[2])
        try:
            rel_scores = rel_scores[[pred_rel_to_inds_dic[rel_pair] for rel_pair in unique_rel_pairs], 1:]
        except:
            print('Inconsistent pred_rel_pairs happenen! Example skipped...')
            pass
        sorting_idx = np.argsort(rel_scores, axis=1)[:, ::-1] + 1

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['n_gt_triplets'] = None
        local_container['gt_triplet_idxs_to_concept_idxs'] = None

        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        local_container['unique_rel_pair_gt_diclist'] = unique_rel_pair_gt_diclist

        for k in self.result_dict[mode + '_instance_recall']:
            # gt_rels are relation (subj, obj, pred) triplets; note that can be more than one pair of the same (subj, obj) exists!
            for idx, gt_rel_list in enumerate(unique_rel_pair_gt_diclist.values()):
                rec_i = len(np.intersect1d(sorting_idx[idx, :k], gt_rel_list)) / len(gt_rel_list)
                self.result_dict[mode + '_instance_recall'][k].append(rec_i)
                if len(gt_rel_list) > 1: # only evaluate on multi-label object pairs
                    self.result_dict[mode + '_multi_instance_recall'][k].append(rec_i)
        '''
        
        return 

class SGInstanceRecall(SceneGraphEvaluation):
    '''
    Compute Instance-wise Recall. Only work under PredCls mode.
    '''
    def __init__(self, result_dict, test_with_cogtree=False, cogtree_file=None):
        super(SGInstanceRecall, self).__init__(result_dict)
        self.test_with_cogtree = test_with_cogtree
        if test_with_cogtree: 
            assert cogtree_file is not None
            self.cogtree_file = cogtree_file

    def register_container(self, mode):
        self.result_dict[mode + '_instance_recall'] = {1: [], 2: [], 3: [], 6: [], 10: []}
        self.result_dict[mode + '_multi_instance_recall'] = {1: [], 2: [], 3: [], 6: [], 10: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_instance_recall'].items():
            result_str += '    IR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Instance Recall(Main).' % mode
        result_str += '\n'

        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_multi_instance_recall'].items():
            result_str += '    multi-IR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Multilabel Instance Recall(Main).' % mode
        result_str += '\n'

        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        obj_scores = local_container['obj_scores']

        pred_rel_to_inds_dic = {tuple(pred_rel_ind): idx for idx, pred_rel_ind in enumerate(pred_rel_inds)}
        unique_rel_pairs = list(set(tuple(gt_rel[:2]) for gt_rel in gt_rels))
        unique_rel_pair_gt_diclist = {rel: [] for rel in unique_rel_pairs}
        for gt_rel in gt_rels:
            unique_rel_pair_gt_diclist[tuple(gt_rel[:2])].append(gt_rel[2])
        try:
            rel_scores = rel_scores[[pred_rel_to_inds_dic[rel_pair] for rel_pair in unique_rel_pairs], 1:]
        except:
            print('Inconsistent pred_rel_pairs happenen! Example skipped...')
            pass
        sorting_idx = np.argsort(rel_scores, axis=1)[:, ::-1] + 1

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['n_gt_triplets'] = None
        local_container['gt_triplet_idxs_to_concept_idxs'] = None

        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        local_container['sorted_preds_for_gt_pair'] = sorting_idx
        local_container['unique_rel_pair_gt_diclist'] = unique_rel_pair_gt_diclist

        for k in self.result_dict[mode + '_instance_recall']:
            # gt_rels are relation (subj, obj, pred) triplets; note that can be more than one pair of the same (subj, obj) exists!
            for idx, gt_rel_list in enumerate(unique_rel_pair_gt_diclist.values()):
                rec_i = len(np.intersect1d(sorting_idx[idx, :k], gt_rel_list)) / len(gt_rel_list)
                self.result_dict[mode + '_instance_recall'][k].append(rec_i)
                if len(gt_rel_list) > 1: # only evaluate on multi-label object pairs
                    self.result_dict[mode + '_multi_instance_recall'][k].append(rec_i)

        return local_container, None, None, None

"""
No Graph Constraint Mean Recall
"""
class SGMeanInstanceRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanInstanceRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        self.result_dict[mode + '_i_mean_recall'] = {1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 10: 0.0}
        self.result_dict[mode + '_i_mean_recall_collect'] = {1: [[] for i in range(self.num_rel)], 2: [[] for i in range(self.num_rel)], 3: [[] for i in range(self.num_rel)], 6: [[] for i in range(self.num_rel)], 10: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_i_mean_recall_list'] = {1: [], 2: [], 3: [], 6: [], 10: []}

        self.result_dict[mode + '_multi_i_mean_recall'] = {1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 10: 0.0}
        self.result_dict[mode + '_multi_i_mean_recall_collect'] = {1: [[] for i in range(self.num_rel)], 2: [[] for i in range(self.num_rel)], 3: [[] for i in range(self.num_rel)], 6: [[] for i in range(self.num_rel)], 10: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_multi_i_mean_recall_list'] = {1: [], 2: [], 3: [], 6: [], 10: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_i_mean_recall'].items():
            result_str += 'mIR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Instance Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_i_mean_recall_list'][10]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_multi_i_mean_recall'].items():
            result_str += 'multi-mIR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Multilabel Mean Instance Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_multi_i_mean_recall_list'][10]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        gt_rels = local_container['gt_rels']
        sorting_idx = local_container['sorted_preds_for_gt_pair']
        unique_rel_pair_gt_diclist = local_container['unique_rel_pair_gt_diclist']

        for k in self.result_dict[mode + '_i_mean_recall_collect']:
            for idx, gt_rel_list in enumerate(unique_rel_pair_gt_diclist.values()):
                match = np.intersect1d(sorting_idx[idx, :k], gt_rel_list)

                recall_hit = [0] * self.num_rel
                recall_count = [0] * self.num_rel
                for local_label in gt_rel_list:
                    recall_count[int(local_label)] += 1
                    recall_count[0] += 1

                for local_label in match:
                    recall_hit[int(local_label)] += 1
                    recall_hit[0] += 1
                
                for n in range(self.num_rel):
                    if recall_count[n] > 0:
                        self.result_dict[mode + '_i_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
                        if len(gt_rel_list) > 1: # only evaluate on multi-label object pairs
                            self.result_dict[mode + '_multi_i_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self, mode):
        num_rel_no_bg = self.num_rel - 1
        
        for k, v in self.result_dict[mode + '_i_mean_recall'].items():
            sum_recall = 0
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_i_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_i_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_i_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_i_mean_recall'][k] = sum_recall / float(num_rel_no_bg)

        for k, v in self.result_dict[mode + '_multi_i_mean_recall'].items():
            sum_recall = 0
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_multi_i_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_multi_i_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_multi_i_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_multi_i_mean_recall'][k] = sum_recall / float(num_rel_no_bg)

        return


"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, test_with_cogtree=False):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)
        self.test_with_cogtree = test_with_cogtree

    def register_container(self, mode):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += ' ng-R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:,0]], nogc_score_inds[:,1]+1))
        nogc_pred_scores = rel_scores[nogc_score_inds[:,0], nogc_score_inds[:,1]+1]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
                nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
            test_with_cogtree=self.test_with_cogtree,
            n_gt_triplets=local_container['n_gt_triplets'], 
            gt_triplet_idxs_to_concept_idxs=local_container['gt_triplet_idxs_to_concept_idxs'],
        )

        local_container['nogc_pred_to_gt'] = nogc_pred_to_gt

        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            if self.test_with_cogtree:
                rec_i = float(len(match)) / float(local_container['n_gt_triplets'])
            else:
                rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)

        return local_container

"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""
class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += '   zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)


"""
No Graph Constraint Mean Recall
"""
class SGNGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNGZeroShotRecall, self).__init__(result_dict)
    
    def register_container(self, mode):
        self.result_dict[mode + '_ng_zeroshot_recall'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_zeroshot_recall'].items():
            result_str += 'ng-zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']

        for k in self.result_dict[mode + '_ng_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_ng_zeroshot_recall'][k].append(zero_rec_i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""
class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + '_accuracy_count'][k])
            result_str += '    A @ %d: %.4f; ' % (k, a_hit/a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
            # for sgcls and predcls
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""
class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        #self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        #self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return


"""
No Graph Constraint Mean Recall
"""
class SGNGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGNGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        self.result_dict[mode + '_ng_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_ng_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_ng_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            result_str += 'ng-mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=No Graph Constraint Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_ng_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_ng_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_ng_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_ng_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_ng_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""
class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)

        return 


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, phrdet=False, 
                 test_with_cogtree=False, n_gt_triplets=0, gt_triplet_idxs_to_concept_idxs=None):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    
    # if test_with_cogtree:
    #     gt_triplet_idxs_to_concept_idxs = defaultdict(list)
    #     n_gt_triplets = concept_idx = len(gt_triplets)
    #     concept_gt_triplets, concept_gt_boxes = [], []
    #     for idx, gt_triplet in enumerate(gt_triplets):
    #         for concept in cogtree_file[gt_triplet[1]]:
    #             if concept != gt_triplet[1]:
    #                 gt_triplet_idxs_to_concept_idxs[idx].append(concept_idx)
    #                 concept_idx += 1

    #                 concept_gt_triplets.append([[gt_triplet[0], concept, gt_triplet[2]]])
    #                 concept_gt_boxes.append([gt_boxes[idx]])
    #     # import pdb; pdb.set_trace()
    #     concept_gt_triplets = np.concatenate(concept_gt_triplets)
    #     gt_triplets = np.concatenate([gt_triplets, concept_gt_triplets])

    #     concept_gt_boxes = np.concatenate(concept_gt_boxes)
    #     gt_boxes = np.concatenate([gt_boxes, concept_gt_boxes])

    keeps = intersect_2d(gt_triplets, pred_triplets)
    if test_with_cogtree:
        for idx, is_found in enumerate(keeps.any(1)[:n_gt_triplets]):
            if is_found and idx in gt_triplet_idxs_to_concept_idxs:
                for jdx in gt_triplet_idxs_to_concept_idxs[idx]:
                    keeps[jdx] = np.array([False] * len(pred_triplets))
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


