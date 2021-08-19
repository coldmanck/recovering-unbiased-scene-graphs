import torch
from torch import nn
from maskrcnn_benchmark.modeling.utils import cat
import torch.nn.functional as F

class LearnableBalancedNorm1d(nn.Module):
    """
    LearnableBalancedNorm1d.
    """

    def __init__(self, num_features, eps=1e-5, normalized_probs=False):
        super(LearnableBalancedNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.normalized_probs = normalized_probs
        self.labeling_prob_theta = nn.Parameter(torch.randn(50))

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))

    def forward(self, relation_logits, rel_labels):
        relation_logits = cat(relation_logits) if isinstance(relation_logits, tuple) else relation_logits
        self._check_input_dim(relation_logits)

        labeling_prob = torch.sigmoid(self.labeling_prob_theta)
        labeling_prob = torch.cat((torch.ones(1).cuda(), labeling_prob)) + self.eps

        relation_probs_norm = F.softmax(relation_logits, dim=-1) / labeling_prob
        if self.normalized_probs:
            # relation_probs_norm /= (torch.sum(relation_probs_norm, dim=-1).view(-1, 1) + self.eps)
            # relation_probs_norm = F.softmax(relation_probs_norm, dim=-1)
            # import pdb; pdb.set_trace()
            relation_probs_norm[:, 0] = 1 - relation_probs_norm[:, 1:].sum(1)

        return relation_probs_norm, labeling_prob

class BalancedNorm1d(nn.Module):
    """
    BalancedNorm1d.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, init_prob=0.03, track_running_stats=True, normalized_probs=True, with_gradient=False):
        super(BalancedNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.normalized_probs = normalized_probs
        self.with_gradient = with_gradient
        if self.track_running_stats:
            self.init_prob = init_prob
            self.register_buffer("running_labeling_prob", torch.tensor([init_prob] * num_features))
            self.running_labeling_prob[0] = 1 # BG labeling prob is always one
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_labeling_prob", None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_labeling_prob.fill_(self.init_prob)
            self.running_labeling_prob[0] = 1 # BG labeling prob is always one
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        # if self.affine:
        #     init.ones_(self.weight)
        #     init.zeros_(self.bias)

    # def forward(self, x):
    #     # Cast all fixed parameters to half() if necessary
    #     if x.dtype == torch.float16:
    #         self.weight = self.weight.half()
    #         self.bias = self.bias.half()
    #         self.running_mean = self.running_mean.half()
    #         self.running_var = self.running_var.half()

    #     scale = self.weight * self.running_var.rsqrt()
    #     bias = self.bias - self.running_mean * scale
    #     scale = scale.reshape(1, -1, 1, 1)
    #     bias = bias.reshape(1, -1, 1, 1)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))

    def forward(self, relation_logits, rel_labels):
        '''
        Takes in the same parameters as those of common loss functions.
        Parameters:
        - input: Input probability score (logits passed through Softmax).
        - target: Target 
        '''
        # import pdb; pdb.set_trace()
        relation_logits = cat(relation_logits) if isinstance(relation_logits, tuple) else relation_logits
        rel_labels = cat(rel_labels) if isinstance(rel_labels, list) else rel_labels

        self._check_input_dim(relation_logits)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            # import pdb; pdb.set_trace()
            fg_idxs = (rel_labels != 0)
            fg_relation_probs = F.softmax(relation_logits[fg_idxs], dim=-1)
            rel_labels_one_hot = torch.zeros_like(fg_relation_probs, dtype=torch.int)
            rel_labels_one_hot[list(range(len(fg_relation_probs))), rel_labels[fg_idxs]] = 1
            labeling_prob = torch.sum(fg_relation_probs * rel_labels_one_hot, dim=0) / torch.sum(rel_labels_one_hot, dim=0)
            non_nan_idxs = ~torch.isnan(labeling_prob)

            if self.with_gradient:
                self.running_labeling_prob[non_nan_idxs] = exponential_average_factor * labeling_prob[non_nan_idxs] + (1 - exponential_average_factor) * self.running_labeling_prob[non_nan_idxs]
            else:
                with torch.no_grad():
                    self.running_labeling_prob[non_nan_idxs] = exponential_average_factor * labeling_prob[non_nan_idxs] + (1 - exponential_average_factor) * self.running_labeling_prob[non_nan_idxs]
        # else:
        #     labeling_prob = self.running_labeling_prob
        assert self.running_labeling_prob[0] == 1

        relation_probs_norm = F.softmax(relation_logits, dim=-1) / (self.running_labeling_prob + self.eps)
        # import pdb; pdb.set_trace()
        if self.normalized_probs:
            # relation_probs_norm /= (torch.sum(relation_probs_norm, dim=-1).view(-1, 1) + self.eps)
            # relation_probs_norm = F.softmax(relation_probs_norm, dim=-1)
            relation_probs_norm[:, 0] = 1 - relation_probs_norm[:, 1:].sum(1)

        return relation_probs_norm, self.running_labeling_prob.detach(), None if not self.training else torch.sum(rel_labels_one_hot, dim=0)