"""Criterion."""
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class DiceLoss(nn.Module):
    """Dice score loss function."""

    def __init__(self, eps=1e-12, weights=None, loss_type='log'):
        """Initialize class."""
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.weights = None
        if weights is not None:
            self.weights = weights / weights.sum()
            assert weights.dim() == 1, \
                "Weights tensor must be 1 dimensional tensor."
        self.loss_type = loss_type
        self.ce_loss = nn.CrossEntropyLoss(torch.FloatTensor([1, 40])).cuda()

    def forward(self, input, target):
        """Forward call."""
        index_to_select = []
        for i in range(input.size(0)):
            # print(target.data[i].sum())
            if target.data[i].sum() >= 0:
                index_to_select.append(i)
        if len(index_to_select) == 0:
            return (0 * input).sum(), 0
        # print(index_to_select)
        index_to_select_tensor = Variable(
            torch.LongTensor(index_to_select).cuda())
        probabilities = F.softmax(input)

        self.processed_target = target.index_select(0, index_to_select_tensor)
        # print(self.processed_target.min(),
        #       self.processed_target.max(), self.processed_target.sum())
        self.probabilities_selected = probabilities.index_select(
            0, index_to_select_tensor)
        return self.ce_loss(
            self.probabilities_selected, self.processed_target), \
            len(index_to_select)
        # print(index_to_select_tensor)
