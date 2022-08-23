import torch
import torch.nn as nn


class LACrossEntropyLoss(nn.Module):
    def __init__(self,
                 tau=1.0,
                 ignore_index=255,
                 class_weight=None,
                 loss_name='loss_logit_adjustment_cross_entropy'):
        super(LACrossEntropyLoss, self).__init__()

        self.tau = tau
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean',
                                                 ignore_index=self.ignore_index)
        self._loss_name = loss_name

    def forward(self,
                inputs,
                targets):
        base_probs = torch.exp(-self.class_weight)
        inputs = inputs + torch.log(base_probs ** self.tau + 1e-12).to(input.device)
        loss = self.cross_entropy(inputs, targets)
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
