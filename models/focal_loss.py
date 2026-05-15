import torch

import torch.nn as nn

import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(

        self,

        alpha=1,

        gamma=2,

        reduction="mean"
    ):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.reduction = reduction

    def forward(

        self,

        logits,

        targets
    ):

        # Standard cross entropy loss
        ce_loss = F.cross_entropy(

            logits,

            targets,

            reduction="none"
        )

        # Convert CE loss into probability
        pt = torch.exp(-ce_loss)

        # Focal loss formula
        focal_loss = self.alpha * (

            (1 - pt) ** self.gamma

        ) * ce_loss

        # Reduction
        if self.reduction == "mean":

            return focal_loss.mean()

        elif self.reduction == "sum":

            return focal_loss.sum()

        else:

            return focal_loss


# Testing focal loss
if __name__ == "__main__":

    # Fake logits
    logits = torch.tensor([

        [2.5, 0.3],

        [0.2, 1.8],

        [1.0, 1.2]

    ])

    # True labels
    targets = torch.tensor([

        0,

        1,

        1
    ])

    # Initialize focal loss
    criterion = FocalLoss()

    # Compute loss
    loss = criterion(

        logits,

        targets
    )

    print("Focal Loss:")

    print(loss)