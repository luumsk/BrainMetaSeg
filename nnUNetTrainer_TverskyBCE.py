from monai.losses import TverskyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch


class CombinedTverskyBCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, pos_w):
        super().__init__()

        self.tversky_loss = TverskyLoss(to_onehot_y=False, sigmoid=True, alpha=alpha, beta=beta)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.pos_w = pos_w

    def forward(self, y_hat, y):
        y_i64 = y.type(torch.int64)
        y_f32 = y.type(torch.float32)
        return self.tversky_loss(y_hat, y_i64) + ((1 + self.pos_w * y_f32) * self.bce_loss(y_hat, y_f32)).mean()


class nnUNetTrainer_TverskyBCE(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.initial_lr = 1e-2
        self.num_epochs = 100
        self.save_every = 5
        self.disable_checkpointing = False
    
    def _build_loss(self):
        loss = CombinedTverskyBCELoss(0.3, 0.7, 10)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)

        return loss
