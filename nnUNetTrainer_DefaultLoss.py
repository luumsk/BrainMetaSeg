from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True


class nnUNetTrainer_DefaultLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.initial_lr = 1e-2
        self.num_epochs = 100
        self.save_every = 5
        self.disable_checkpointing = False