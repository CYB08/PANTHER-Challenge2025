import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.BLUNet import get_blunet_from_plans
from torch.optim import Adam
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
from typing import Tuple, Union, List

class nnUNetTrainerBLUnet(nnUNetTrainer):
    def __init__(
        self, 
        plans: dict, 
        configuration: str, 
        fold: int, 
        dataset_json: dict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        arch_kwargs = {
            'n_stages': 6,
            'features_per_stage': [32, 64, 128, 256, 320, 320],
            'conv_op': 'torch.nn.modules.conv.Conv3d',
            'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'strides': [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1], [1, 1, 1]],
            'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
            'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
            'order_range': [2, 2, 2, 3, 3, 4],
        }

        model = get_blunet_from_plans(
            arch_kwargs,
            num_input_channels,
            num_output_channels,
            deep_supervision=enable_deep_supervision
            )

        return model

class nnUNetTrainerNoMirroringBLUnet_500epochs(nnUNetTrainerBLUnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

class nnUNetTrainerNoMirroringBLUnet_1000epochs(nnUNetTrainerBLUnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

class nnUNetTrainerBLUnet_1000epoch(nnUNetTrainerBLUnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000