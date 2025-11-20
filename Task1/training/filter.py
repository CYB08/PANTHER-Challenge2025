"""
Semi-supervised Pancreatic Tumor Segmentation: Stage 1
- Generate pseudo-labels using nnUNet inference on unlabeled MRI data
- Prepare predictions for quality evaluation
"""

import torch
import shutil
import tempfile
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import SimpleITK as sitk
import numpy as np


class PseudoLabelGenerator:
    """Generate pseudo-labels for semi-supervised learning"""
    
    def __init__(self, model_folder, device='cuda'):
        self.model_folder = Path(model_folder)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.predictor = None
    
    def filter_weights(self):
        """Remove domain adaptation layers from checkpoints"""
        print(f"Filtering weights in: {self.model_folder}")
        
        for fold_folder in sorted(self.model_folder.glob('fold_*')):
            checkpoint_path = fold_folder / 'checkpoint_final.pth'
            if not checkpoint_path.exists():
                continue
            
            # Backup and load
            backup_path = fold_folder / 'checkpoint_final_backup.pth'
            if not backup_path.exists():
                shutil.copy2(checkpoint_path, backup_path)
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Filter domain classifier weights
            filtered_weights = {k: v for k, v in checkpoint['network_weights'].items() 
                              if 'domain_classifier' not in k}
            
            checkpoint['network_weights'] = filtered_weights
            torch.save(checkpoint, checkpoint_path)
            print(f"  Filtered {fold_folder.name}: removed domain adaptation weights")
    
    def initialize_predictor(self):
        """Initialize nnUNet predictor"""
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False
        )
        self.predictor.initialize_from_trained_model_folder(
            str(self.model_folder),
            use_folds=('0',),
            checkpoint_name='checkpoint_final.pth'
        )
        print("Predictor initialized")
    
    def predict_case(self, input_file, output_file):
        """Generate pseudo-label for a single case"""
        case_name = input_file.stem.replace('_0000', '')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_dir = Path(temp_dir) / "input"
            temp_output_dir = Path(temp_dir) / "output"
            temp_input_dir.mkdir()
            temp_output_dir.mkdir()
            
            # Copy to temp with nnUNet naming
            temp_input = temp_input_dir / f"{case_name}_0000.nii.gz"
            shutil.copy2(input_file, temp_input)
            
            # Run inference
            self.predictor.predict_from_files(
                str(temp_input_dir),
                str(temp_output_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1
            )
            
            # Copy result
            pred_file = temp_output_dir / f"{case_name}.nii.gz"
            if pred_file.exists():
                shutil.copy2(pred_file, output_file)
                return True
            return False
    
    def batch_generate_pseudolabels(self, input_folder, output_folder):
        """Generate pseudo-labels for all cases in input folder"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(input_path.glob("*_0000.nii.gz"))
        print(f"Processing {len(image_files)} MRI scans...")
        
        if self.predictor is None:
            self.initialize_predictor()
        
        success_count = 0
        for i, img_file in enumerate(image_files, 1):
            case_name = img_file.stem.replace('_0000', '')
            output_file = output_path / f"{case_name}.nii.gz"
            
            if output_file.exists():
                print(f"[{i}/{len(image_files)}] {case_name}: exists, skipped")
                success_count += 1
                continue
            
            print(f"[{i}/{len(image_files)}] Processing {case_name}...")
            if self.predict_case(img_file, output_file):
                success_count += 1
        
        print(f"Pseudo-label generation completed: {success_count}/{len(image_files)}")


def main():
    """
    Stage 1: Generate pseudo-labels for semi-supervised learning
    """
    
    # Configuration
    MODEL_FOLDER = "/path/to/trained_model"
    INPUT_FOLDER = "/path/to/unlabeled_mri"
    OUTPUT_FOLDER = "/path/to/pseudolabels"
    
    generator = PseudoLabelGenerator(MODEL_FOLDER)
    
    # Step 1: Filter weights (run once)
    generator.filter_weights()
    
    # Step 2: Generate pseudo-labels
    generator.batch_generate_pseudolabels(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()