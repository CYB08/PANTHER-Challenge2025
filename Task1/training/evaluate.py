"""
Semi-supervised Pancreatic Tumor Segmentation: Stage 2
- Evaluate pseudo-label quality using Dice scores
- Select high-quality pseudo-labels for next training iteration
"""

from pathlib import Path
import json
import shutil
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class PseudoLabelSelector:
    """Select high-quality pseudo-labels based on evaluation metrics"""
    
    def __init__(self, dice_threshold=0.7, labels=[1, 2]):
        self.dice_threshold = dice_threshold
        self.labels = labels
    
    def evaluate_predictions(self, pred_folder, gt_folder, output_json):
        """Compute Dice scores for pseudo-labels against ground truth"""
        print("Evaluating pseudo-label quality...")
        print(f"Predictions: {pred_folder}")
        print(f"Ground truth: {gt_folder}")
        
        metrics = compute_metrics_on_folder(
            folder_ref=gt_folder,
            folder_pred=pred_folder,
            output_file=output_json,
            image_reader_writer=SimpleITKIO(),
            file_ending=".nii.gz",
            regions_or_labels=self.labels,
            ignore_label=None,
            num_processes=4,
            chill=True
        )
        
        print(f"Results saved to: {output_json}")
        
        # Print summary
        if 'foreground_mean' in metrics:
            print(f"\nOverall Dice: {metrics['foreground_mean']['Dice']:.4f}")
        
        if 'mean' in metrics:
            print("\nPer-class Dice:")
            for label in self.labels:
                if label in metrics['mean']:
                    print(f"  Class {label}: {metrics['mean'][label]['Dice']:.4f}")
        
        return metrics
    
    def select_high_quality_cases(self, metrics_json, pred_folder, output_folder):
        """
        Select cases with Dice scores above threshold
        Copy selected pseudo-labels to output folder for next training iteration
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)
        
        # Extract per-case metrics
        if 'metric_per_case' not in metrics:
            print("No per-case metrics found")
            return
        
        per_case = metrics['metric_per_case']
        selected_cases = []
        
        for case_id, case_metrics in per_case.items():
            # Check if all tumor classes meet threshold
            dice_scores = [case_metrics[str(label)]['Dice'] 
                          for label in self.labels 
                          if str(label) in case_metrics]
            
            if dice_scores and min(dice_scores) >= self.dice_threshold:
                selected_cases.append(case_id)
                
                # Copy to output folder
                src_file = Path(pred_folder) / f"{case_id}.nii.gz"
                dst_file = output_path / f"{case_id}.nii.gz"
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
        
        print(f"\nSelected {len(selected_cases)}/{len(per_case)} high-quality cases")
        print(f"Threshold: Dice >= {self.dice_threshold}")
        print(f"Output: {output_folder}")
        
        # Save selection list
        selection_file = output_path / "selected_cases.txt"
        with open(selection_file, 'w') as f:
            f.write('\n'.join(sorted(selected_cases)))
        
        return selected_cases


def main():
    """
    Stage 2: Select high-quality pseudo-labels
    """
    
    # Configuration
    PSEUDO_LABELS = "/path/to/pseudolabels"
    VALIDATION_GT = "/path/to/validation_labels"
    METRICS_JSON = "/path/to/evaluation_results.json"
    SELECTED_OUTPUT = "/path/to/selected_pseudolabels"
    
    DICE_THRESHOLD = 0.7
    LABELS = [1, 2]  # Tumor classes
    
    selector = PseudoLabelSelector(DICE_THRESHOLD, LABELS)
    
    # Step 1: Evaluate quality
    metrics = selector.evaluate_predictions(
        PSEUDO_LABELS,
        VALIDATION_GT,
        METRICS_JSON
    )
    
    # Step 2: Select high-quality cases
    selected = selector.select_high_quality_cases(
        METRICS_JSON,
        PSEUDO_LABELS,
        SELECTED_OUTPUT
    )


if __name__ == "__main__":
    main()