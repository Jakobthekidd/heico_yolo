import os
import yaml
import argparse
import torch
import numpy as np
from ultralytics import YOLO, RTDETR


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    
    # --- General/Model Arguments ---
    parser.add_argument('--config', type=str, default='config/paths/cluster.yaml', help='Path to general config')
    parser.add_argument('--dataset_config', type=str, default='config/dataset/heico_end.yaml', help='Path to data YAML')
    parser.add_argument('--model', type=str, default='yolo11l.pt', help='Base model to start training from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial Learning rate')

    # --- Augmentation Arguments (ONLY supported ones passed directly) ---
    parser.add_argument('--hsv_h', type=float, default=0.015, help='image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0, help='image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='image translation (fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='image scale augmentation (fraction)')
    parser.add_argument('--shear', type=float, default=0.0, help='image shear augmentation (deg)')
    parser.add_argument('--perspective', type=float, default=0.0, help='image perspective augmentation (fraction), range 0.0-0.001')
    parser.add_argument('--flipud', type=float, default=0.0, help='image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='0.0 to disable mosaic')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup probability')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='copy-paste probability')

    # --- Configuration File Argument for unsupported hyperparams like 'blur' ---
    # This must be used to pass 'blur'
    parser.add_argument('--hyp_cfg', type=str, default=None, help='Path to custom hyperparameter YAML (e.g., to add blur)')

    # --- Save/Logging Arguments ---
    parser.add_argument('--project', type=str, default='heico_', help='Project name for save directory')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--save_period', type=int, default=0)
    parser.add_argument('--save_best', action='store_true', help='Save the best checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    torch.cuda.empty_cache()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    if os.path.exists(args.model):
        model_path = args.model
    else:
        model_path = os.path.join(script_dir, 'models', args.model)

    print(f"Loading model architecture: {args.model}")
    

    if "yolo" in args.model.lower():
        model = YOLO(model_path)
    else:
        model = RTDETR(model_path)

    print(f"--- Starting Training for {args.epochs} epochs with Augmentations ---")
    
    # --- Pass all supported arguments, including the optional hyp_cfg file via 'cfg' ---
    train_results = model.train(
        data=args.dataset_config,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.lr,
        cfg=args.hyp_cfg, # <-- This is where 'blur' goes if a hyp file is provided
        project=os.path.join(cfg.get('output_dir', 'runs'), args.project),
        name=args.name or f"{os.path.splitext(os.path.basename(args.model))[0]}_{args.project}",
        save_period=args.save_period,
        save=args.save_best,
        plots=True,
        # AUGMENTATION PARAMETERS (Supported directly by Ultralytics train function)
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste
    )
    
    print("Training finished.")
    
  
    best_weight_path = os.path.join(train_results.save_dir, 'weights', 'best.pt')
    if not os.path.exists(best_weight_path):
        print(f"Best weights not found at {best_weight_path}, trying 'last.pt'...")
        best_weight_path = os.path.join(train_results.save_dir, 'weights', 'last.pt')

    print(f"\n--- Loading Best Model for Validation: {best_weight_path} ---")
    

    if "yolo" in args.model.lower():
        val_model = YOLO(best_weight_path)
    else:
        val_model = RTDETR(best_weight_path)

    metrics = val_model.val(data=args.dataset_config, split='val')


    print("\n--- Detailed Validation Metrics ---")
    
    # 1. Mean Average Precision (mAP)
    print("### Mean Average Precision (mAP) ###")
    print(f"mAP@50:    {metrics.box.map50:.4f} (IoU threshold = 0.5)")
    print(f"mAP@50-95: {metrics.box.map:.4f} (Average mAP across IoU thresholds 0.5 to 0.95)")
    print(f"mAP@75:    {metrics.box.map75:.4f} (IoU threshold = 0.75)")
    
    # 2. Precision, Recall, and F1 Score
    print("\n### Precision, Recall, and F1 Score (Averaged) ###")
    mean_precision = metrics.box.mp
    mean_recall = metrics.box.mr
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall:    {mean_recall:.4f}")

    if (mean_precision + mean_recall) > 0:
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        print(f"F1 Score:  {f1_score:.4f}")
    
    # 3. Class-specific Metrics
    print("\n### Class-Specific Metrics ###")
    try:
        if hasattr(metrics.box, 'p') and metrics.box.p is not None and metrics.box.p.size > 1:
            class_names = val_model.names 
            class_precision = metrics.box.p 
            class_recall = metrics.box.r 
            class_ap50 = metrics.box.ap50 
            class_ap = metrics.box.ap     
            
            # Print Class-Specific Table
            print(f"{'Class':<20} {'P':<10} {'R':<10} {'AP@50':<10} {'AP@50-95':<10}")
            print("-" * 60)
            for i, name in class_names.items():
                print(f"{name:<20} {class_precision[i]:<10.4f} {class_recall[i]:<10.4f} {class_ap50[i]:<10.4f} {class_ap[i]:<10.4f}")
        else:
            print("Class-specific metrics arrays are not available or are empty.")
    except Exception as e:
        print(f"Error printing class-specific metrics: {e}")
        
    # 4. Confusion Matrix (Confusion Metrics)
    print("\n### Confusion Matrix (Confusion Metrics) ###")
    try:
        cm = metrics.confusion_matrix.matrix
        print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
        print(cm)
        
        # Normalized Confusion Matrix (Often more informative)
        print("\nNormalized Confusion Matrix:")
        normalized_cm = cm / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) 
        print(normalized_cm)
        
        print(f"\nCharts saved to: {metrics.save_dir}")
    except AttributeError:
        print("\nCould not retrieve confusion matrix object directly or matrix is empty.")
    except Exception as e:
        print(f"An unexpected error occurred while printing the confusion matrix: {e}")

if __name__ == "__main__":
    main()