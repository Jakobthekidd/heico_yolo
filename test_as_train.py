"""


Using Test as training and the train set as ground truth



"""

# import os
# import yaml
# import argparse
# import torch
# import numpy as np
# from ultralytics import YOLO, RTDETR


# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Training Script")
    
#     parser.add_argument('--config', type=str, default='config/paths/cluster.yaml', help='Path to general config')
#     parser.add_argument('--dataset_config', type=str, default='config/dataset/heico_end.yaml', help='Path to data YAML')
#     parser.add_argument('--model', type=str, default='yolo11l.pt', help='Base model to start training from')
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--img_size', type=int, default=640)
#     parser.add_argument('--lr', type=float, default=0.0001, help='Initial Learning rate')

#     parser.add_argument('--project', type=str, default='heico_', help='Project name for save directory')
#     parser.add_argument('--name', type=str, default=None, help='Experiment name')
#     parser.add_argument('--save_period', type=int, default=0)
#     parser.add_argument('--save_best', action='store_true', help='Save the best checkpoint')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     cfg = load_config(args.config)

#     torch.cuda.empty_cache()
    
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
    
#     if os.path.exists(args.model):
#         model_path = args.model
#     else:
#         model_path = os.path.join(script_dir, 'models', args.model)

#     print(f"Loading model architecture: {args.model}")
    

#     if "yolo" in args.model.lower():
#         model = YOLO(model_path)
#     else:
#         model = RTDETR(model_path)

#     print(f"--- Starting Training for {args.epochs} epochs ---")
    

#     train_results = model.train(
#         data=args.dataset_config,
#         epochs=args.epochs,
#         imgsz=args.img_size,
#         batch=args.batch_size,
#         lr0=args.lr,
#         project=os.path.join(cfg.get('output_dir', 'runs'), args.project),
#         name=args.name or f"{os.path.splitext(os.path.basename(args.model))[0]}_{args.project}",
#         save_period=args.save_period,
#         save=args.save_best,
#         plots=True 
#     )
    
#     print("Training finished.")
    
  
#     best_weight_path = os.path.join(train_results.save_dir, 'weights', 'best.pt')
#     if not os.path.exists(best_weight_path):
#         print(f"Best weights not found at {best_weight_path}, trying 'last.pt'...")
#         best_weight_path = os.path.join(train_results.save_dir, 'weights', 'last.pt')

#     print(f"\n--- Loading Best Model for Validation: {best_weight_path} ---")
    

#     if "yolo" in args.model.lower():
#         val_model = YOLO(best_weight_path)
#     else:
#         val_model = RTDETR(best_weight_path)

#     metrics = val_model.val(data=args.dataset_config, split='val')


#     print("\n--- Detailed Validation Metrics ---")

#     print(f"mAP@50:    {metrics.box.map50:.4f}")
#     print(f"mAP@50-95: {metrics.box.map:.4f}")
    
   
#     mean_precision = metrics.box.mp
#     mean_recall = metrics.box.mr
#     print(f"Precision: {mean_precision:.4f}")
#     print(f"Recall:    {mean_recall:.4f}")

    
#     if (mean_precision + mean_recall) > 0:
#         f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
#         print(f"F1 Score:  {f1_score:.4f}")
    

#     try:
#         cm = metrics.confusion_matrix.matrix
#         print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
#         print(cm)
#         print(f"\nCharts saved to: {metrics.save_dir}")
#     except AttributeError:
#         print("\nCould not retrieve confusion matrix object directly.")

# if __name__ == "__main__":
#     main()



""""

New code with iou,confusion matrix

"""



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
    
    parser.add_argument('--config', type=str, default='config/paths/cluster.yaml', help='Path to general config')
    parser.add_argument('--dataset_config', type=str, default='config/dataset/heico_endo_test.yaml', help='Path to data YAML')
    parser.add_argument('--model', type=str, default='yolo11l.pt', help='Base model to start training from')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial Learning rate')

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

    print(f"--- Starting Training for {args.epochs} epochs ---")
    

    train_results = model.train(
        data=args.dataset_config,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.lr,
        project=os.path.join(cfg.get('output_dir', 'runs'), args.project),
        name=args.name or f"{os.path.splitext(os.path.basename(args.model))[0]}_{args.project}",
        save_period=args.save_period,
        save=args.save_best,
        plots=True 
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
    print(f"mAP@75:    {metrics.box.map75:.4f} (IoU threshold = 0.75) # Added")
    
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
        # Check if the metrics contain the necessary arrays
        if hasattr(metrics.box, 'p') and metrics.box.p is not None and metrics.box.p.size > 1:
            class_names = val_model.names 
            class_precision = metrics.box.p 
            class_recall = metrics.box.r 
            class_map50 = metrics.box.map_class_50
            class_map = metrics.box.map_class_95 
            
            # Print Class-Specific Table
            print(f"{'Class':<20} {'P':<10} {'R':<10} {'mAP@50':<10} {'mAP@50-95':<10}")
            print("-" * 60)
            for i, name in class_names.items():
                print(f"{name:<20} {class_precision[i]:<10.4f} {class_recall[i]:<10.4f} {class_map50[i]:<10.4f} {class_map[i]:<10.4f}")
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
        # Normalize by true counts (rows) to show recall/TPrate for each class
        normalized_cm = cm / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) 
        print(normalized_cm)
        
        print(f"\nCharts saved to: {metrics.save_dir}")
    except AttributeError:
        print("\nCould not retrieve confusion matrix object directly or matrix is empty.")
    except Exception as e:
        print(f"An unexpected error occurred while printing the confusion matrix: {e}")


if __name__ == "__main__":
    main()



