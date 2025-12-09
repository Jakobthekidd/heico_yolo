# import os
# import torch
# from ultralytics import YOLO, RTDETR

# # Removed: from config.wandb_setup import start_wandb
# # Removed: def load_config(config_path):
# # Removed: def parse_args():

# # 1. ⚙️ Define Configuration Class (Replaces ArgParse)
# class Config:
#     """Class to hold all training parameters, replacing argparse."""
#     # Training Parameters
#     dataset_config = '/home/s124z/Code/YOLO/heico_code/config/dataset/heico_end.yaml' # Path to data YAML
#     model = '/home/s124z/Code/YOLO/models/yolo11s.pt'                             # Base model to start training from
#     epochs = 50
#     batch_size = 16
#     img_size = 640
#     lr = 0.0001
    
#     # Logging / Saving Parameters
#     project = 'endoscapes_detect'
#     name = None # Set to None for auto-naming, or a specific string like 'my_test_run'
#     save_period = 0
#     save_best = True  # Changed default to True, common for local training
    
# # 2. Main Training Function
# def main():
#     # Load parameters from the Config class
#     args = Config()
    
#     # Local PC Adjustment: Define the local output directory
#     LOCAL_OUTPUT_DIR = '/home/s124z/Code/YOLO/heico_dataset/runs' 
    
#     # Initialize WandB (Simplified, requires 'wandb' to be installed)
  
    
#     # Device handling
#     torch.cuda.empty_cache()
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # Determine the model path (uses local file or downloads if not found)
#     model_path = args.model
#     if not os.path.exists(model_path):
#         local_model_path = os.path.join(script_dir, 'models', args.model)
#         if os.path.exists(local_model_path):
#             model_path = local_model_path
#         # Else, leave as args.model, and Ultralytics will download it
        
#     print(f"Loading model architecture: {args.model}")

#     # Detect model type and load
#     if "yolo" in args.model.lower():
#         model = YOLO(model_path)
#     else:
#         model = RTDETR(model_path)

#     print(f"--- Starting Training for {args.epochs} epochs ---")
    
#     # Start Training: Parameters are passed directly to model.train()
#     results = model.train(
#         data=args.dataset_config,
#         epochs=args.epochs,
#         imgsz=args.img_size,
#         batch=args.batch_size,
#         lr0=args.lr,
#         project=os.path.join(LOCAL_OUTPUT_DIR, args.project),
#         name=args.name or f"{os.path.splitext(os.path.basename(args.model))[0]}_{args.project}",
#         save_period=args.save_period,
#         save=args.save_best,
#     )
    
#     print("Training finished.")
#     print(f"Best model saved to: {results.save_dir}/weights/best.pt")

# if __name__ == "__main__":
#     main()

import os
import sys

# ==============================================================================
# 🔇 THE SILENCER (Must be at the VERY TOP)
# ==============================================================================
# 1. Import the REAL tqdm first (avoids the __spec__ error you saw earlier)
import tqdm

# 2. Define a silent wrapper
class MockTQDM:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        if self.iterable:
            yield from self.iterable
    def __getattr__(self, name):
        # Absorb calls like .update(), .set_description(), etc.
        return lambda *args, **kwargs: None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

# 3. Overwrite the class inside the module BEFORE Ultralytics loads
# Ultralytics will now inadvertently import our MockTQDM instead of the real one
tqdm.tqdm = MockTQDM
tqdm.auto.tqdm = MockTQDM
# ==============================================================================


# NOW we import the rest. Ultralytics will pick up the "broken" tqdm.
import os
import sys

# ==============================================================================
# 🔇 THE SILENCER (Final Fix)
# ==============================================================================
import tqdm
# We must explicitly import the auto submodule to patch it
try:
    import tqdm.auto
except ImportError:
    pass

class MockTQDM:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        if self.iterable:
            yield from self.iterable
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

# Overwrite the class in the real module
tqdm.tqdm = MockTQDM

# Overwrite the auto submodule if it exists (this fixes your AttributeError)
if hasattr(tqdm, 'auto'):
    tqdm.auto.tqdm = MockTQDM
# ==============================================================================


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
    parser.add_argument('--config', type=str, default='config/paths/local.yaml')
    parser.add_argument('--dataset_config', type=str, default='config/dataset/heico_end.yaml')
    parser.add_argument('--model', type=str, default='yolo11s.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--project', type=str, default='heico_')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--save_period', type=int, default=0)
    parser.add_argument('--no_save', action='store_false', dest='save_best', help='Disable saving')
    parser.set_defaults(save_best=True)
    return parser.parse_args()

def print_epoch_start(trainer):
    print(f"Epoch {trainer.epoch + 1}/{trainer.epochs} running...")

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

    model.add_callback("on_train_epoch_start", print_epoch_start)

    print(f"--- Starting Training for {args.epochs} epochs ---")
    
    # Start Training
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
        plots=True,       
        verbose=False     
    )
    
    print("Training finished.")
    
    # ==========================================
    #      POST-TRAINING METRICS & EVALUATION
    # ==========================================
    
    best_weight_path = os.path.join(train_results.save_dir, 'weights', 'best.pt')
    if not os.path.exists(best_weight_path):
        print(f"Best weights not found, trying 'last.pt'...")
        best_weight_path = os.path.join(train_results.save_dir, 'weights', 'last.pt')

    print(f"\n--- Loading Best Model for Validation: {best_weight_path} ---")
    
    if "yolo" in args.model.lower():
        val_model = YOLO(best_weight_path)
    else:
        val_model = RTDETR(best_weight_path)

    metrics = val_model.val(data=args.dataset_config, split='val')

    print("\n--- Detailed Validation Metrics ---")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    
    mean_precision = metrics.box.mp
    mean_recall = metrics.box.mr
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall:    {mean_recall:.4f}")

    if (mean_precision + mean_recall) > 0:
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        print(f"F1 Score:  {f1_score:.4f}")
    
    try:
        cm = metrics.confusion_matrix.matrix
        print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
        print(cm)
        print(f"\nCharts saved to: {metrics.save_dir}")
    except AttributeError:
        print("\nCould not retrieve confusion matrix object directly.")

if __name__ == "__main__":
    main()