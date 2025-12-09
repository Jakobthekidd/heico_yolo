import os
import yaml
import argparse
import torch
from ultralytics import YOLO, RTDETR
from config.wandb_setup import start_wandb
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    # Training Configuration
    parser.add_argument('--config', type=str, default='config/paths/cluster.yaml', help='Path to general config')
    parser.add_argument('--dataset_config', type=str, default='config/dataset/endoscapes.yaml', help='Path to data YAML')
    parser.add_argument('--model', type=str, default='yolo11x.pt', help='Base model to start training from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial Learning rate')
    # Logging / Saving
    parser.add_argument('--project', type=str, default='endoscapes_detect', help='Project name for save directory')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--save_period', type=int, default=0)
    parser.add_argument('--save_best', action='store_true', help='Save the best checkpoint')
    return parser.parse_args()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    # Initialize WandB
    start_wandb(args)
    # Device handling
    torch.cuda.empty_cache()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check if model is a path or a name, handle accordingly
    if os.path.exists(args.model):
        model_path = args.model
    else:
        model_path = os.path.join(script_dir, 'models', args.model)
    print(f"Loading model architecture: {args.model}")
    # Detect model type
    if "yolo" in args.model.lower():
        model = YOLO(model_path)
    else:
        model = RTDETR(model_path)
    print(f"--- Starting Training for {args.epochs} epochs ---")
    # Start Training
    results = model.train(
        data=args.dataset_config,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.lr,
        project=os.path.join(cfg.get('output_dir', 'runs'), args.project),
        name=args.name or f"{os.path.splitext(os.path.basename(args.model))[0]}_{args.project}",
        save_period=args.save_period,
        save=args.save_best,
    )
    print("Training finished.")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
if __name__ == "__main__":
    main()
