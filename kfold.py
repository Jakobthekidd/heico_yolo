import os
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO, RTDETR

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_train_image_paths(yaml_path):
    """
    Parses the dataset YAML and ONLY loads images from the 'train' key.
    """
    cfg = load_config(yaml_path)
    base_path = Path(cfg.get('path', ''))
    
    # helper to resolve paths
    def resolve_path(p):
        # If absolute, return as is
        if os.path.isabs(p):
            return Path(p)
        # If relative, join with the 'path' in yaml or current dir
        return (base_path / p).resolve()

    image_files = []
    
    # --- STRICTLY ONLY LOOK AT 'train' KEY ---
    if 'train' in cfg:
        train_path = cfg['train']
        
        # Handle case where train might be a list of paths
        paths_to_check = train_path if isinstance(train_path, list) else [train_path]
        
        for p in paths_to_check:
            resolved_p = resolve_path(p)
            
            if resolved_p.is_dir():
                # Glob all common image formats
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.tif']
                for ext in extensions:
                    image_files.extend(list(resolved_p.rglob(ext)))
                    
            elif resolved_p.is_file() and resolved_p.suffix == '.txt':
                # If it points to a txt file list
                with open(resolved_p, 'r') as f:
                    image_files.extend([Path(line.strip()) for line in f.readlines()])
    else:
        raise ValueError("The dataset YAML must contain a 'train' key.")
    
    # Remove duplicates and convert to string for compatibility
    return sorted(list(set([str(x) for x in image_files])))

def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold Cross Validation (Train Data Only)")
    parser.add_argument('--config', type=str, default='config/paths/cluster.yaml', help='Path to general config')
    parser.add_argument('--dataset_config', type=str, default='config/dataset/heico_end.yaml', help='Path to data YAML')
    parser.add_argument('--model', type=str, default='yolo11l.pt', help='Base model to start training from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--project', type=str, default='heico_cv', help='Project name')
    return parser.parse_args()

def main():
    args = parse_args()
    base_cfg = load_config(args.dataset_config)
    class_names = base_cfg.get('names', {})
    
    # 1. Gather ONLY Training Data
    print(f"--- Loading images from 'train' path in {args.dataset_config} ---")
    all_images = np.array(get_train_image_paths(args.dataset_config))
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in the train path! Check {args.dataset_config}")
    print(f"Total training images found: {len(all_images)}")

    # 2. Initialize K-Fold
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_metrics = []

    # 3. Loop through Folds
    for i, (train_index, val_index) in enumerate(kf.split(all_images)):
        fold_idx = i + 1
        print(f"\n{'='*30}")
        print(f"   STARTING FOLD {fold_idx}/{args.k_folds}")
        print(f"{'='*30}")

        # Split the single 'train' folder into a new train/val split for this fold
        train_files = all_images[train_index]
        val_files = all_images[val_index]

        # --- Create Temporary Fold Files ---
        save_path = Path(args.project) / f'fold_{fold_idx}'
        save_path.mkdir(parents=True, exist_ok=True)
        
        train_txt_path = save_path / 'train_fold.txt'
        val_txt_path = save_path / 'val_fold.txt'
        
        # Write .txt files listing image paths
        with open(train_txt_path, 'w') as f:
            f.write('\n'.join(train_files))
            
        with open(val_txt_path, 'w') as f:
            f.write('\n'.join(val_files))

        # Create temporary YAML config for this specific fold
        fold_yaml_path = save_path / f'data_fold_{fold_idx}.yaml'
        fold_yaml_data = {
            'path': str(save_path.absolute()), 
            'train': str(train_txt_path.absolute()),
            'val': str(val_txt_path.absolute()), # This fold's validation comes from the original train folder
            'names': class_names,
            'nc': len(class_names)
        }
        
        with open(fold_yaml_path, 'w') as f:
            yaml.dump(fold_yaml_data, f)

        # --- Initialize Model ---
        torch.cuda.empty_cache()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Handle model path
        if os.path.exists(args.model):
            model_path = args.model
        else:
            model_path = os.path.join(script_dir, 'models', args.model)
        
        if "yolo" in args.model.lower():
            model = YOLO(model_path)
        else:
            model = RTDETR(model_path)

        # --- Train ---
        print(f"Training Fold {fold_idx}...")
        model.train(
            data=str(fold_yaml_path),
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            lr0=args.lr,
            project=args.project,
            name=f"fold_{fold_idx}",
            plots=True,
            exist_ok=True 
        )

        # --- Validate ---
        # We must reload the best model from THIS fold to validate it
        best_weight_path = os.path.join(args.project, f"fold_{fold_idx}", 'weights', 'best.pt')
        if not os.path.exists(best_weight_path):
             best_weight_path = os.path.join(args.project, f"fold_{fold_idx}", 'weights', 'last.pt')
        
        print(f"Validating Fold {fold_idx}...")
        if "yolo" in args.model.lower():
            val_model = YOLO(best_weight_path)
        else:
            val_model = RTDETR(best_weight_path)

        # Validate specifically on the split we reserved for validation in this fold
        metrics = val_model.val(data=str(fold_yaml_path))
        
        # Calculate F1
        mp = metrics.box.mp
        mr = metrics.box.mr
        f1 = 2 * (mp * mr) / (mp + mr + 1e-16)

        fold_res = {
            'fold': fold_idx,
            'map50': metrics.box.map50,
            'map50-95': metrics.box.map,
            'precision': mp,
            'recall': mr,
            'f1': f1
        }
        fold_metrics.append(fold_res)

    # ==========================================
    #      AGGREGATE RESULTS
    # ==========================================
    print("\n" + "="*40)
    print(f"   FINAL K-FOLD RESULTS (Source: Train Data Only)")
    print("="*40)
    
    avg_map50 = np.mean([x['map50'] for x in fold_metrics])
    avg_map = np.mean([x['map50-95'] for x in fold_metrics])
    avg_p = np.mean([x['precision'] for x in fold_metrics])
    avg_r = np.mean([x['recall'] for x in fold_metrics])
    avg_f1 = np.mean([x['f1'] for x in fold_metrics])

    print(f"{'Fold':<6} | {'mAP@50':<10} | {'mAP@50-95':<10} | {'F1':<10}")
    print("-" * 46)
    for m in fold_metrics:
        print(f"{m['fold']:<6} | {m['map50']:<10.4f} | {m['map50-95']:<10.4f} | {m['f1']:<10.4f}")
    print("-" * 46)
    print(f"{'AVG':<6} | {avg_map50:<10.4f} | {avg_map:<10.4f} | {avg_f1:<10.4f}")

if __name__ == "__main__":
    main()