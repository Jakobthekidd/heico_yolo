import os
import hydra
from omegaconf import OmegaConf
import yaml
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import torch

def dataframe_column_to_txt(df, dataset_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for value in df["filename"].tolist():
            value = os.path.join(dataset_path, "images", value)
            f.write(value + "\n")
    return output_path

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


def compute_dice(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return (2 * interArea) / float(areaA + areaB)

def evaluate_predictions(model, val_df, image_root):
    ious = []
    dices = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        img_path = os.path.join(image_root, row["image_relpath"])
        results = model.predict(source=img_path, verbose=False)

        preds = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        gts = np.array(row["bboxes_xyxy"])  # must be list of [x1,y1,x2,y2]

        if len(preds) == 0 or len(gts) == 0:
            continue

        # simple greedy pairing
        for gt in gts:
            best_iou = 0
            best_dice = 0
            for pr in preds:
                iou = compute_iou(gt, pr)
                dice = compute_dice(gt, pr)
                best_iou = max(best_iou, iou)
                best_dice = max(best_dice, dice)
            ious.append(best_iou)
            dices.append(best_dice)

    return np.mean(ious), np.mean(dices)

@hydra.main(config_path="config", config_name="config")
def train(CFG):
    # print(CFG)
    print("\n========== LOADING DATA ==========")
    df = pd.read_csv(CFG.data.csv_path)

    # assume df includes columns: image_relpath, fold, bboxes_xyxy (eval), etc.
    # df["bboxes_xyxy"] = df["bboxes_xyxy"].apply(eval)

    output_root = os.path.join(CFG.paths.output_dir, CFG.exp)
    os.makedirs(output_root, exist_ok=True)

    print("========== STARTING 5-FOLD TRAINING ==========")

    for fold in CFG.trn_fold:
        print(f"\n================= FOLD {fold} =================")

        fold_dir = os.path.join(output_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_df = df[df["set"] == "train"].reset_index(drop=True)
        test_df   = df[df["set"] == "test"].reset_index(drop=True)
        
        # Split data
        train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        val_df   = train_df[train_df["fold"] == fold].reset_index(drop=True)

        print(f"Train samples: {len(train_df)}"
              f", Val samples: {len(val_df)}"
                f", Test samples: {len(test_df)}")

        # Create txt files
        txt_dir = os.path.join(fold_dir, "txts")
        os.makedirs(txt_dir, exist_ok=True)

        train_txt = dataframe_column_to_txt(train_df, CFG.paths.dataset_path, os.path.join(txt_dir, "train.txt"))
        val_txt   = dataframe_column_to_txt(val_df,   CFG.paths.dataset_path, os.path.join(txt_dir, "val.txt"))
        test_txt  = dataframe_column_to_txt(test_df,   CFG.paths.dataset_path, os.path.join(txt_dir, "test.txt"))

        original_cwd = hydra.utils.get_original_cwd()
        yolo_config_path = os.path.join(original_cwd, "config/yolo_config.yaml")

        # Prepare YOLO config
        with open(yolo_config_path, "r") as f:
            ycfg = yaml.safe_load(f)

        ycfg["train"] = train_txt
        ycfg["val"]   = val_txt
        ycfg["test"]  = test_txt
        ycfg["path"]  = CFG.paths.dataset_path  # so weights save under fold dir

        updated_cfg_path = os.path.join(txt_dir, "yolo_config.yaml")
        with open(updated_cfg_path, "w") as f:
            yaml.dump(ycfg, f)

        # Train YOLO
        print("Training YOLO model ...")
        # model = YOLO(CFG.model_name)

        # model.train(
        #     data=updated_cfg_path,
        #     epochs=CFG.epochs,
        #     imgsz=CFG.imgsz,
        #     batch=CFG.batch,
        #     seed=CFG.seed,
        #     project=CFG.wandb_project,
        #     name=CFG.exp,
        # )

        # # Evaluate IOU & Dice
        # print("Running evaluation ...")
        # best_model_path = os.path.join(fold_dir, "run/weights/best.pt")
        # best_model = YOLO(best_model_path)

        # mean_iou, mean_dice = evaluate_predictions(best_model, val_df, CFG.data.image_root)

        # print(f"Fold {fold} IoU:  {mean_iou:.4f}")
        # print(f"Fold {fold} Dice: {mean_dice:.4f}")

        # with open(os.path.join(fold_dir, "metrics.txt"), "w") as f:
        #     f.write(f"mean_IOU: {mean_iou}\n")
        #     f.write(f"mean_DICE: {mean_dice}\n")

        # torch.cuda.empty_cache()

    print("\n========== FINISHED ==========")

    # Print required CFG parameters
    print("\n===== REQUIRED CFG PARAMETERS =====")
    print(OmegaConf.to_yaml(CFG))


if __name__ == "__main__":
    train()
