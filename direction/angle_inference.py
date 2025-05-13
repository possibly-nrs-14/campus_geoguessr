# infer_angle.py
import os
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T
import timm
from torch.utils.data import DataLoader, Dataset

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "angle_regressor.pt"  
ARCH      = "swin_large_patch4_window7_224.ms_in22k"
IMG_SIZE  = 224


norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
transform = T.Compose([
    T.Resize(int(IMG_SIZE*1.15)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    norm,
])

def sorted_images(folder):
    exts = (".jpg",".jpeg",".png")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort(key=lambda fn: int(re.search(r"(\d+)",fn).group(1)))
    return files

def mean_abs_angular_error(pred, true):
    diff = torch.abs(pred - true)
    diff = torch.minimum(diff, 360 - diff)
    return diff.mean().item()

class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(ARCH, pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 2)
    def forward(self, x):
        return self.head(self.backbone(x))

@torch.no_grad()
def infer_folder(model, folder, start_id):
    files = sorted_images(folder)
    ids, preds, fnames = [], [], []
    for i, fname in enumerate(files, start=start_id):
        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        out = model(x).cpu()
        ang = (torch.rad2deg(torch.atan2(out[0,0], out[0,1])) + 360) % 360
        ids.append(i)
        preds.append(ang.item())
        fnames.append(fname)
    return ids, preds, fnames

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=32, help="batch size (unused)")
    args = p.parse_args()
    model = RegNet().to(DEVICE).eval()
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)
    val_folder = "images_val"
    val_ids,  val_preds,  val_files  = infer_folder(model, val_folder, 0)
    df_val = pd.read_csv("labels_val.csv")
    true_map = {row.filename: row.angle for _,row in df_val.iterrows()}
    true_vals = [true_map[f] for f in val_files]
    mae_val   = mean_abs_angular_error(torch.tensor(val_preds), torch.tensor(true_vals))
    print(f"Validation MAAE = {mae_val:.2f}°")

    test_folder = "images_test"
    test_ids, test_preds, test_files = infer_folder(model, test_folder, len(val_ids))

   
    all_ids   = val_ids + test_ids
    all_preds = val_preds + test_preds

    out = pd.DataFrame({
        "id":   all_ids,
        "angle": all_preds
    })
    out.to_csv("solutions.csv", index=False)
    print("Saved predictions → solutions.csv  (id,angle)")

if __name__ == "__main__":
    main()
