# infer_latlon.py
import os, re, argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from torch.utils.data import DataLoader

# ─────────────── Configuration ───────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH  = "latlon_multitask.pt"
ARCH       = "convnext_large"     # same as your GeoMT backbone
BAD_VAL_IDS = {95,145,146,158,159,160,161}

# Bounding‐box filter for train‐set cleaning
LAT_MIN, LAT_MAX = 218000, 222000
LON_MIN, LON_MAX = 140500, 146000

def sorted_image_list(folder):
    """Return list of files 'img_XXXX.*' sorted by integer XXXX."""
    files = [f for f in os.listdir(folder)
             if re.match(r"img_(\d+)\.(?:jpg|jpeg|png)$", f, re.I)]
    files.sort(key=lambda fn: int(re.search(r"img_(\d+)", fn).group(1)))
    return files

class GeoMT(nn.Module):
    """Multi‐task: region_cls + lat/lon regression."""
    def __init__(self, arch=ARCH):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        self.cls = nn.Linear(feat, 15)
        self.reg = nn.Linear(feat, 2)
    def forward(self, x):
        f = self.backbone(x)
        return self.cls(f), self.reg(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=32, help="Batch size for inference")
    args = p.parse_args()

    train_df = pd.read_csv("labels_train.csv")
    mask = (
        (train_df.latitude  >= LAT_MIN) & (train_df.latitude  <= LAT_MAX) &
        (train_df.longitude >= LON_MIN) & (train_df.longitude <= LON_MAX)
    )
    df_clean = train_df[mask]
    lat_mu, lat_sd = df_clean.latitude.mean(), df_clean.latitude.std()
    lon_mu, lon_sd = df_clean.longitude.mean(), df_clean.longitude.std()

    transform = T.Compose([
        T.Resize(int(224*1.15)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

   
    model = GeoMT().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    val_df = pd.read_csv("labels_val.csv")
  
    val_df["num_id"] = val_df.filename.str.extract(r"(\d+)").astype(int)
    val_df = val_df[~val_df.num_id.isin(BAD_VAL_IDS)].reset_index(drop=True)
    val_files = val_df.filename.tolist()

    val_preds_z = []
    val_ids_num = []
    with torch.no_grad():
        for fname in val_files:
            img = Image.open(os.path.join("images_val", fname)).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)
            _, coord_z = model(x)
            val_preds_z.append(coord_z[0].cpu().numpy())
            val_ids_num.append(int(re.search(r"img_(\d+)", fname).group(1)))

    val_preds_z = np.stack(val_preds_z, axis=0)

    lat_val_raw = val_preds_z[:,0]*lat_sd + lat_mu
    lon_val_raw = val_preds_z[:,1]*lon_sd + lon_mu

    true_raw_lat = val_df.latitude.to_numpy()
    true_raw_lon = val_df.longitude.to_numpy()
    true_z_lat   = (true_raw_lat - lat_mu)/lat_sd
    true_z_lon   = (true_raw_lon - lon_mu)/lon_sd

    mse_z_lat = np.mean((val_preds_z[:,0] - true_z_lat)**2)
    mse_z_lon = np.mean((val_preds_z[:,1] - true_z_lon)**2)
    mse_z     = (mse_z_lat + mse_z_lon)/2

    mse_raw_lat = np.mean((lat_val_raw - true_raw_lat)**2)
    mse_raw_lon = np.mean((lon_val_raw - true_raw_lon)**2)
    mse_raw     = (mse_raw_lat + mse_raw_lon)/2

    print("VAL Metrics:")
    print(f"  MSE (z‐space):      lat = {mse_z_lat:.6f}, lon = {mse_z_lon:.6f}, avg = {mse_z:.6f}")
    print(f"  MSE (original):     lat = {mse_raw_lat:.2f}, lon = {mse_raw_lon:.2f}, avg = {mse_raw:.2f}")

    test_files = sorted_image_list("images_test")
    test_preds_raw = []
    test_ids_num = []
    with torch.no_grad():
        for fname in test_files:
            img = Image.open(os.path.join("images_test", fname)).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)
            _, coord_z = model(x)
            cz = coord_z[0].cpu().numpy()
            lat_r = cz[0]*lat_sd + lat_mu
            lon_r = cz[1]*lon_sd + lon_mu
            test_preds_raw.append((lat_r, lon_r))
            test_ids_num.append(369 + int(re.search(r"img_(\d+)", fname).group(1)))

    df_out = pd.DataFrame({
        "id":        val_ids_num + test_ids_num,
        "Latitude":  np.round(lat_val_raw).astype(int).tolist()  +
                     [int(round(p[0])) for p in test_preds_raw],
        "Longitude": np.round(lon_val_raw).astype(int).tolist()  +
                     [int(round(p[1])) for p in test_preds_raw],
    })
    df_out.to_csv("solutions.csv", index=False)
    print("Saved solutions.csv (id, Latitude, Longitude)")

if __name__ == "__main__":
    main()
