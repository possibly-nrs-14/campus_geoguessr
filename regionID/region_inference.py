# infer_convnext.py
import os
import re
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CKPT_PATH = "best_region_model.pt"  


NUM_CLASSES = 15

# Folders
VAL_DIR  = "images_val"
TEST_DIR = "images_test"

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


def sorted_image_list(folder):
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    return sorted(files, key=lambda fn: int(re.search(r"(\d+)", fn).group(1)))

def load_convnext(checkpoint_path):
    model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_feat, NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def infer_folder(model, folder, start_id):
    files = sorted_image_list(folder)
    ids, preds, fnames = [], [], []
    for idx, fname in enumerate(files, start=start_id):
        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        out = model(x)
        pred = int(out.argmax(1).item()) + 1 
        ids.append(idx)
        preds.append(pred)
        fnames.append(fname)
    return ids, preds, fnames


def main():

    model = load_convnext(CKPT_PATH)

    val_ids,  val_preds,  val_files  = infer_folder(model, VAL_DIR,  0)
    test_ids, test_preds, test_files = infer_folder(model, TEST_DIR, len(val_ids))

    labels_df = pd.read_csv("labels_val.csv")
    true_map  = dict(zip(labels_df.filename, labels_df.Region_ID.astype(int)))
    true_vals = [true_map[f] for f in val_files]
    val_acc = sum(p==t for p,t in zip(val_preds, true_vals)) / len(true_vals)
    print(f"Validation accuracy: {val_acc*100:.2f}%")

    all_ids   = val_ids  + test_ids
    all_preds = val_preds + test_preds

    out_df = pd.DataFrame({
        "id":         all_ids,
        "Region_ID":  all_preds
    })
    out_df.to_csv("solutions.csv", index=False)
    print("Saved predictions → solutions.csv")

if __name__ == "__main__":
    main()
