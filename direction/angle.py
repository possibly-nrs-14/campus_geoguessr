import math, argparse, tqdm, pandas as pd
from copy import deepcopy
from PIL import Image
import torch, timm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
def build_tfm(train, sz=224):
    if train:
        return T.Compose([
            T.Resize(int(sz*1.15)),
            T.CenterCrop(sz),
            T.ColorJitter(0.25,0.25,0.25,0.1),
            T.ToTensor(), norm
        ])
    return T.Compose([
        T.Resize(int(sz*1.15)), T.CenterCrop(sz),
        T.ToTensor(), norm])

class AngleDS(Dataset):
    def __init__(self, csv, root, train, size=224):
        self.df = pd.read_csv(csv)
        self.root, self.train = root, train
        self.tfm = build_tfm(train, size)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(f"{self.root}/{r.filename}").convert("RGB")
        img = self.tfm(img)
        ang = r.angle % 360
        vec = torch.tensor([math.sin(math.radians(ang)),
                            math.cos(math.radians(ang))], dtype=torch.float32)
        return img, vec, ang, r.filename

# ───────────────────────────── model
class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_large_patch4_window7_224.ms_in22k',  
            pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 2)
    def forward(self, x):
        return self.head(self.backbone(x))

def angle_from_vec(v):
    deg = (torch.rad2deg(torch.atan2(v[:,0], v[:,1])) + 360) % 360
    return deg

def maae(pred, true):
    d = (pred - true).abs()
    d = torch.minimum(d, 360 - d)
    return d.mean().item()

def train_model(epochs, bs):
    ds_tr = AngleDS("labels_train.csv", "images_train", True, 224)
    ds_vl = AngleDS("labels_val.csv",   "images_val",   False, 224)
    tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(ds_vl, batch_size=bs, shuffle=False,
                    num_workers=4, pin_memory=True)

    net = RegNet().to(DEVICE)
    opt = AdamW(net.parameters(), lr=1e-4, weight_decay=0.05)
    sched = SequentialLR(opt, [
        LinearLR(opt, 0.2, 1.0, total_iters=5),
        CosineAnnealingLR(opt, T_max=epochs-5)
    ], milestones=[5])
    best, best_w = math.inf, None
    cosine_loss = nn.CosineEmbeddingLoss()

    for ep in range(1, epochs+1):
        net.train()
        tot=0
        n=0
        for x, vec, *_ in tqdm.tqdm(tr, leave=False):
            x, vec = x.to(DEVICE), vec.to(DEVICE)
            out = net(x)
            out_norm = nn.functional.normalize(out, dim=1)
            tgt_norm = nn.functional.normalize(vec, dim=1)
            loss = cosine_loss(out_norm, tgt_norm,
                               torch.ones(len(x), device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()

        net.eval()
        pred, true = [], []
        with torch.no_grad():
            for x, _, ang, _ in vl:
                v = net(x.to(DEVICE)).cpu()
                pred.append(angle_from_vec(v))
                true.append(ang.clone().detach())
        pred = torch.cat(pred); true = torch.cat(true)
        val_maae = maae(pred, true)
        print(f"Epoch {ep}/{epochs}  val MAAE {val_maae:.2f}°")
        if val_maae < best:
            best, best_w = val_maae, deepcopy(net.state_dict())

    torch.save(best_w, "angle_regressor.pt")
    print(f"✓ Best val MAAE = {best:.2f}°, model saved.")
    return best_w, ds_vl, vl

@torch.no_grad()
def save_csv(state_dict, ds_vl, dl):
    net = RegNet().to(DEVICE)
    net.load_state_dict(state_dict); net.eval()
    fn, truth, pred = [], [], []
    for x, _, ang, fname in dl:
        v = net(x.to(DEVICE)).cpu()
        pred.extend(angle_from_vec(v).tolist())
        truth.extend(ang)
        fn.extend(fname)
    df = pd.DataFrame({
        "filename": fn,
        "true_angle": truth,
        "pred_angle": pred
    })
    df.to_csv("angle_val_reg.csv", index=False)
    print("CSV saved → angle_val_reg.csv")
    print(f"Final MAAE = {maae(torch.tensor(pred), torch.tensor(truth)):.2f}°")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs",     type=int, default=16)
    args = ap.parse_args()

    best_state, ds_vl, vl_dl = train_model(args.epochs, args.bs)
    save_csv(best_state, ds_vl, vl_dl)
