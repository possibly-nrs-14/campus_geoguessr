
import os, argparse, math, tqdm, pandas as pd
import numpy as np
from copy import deepcopy
from PIL import Image
import torch, timm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

BAD_VAL_IDS = {95, 145, 146, 158, 159, 160, 161}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────── dataset
def make_tfm(train,size):
    if train:
        return T.Compose([
            T.Resize(int(size*1.15)),
            T.RandomResizedCrop(size,scale=(0.85,1.0)),
            T.ColorJitter(0.25,0.25,0.25,0.1),
            T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return T.Compose([
        T.Resize(int(size*1.15)),T.CenterCrop(size),
        T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class LLDS(Dataset):
    def __init__(self, csv, root, train, size):
        df = pd.read_csv(csv)
        if not train:
            ids = df["filename"].str.extract(r"(\d+)")[0].astype(int)
            df = df[~ids.isin(BAD_VAL_IDS)].reset_index(drop=True)
        if train:
            mask = (
            (df["latitude"]  >= 218000) & (df["latitude"]  <= 222000) &
            (df["longitude"] >= 140500) & (df["longitude"] <= 146000)
            )
            df = df[mask].reset_index(drop=True)
        self.df = df
        self.root = root
        self.train = train
        self.size = size
        self.tfm = make_tfm(train, size)

        self.lat_mu, self.lon_mu = self.df.latitude.mean(),  self.df.longitude.mean()
        self.lat_sd, self.lon_sd = self.df.latitude.std(),     self.df.longitude.std()

    def set_size(self, size):
        self.size = size
        self.tfm = make_tfm(self.train, size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root, r.filename)).convert("RGB")
        img = self.tfm(img)

        region = int(r.Region_ID) - 1
        coord = torch.tensor([
            (r.latitude  - self.lat_mu) / self.lat_sd,
            (r.longitude - self.lon_mu) / self.lon_sd
        ], dtype=torch.float32)

        return img, region, coord


class GeoMT(nn.Module):
    def __init__(self,arch='convnext_large'):
        super().__init__()
        self.backbone=timm.create_model(arch,pretrained=True,num_classes=0)
        f=self.backbone.num_features
        self.cls=nn.Linear(f,15)
        self.reg=nn.Linear(f,2)
    def forward(self,x):
        f=self.backbone(x)
        return self.cls(f),self.reg(f)
def run_epoch(net,dl,opt=None,λ=15.0):
    ce=nn.CrossEntropyLoss()
    mse=nn.MSELoss()
    train=opt is not None
    net.train() if train else net.eval()
    tot_reg,tot=0,0
    with torch.set_grad_enabled(train):
        for x,reg,coord in tqdm.tqdm(dl,leave=False):
            x,reg,coord=x.to(DEVICE),reg.to(DEVICE),coord.to(DEVICE)
            o_cls,o_reg=net(x)
            loss=ce(o_cls,reg)+λ*mse(o_reg,coord)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_reg+=mse(o_reg,coord).item()*x.size(0); tot+=x.size(0)
    return tot_reg/tot


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--bs",type=int,default=16)
    args=ap.parse_args()

    ds_tr=LLDS("labels_train.csv","images_train",True,224)
    ds_vl=LLDS("labels_val.csv","images_val",False,224)

    def mk(dl_ds,shuffle): return DataLoader(dl_ds,batch_size=args.bs,
                     shuffle=shuffle,num_workers=4,pin_memory=True)
    tr_dl=mk(ds_tr,True); vl_dl=mk(ds_vl,False)

    net=GeoMT().to(DEVICE)
    head,body=[],[]
    for n,p in net.named_parameters():
        (head if 'cls' in n or 'reg' in n else body).append(p)
    opt=AdamW([{"params":body,"lr":1e-4},{"params":head,"lr":1e-3}],
              weight_decay=0.05)
    sched=SequentialLR(opt,[LinearLR(opt,0.1,1.,5),
                            CosineAnnealingLR(opt,T_max=args.epochs-5)],
                       milestones=[5])

    best,bw=math.inf,None
    for _ in range(args.epochs):
        tr = run_epoch(net,tr_dl,opt)
        vl = run_epoch(net,vl_dl)
        sched.step()
        print(f"ep{_}  tr MSE(z) {tr:.4f} val MSE(z) {vl:.4f}")
        if vl<best: 
            best,bw=vl,deepcopy(net.state_dict())


    net.load_state_dict(bw)
    torch.save(bw,"latlon_multitask.pt")
  
    net.eval()
    preds=[]
    with torch.no_grad():
        for x,_,_ in vl_dl: 
            preds.append(net(x.to(DEVICE))[1].cpu())
    preds=torch.cat(preds)
    lat = preds[:,0]*ds_tr.lat_sd+ds_tr.lat_mu
    lon = preds[:,1]*ds_tr.lon_sd+ds_tr.lon_mu
    pd.DataFrame({"filename":ds_vl.df.filename,
                  "latitude":lat.round().int(),
                  "longitude":lon.round().int()}
                 ).to_csv("multitask_val_preds.csv",index=False)
    lat_true_np = ds_vl.df.latitude.values
    lon_true_np = ds_vl.df.longitude.values

    lat_true = torch.tensor(lat_true_np, dtype=lat.dtype, device=lat.device)
    lon_true = torch.tensor(lon_true_np, dtype=lon.dtype, device=lon.device)

    mse_lat = torch.mean((lat - lat_true) ** 2).item()
    mse_lon = torch.mean((lon - lon_true) ** 2).item()

    print(f"\nVAL MSE lat={mse_lat:.4f}  lon={mse_lon:.4f}  avg={(mse_lat + mse_lon)/2:.4f}")
    print("CSV saved → multitask_val_preds.csv")

if __name__=="__main__":
    main()
   
