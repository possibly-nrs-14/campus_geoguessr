import os
import pandas as pd
from PIL import Image
from typing import Optional
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
import re
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
# all_models = models.list_models()
# print(all_models)
# exit(0)

class CampusDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["Region_ID"]) - 1
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_dataloaders(batch_size=32, num_workers=4):
    train_ds = CampusDataset(
        csv_file="labels_train.csv",
        root_dir="images_train",
        transform=get_transforms(train=True),
    )
    val_ds = CampusDataset(
        csv_file="labels_val.csv",
        root_dir="images_val",
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model(num_classes=15, freeze_backbone=False):
    model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
     # replace classifier head
    if hasattr(model, "classifier"):
        # torchvision ConvNeXt
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    elif hasattr(model, "head"):
        # timm ConvNeXt or other models with head
        # head can be a Linear or a Sequential ending with Linear
        head = model.head
        if isinstance(head, nn.Linear):
            in_feat = head.in_features
            model.head = nn.Linear(in_feat, num_classes)
        else:
            # assume Sequential
            *rest, last = list(head.children())
            in_feat = last.in_features
            new_head = nn.Sequential(*rest, nn.Linear(in_feat, num_classes))
            model.head = new_head
    elif hasattr(model, "fc"):
        # fallback for models with fc attribute
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    return model


def save_predictions_csv(model, val_loader, device, outfile="best_val_predictions.csv"):
   
    model.eval()

    fnames = val_loader.dataset.data["filename"].tolist()

    all_preds, all_truths = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            batch_preds = outputs.argmax(1).cpu().tolist()
            all_preds.extend([p + 1 for p in batch_preds])
            all_truths.extend((labels.cpu() + 1).tolist())

    # Build DataFrame
    df = pd.DataFrame({
        "filename": fnames,
        "true_region_id": all_truths,
        "pred_region_id": all_preds,
    })
    df.to_csv(outfile, index=False)
    print(f"Saved predictions for best model to {outfile}") 


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f'Epoch {epoch + 1}'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def fit(num_epochs=100, batch_size=16, save_path="best_region_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(batch_size=batch_size)
    model = build_model()
    model.to(device)
    if os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded checkpoint '{save_path}', continuing training…")
    lr_backbone = 1.5e-4
    lr_head     = 1.5e-3
    weight_decay = 0.05

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ], weight_decay=weight_decay)


    warmup_epochs = 5
    scheduler = SequentialLR(optimizer,
                             schedulers=[
                                 LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                                 CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
                             ],
                             milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(0, num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            save_predictions_csv(model, val_loader, device)
            print(f"Saved best model with acc {best_val_acc:.4f}")

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

fit()

