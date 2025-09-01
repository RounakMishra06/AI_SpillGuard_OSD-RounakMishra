# in train.py (setup part)
from torch.utils.data import DataLoader
from src.dataset import OilSpillDataset
from src.transforms import get_preproc_transforms, get_train_aug_transforms
import torch, torch.optim as optim
from src.unet import UNet
from src.losses import bce_dice_loss
from src.metrics import threshold, iou, dice

train_ds = OilSpillDataset("data/train/images", "data/train/masks", transform=get_train_aug_transforms())
val_ds   = OilSpillDataset("data/val/images",   "data/val/masks",   transform=get_preproc_transforms())

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_ch=3, out_ch=1).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 11):
    model.train()
    for (batch, _) in train_dl:
        img = batch["image"].to(device, dtype=torch.float32)
        msk = batch["mask"].unsqueeze(1).to(device, dtype=torch.float32)  # [B,1,H,W]
        logits = model(img)
        loss = bce_dice_loss(logits, msk)
        opt.zero_grad(); loss.backward(); opt.step()

    # quick val
    model.eval(); ious = []; dices = []
    with torch.no_grad():
        for (batch, _) in val_dl:
            img = batch["image"].to(device, dtype=torch.float32)
            msk = batch["mask"].unsqueeze(1).to(device, dtype=torch.float32)
            logits = model(img)
            pred = threshold(logits)
            ious.append(iou(pred, msk))
            dices.append(dice(pred, msk))
    print(f"Epoch {epoch}: IoU={sum(ious)/len(ious):.3f}  Dice={sum(dices)/len(dices):.3f}")