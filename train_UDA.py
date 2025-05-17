from utils.utils_preprocess_data import preprocess_data
from utils.utils_domain_dataset import DomainDataset, collate_fn
from utils.utils_uda_model import UDARetinaNet
from utils.utils_train_loop import train_da
from torch.utils.data import DataLoader
import os
import torch
import torchvision


# Global parameters
DATA_DIR = "/data"  # path to the input directory
DATASETS_NAMES = ['brats', 'bmshare']
CHECKPOINTS_DIR = "/checkpoints"




# create the data directories
preprocess_data(DATA_DIR, DATASETS_NAMES)

# folder structure
source_train_dir = os.path.join(DATA_DIR, "brats", "yolo_data/train")
source_val_dir = os.path.join(DATA_DIR, "brats", "yolo_data/val")
target_train_dir = os.path.join(DATA_DIR, "bmshare", "yolo_data/train")
target_val_dir = os.path.join(DATA_DIR, "bmshare", "yolo_data/val")

# create the dataloaders
batch_size = 4

# source
source_train_dataset = DomainDataset(source_train_dir, domain_label=0, img_size=256, img_extension=".png")
source_val_dataset   = DomainDataset(source_val_dir,   domain_label=0, img_size=256, img_extension=".png")
source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
source_val_loader   = DataLoader(source_val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
# target
target_train_dataset = DomainDataset(target_train_dir, domain_label=1, img_size=256, img_extension=".png")
target_val_dataset   = DomainDataset(target_val_dir,   domain_label=1, img_size=256, img_extension=".png")
target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
target_val_loader   = DataLoader(target_val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

# create the model
model = torchvision.models.detection.retinanet_resnet50_fpn(weights='DEFAULT')
for name, param in model.backbone.body.named_parameters():
    if name.startswith("conv1") or name.startswith("layer1"):
        param.requires_grad = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
model_da = UDARetinaNet(model).to(device)
optimizer = torch.optim.AdamW(model_da.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# reverse the source and target loaders if needed
reverse = False
if reverse:
    source_train_loader, source_val_loader, target_train_loader, target_val_loader = target_train_loader, target_val_loader, source_train_loader, source_val_loader

# train the model
train_da(source_train_loader, source_val_loader,
              target_train_loader, target_val_loader,
              model_da, optimizer, scheduler, max_epochs=100,
              root_dir=CHECKPOINTS_DIR,
              resume_training=False
         )