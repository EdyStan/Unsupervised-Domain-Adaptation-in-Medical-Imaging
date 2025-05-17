from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np


class DomainDataset(Dataset):
    def __init__(self, root_dir, domain_label=0, img_size=256, img_extension=".png", transforms_fn=None):
        self.images_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(img_extension)]
        self.domain_label = domain_label
        self.img_extension = img_extension
        self.img_size = img_size

        self.transforms = transforms_fn or transforms.Compose([
            # Resize the image to a fixed size
            transforms.Resize((256, 256)),
            # Randomly flip the image horizontally for augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            # Randomly rotate the image within a small angle range
            transforms.RandomRotation(degrees=15),
            # Apply random changes in brightness, contrast, saturation, and hue
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            # Convert the image to a tensor
            transforms.ToTensor(),
            # Normalize using ImageNet statistics
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        label_filename = img_filename.replace(self.img_extension, ".txt")

        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size  # original size before transform
        image = self.transforms(image)

        # Load YOLO label [class, x_center, y_center, width, height] (normalized)
        label_path = os.path.join(self.labels_dir, label_filename)
        if os.path.exists(label_path) and os.stat(label_path).st_size > 0:
            label = torch.tensor(np.loadtxt(label_path).reshape(-1, 5), dtype=torch.float32)

            # Convert to [xmin, ymin, xmax, ymax]
            boxes = []
            classes = []
            for obj in label:
                cls, x_center, y_center, bw, bh = obj.tolist()
                xmin = (x_center - bw / 2) * w
                ymin = (y_center - bh / 2) * h
                xmax = (x_center + bw / 2) * w
                ymax = (y_center + bh / 2) * h
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(int(cls))

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(classes, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "domain_label": torch.tensor(self.domain_label, dtype=torch.float32),
            "image_id": torch.tensor([idx])  # optional, good for eval/debug
        }

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))