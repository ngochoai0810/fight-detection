"""Dataset cho M1 (tung frame) va M3 (chuoi frame). Nhan: Fight = 1, NonFight = 0."""
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from prepare_data import read_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def list_clips(root, split):
    clips = []
    for label_name, label in (("Fight", 1), ("NonFight", 0)):
        for clip_dir in sorted((Path(root) / split / label_name).glob("*")):
            frames = sorted(clip_dir.glob("*.jpg"))
            if frames:
                clips.append({"video": clip_dir.name, "label": label, "frames": frames})
    return clips


def sample_augmentation(image_size):
    """Rut tham so mot lan de co the ap dung giong nhau cho moi frame trong clip."""
    crop = random.randint(int(image_size * 0.85), image_size)
    return {
        "flip": random.random() < 0.5,
        "brightness": random.uniform(0.8, 1.2),
        "contrast": random.uniform(0.8, 1.2),
        "saturation": random.uniform(0.8, 1.2),
        "crop": crop,
        "top": random.random(),
        "left": random.random(),
    }


def load_tensor(path, image_size, params=None):
    image = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    if params:
        crop = params["crop"]
        top = int(params["top"] * (image_size - crop))
        left = int(params["left"] * (image_size - crop))
        tensor = TF.resized_crop(tensor, top, left, crop, crop, [image_size, image_size], antialias=True)
        if params["flip"]:
            tensor = TF.hflip(tensor)
        tensor = TF.adjust_brightness(tensor, params["brightness"])
        tensor = TF.adjust_contrast(tensor, params["contrast"])
        tensor = TF.adjust_saturation(tensor, params["saturation"])
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class FrameDataset(Dataset):
    def __init__(self, root, split, image_size=224, augment=False):
        self.items = [
            (frame, clip["label"], clip["video"]) for clip in list_clips(root, split) for frame in clip["frames"]
        ]
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label, video = self.items[index]
        params = sample_augmentation(self.image_size) if self.augment else None
        return load_tensor(path, self.image_size, params), label, video


class ClipDataset(Dataset):
    def __init__(self, root, split, image_size=224, augment=False):
        self.clips = list_clips(root, split)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        params = sample_augmentation(self.image_size) if self.augment else None
        frames = [load_tensor(path, self.image_size, params) for path in clip["frames"]]
        return torch.stack(frames), clip["label"], clip["video"]
