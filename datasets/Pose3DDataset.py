import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class Pose3DDataset(Dataset):
    """
    Dataset that loads one random view image per sample and corresponding 3D joints.

    Expected directory structure:
    data_root/
        frames/
            frame_0001/
                cam1.jpg
                cam2.jpg
                ...
            frame_0002/
                ...
        joints3d/
            frame_0001.npy
            frame_0002.npy
            ...
    """

    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.frames_dir = os.path.join(data_root, "frames")
        self.joints_dir = os.path.join(data_root, "joints3d")
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.frame_names = sorted(os.listdir(self.frames_dir))

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):
        frame_name = self.frame_names[idx]
        frame_folder = os.path.join(self.frames_dir, frame_name)

        # List all camera images for this frame
        cam_images = [f for f in os.listdir(frame_folder) if f.endswith(('.jpg', '.png'))]
        assert cam_images, f"No camera images found in {frame_folder}"

        # Choose a random camera image
        img_name = random.choice(cam_images)
        img_path = os.path.join(frame_folder, img_name)

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load corresponding 3D joints
        joints_path = os.path.join(self.joints_dir, frame_name + ".npy")
        joints = np.load(joints_path).astype(np.float32)  # Shape: (21, 3)

        joints = torch.from_numpy(joints)

        return image, joints
