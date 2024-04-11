import os
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import utils
import kornia
from kornia.augmentation.container import AugmentationSequential


img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224,224), (0.8,1), p=0.3),
    kornia.augmentation.Resize((224, 224)),
    kornia.augmentation.RandomBrightness(brightness=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomContrast(contrast=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
    kornia.augmentation.RandomSaturation((0.8,1.2), p=0.2),
    kornia.augmentation.RandomHue((-0.1,0.1), p=0.2),
    kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
    kornia.augmentation.RandomGrayscale(p=0.2),
    data_keys=["input"],
)

class NSDDataset(Dataset):
    def __init__(self, root_dir, extensions=None, pool_num=8192, pool_type="max", length=None):
        self.root_dir = root_dir
        self.extensions = extensions if extensions else []
        self.pool_num = pool_num
        self.pool_type = pool_type
        self.samples = self._load_samples()
        self.samples_keys = sorted(self.samples.keys())
        self.length = length
        if length is not None:
            if length > len(self.samples_keys):
                pass # enlarge the dataset
            elif length > 0:
                self.samples_keys = self.samples_keys[:length]
            elif length < 0:
                self.samples_keys = self.samples_keys[length:]
            elif length == 0:
                raise ValueError("length must be a non-zero value!")
        else:
            self.length = len(self.samples_keys)

    def _load_samples(self):
        files = os.listdir(self.root_dir)
        samples = {}
        for file in files:
            file_path = os.path.join(self.root_dir, file)
            sample_id, ext = file.split(".",maxsplit=1)
            if ext in self.extensions:
                if sample_id in samples.keys():
                    samples[sample_id][ext] = file_path
                else:
                    samples[sample_id]={"subj": file_path}
                    samples[sample_id][ext] = file_path
            # print(samples)
        return samples
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image
    
    def _load_npy(self, npy_path):
        array = np.load(npy_path)
        array = torch.from_numpy(array)
        return array
    
    def vox_process(self, x):
        if self.pool_num is not None:
            x = pool_voxels(x, self.pool_num, self.pool_type)
        return x
    
    def subj_process(self, key):
        id = int(key.split("/")[-2].split("subj")[-1])
        return id
    
    def aug_process(self, brain3d):
        return brain3d

    def __len__(self):
        # return len(self.samples_keys)
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.samples_keys)
        sample_key = self.samples_keys[idx]
        sample = self.samples[sample_key]
        items = []
        for ext in self.extensions:
            if ext == "jpg":
                items.append(self._load_image(sample[ext]))
            elif ext == "nsdgeneral.npy":
                voxel = self._load_npy(sample[ext])
                items.append(self.vox_process(voxel))
            elif ext == "coco73k.npy":
                items.append(self._load_npy(sample[ext]))
            elif ext == "subj":
                items.append(self.subj_process(sample[ext]))
            elif ext == "wholebrain_3d.npy":
                brain3d = self._load_npy(sample[ext])
                items.append(self.aug_process(brain3d, ))

        return items

def pool_voxels(voxels, pool_num, pool_type):
    voxels = voxels.float()
    if pool_type == 'avg':
        voxels = nn.AdaptiveAvgPool1d(pool_num)(voxels)
    elif pool_type == 'max':
        voxels = nn.AdaptiveMaxPool1d(pool_num)(voxels)
    elif pool_type == "resize":
        voxels = voxels.unsqueeze(1) # Add a dimension to make it (B, 1, L)
        voxels = F.interpolate(voxels, size=pool_num, mode='linear', align_corners=False)
        voxels = voxels.squeeze(1)

    return voxels

def get_dataloader(
        root_dir,
        batch_size,
        num_workers=1,
        seed=42,
        is_shuffle=True,
        extensions=['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"],
        pool_type=None,
        pool_num=None,
        length=None,
    ):
    utils.seed_everything(seed)
    dataset = NSDDataset(root_dir=root_dir, extensions=extensions, pool_num=pool_num, pool_type=pool_type, length=length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=is_shuffle)

    return dataloader

def get_dls(subject, data_path, batch_size, val_batch_size, num_workers, pool_type, pool_num, length, seed):
    train_path = "{}/webdataset_avg_split/train/subj0{}".format(data_path, subject)
    val_path = "{}/webdataset_avg_split/val/subj0{}".format(data_path, subject)
    extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"]

    train_dl = get_dataloader(
        train_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=extensions,
        pool_type=pool_type,
        pool_num=pool_num,
        is_shuffle=True,
        length=length,
    )

    val_dl = get_dataloader(
        val_path,
        batch_size=val_batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=extensions,
        pool_type=pool_type,
        pool_num=pool_num,
        is_shuffle=False,
    )

    num_train=len(train_dl.dataset)
    num_val=len(val_dl.dataset)
    print(train_path,"\n",val_path)
    print("number of train data:", num_train)
    print("batch_size", batch_size)
    print("number of val data:", num_val)
    print("val_batch_size", val_batch_size)

    return train_dl, val_dl
