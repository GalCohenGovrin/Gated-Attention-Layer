import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms


class CTLoader(data.Dataset):

    def __init__(
        self,
        root = '/content/Data/',
        split="train",
        is_transform=True,
        img_size=512,
        augmentations=None,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.test_mode = test_mode
        self.n_classes = 3
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        for split in ["train", "val"]:
                path = pjoin(self.root, split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list

        if not self.test_mode:
            self.setup_annotations()

        self.tf = transforms.Compose(
            [
                #transforms.ToTensor(),
                transforms.Normalize([0.192], [0.263]),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])
    
    #TODO: fix path to correct data
    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "fixed_data", "ct", self.split, "ct" + im_name)
        lbl_path = pjoin(self.root, "fixed_data", "seg", self.split, "seg" + im_name)
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        im = torch.from_numpy(np.array(im)/255.)
        im = torch.unsqueeze(im, 0).float()
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl = torch.unsqueeze(lbl, 0)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        #if self.img_size == ("same", "same"):
         #   pass
        #else:
         #   img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        #    lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        #lbl = torch.from_numpy(np.array(lbl)).long()
        
        return img, lbl

    def setup_annotations(self):

        target_path = pjoin(self.root, "fixed_data")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            os.makedirs(pjoin(target_path, "ct"))
            os.makedirs(pjoin(target_path, "seg"))
            os.makedirs(pjoin(target_path, "ct", "train"))
            os.makedirs(pjoin(target_path, "ct", "val"))
            os.makedirs(pjoin(target_path, "seg", "train"))
            os.makedirs(pjoin(target_path, "seg", "val"))

        pre_encoded = glob.glob(pjoin(target_path, "*.png"))
        expected = np.unique(self.files["train"] + self.files["val"]).size

        if len(pre_encoded) != expected:
            print("Pre-encoding segmentation masks...")
            for split in ["train", "val"]:
                for ii in tqdm(self.files[split]):
                    img_name = "ct" + ii
                    seg_name = "seg" + ii
                    img_path = pjoin(self.root, "train_val", "ct", split, img_name)
                    seg_path = pjoin(self.root, "train_val", "seg", split, seg_name)
                    
                    img = np.array(Image.open(img_path).convert('L'))
                    seg = np.array(Image.open(seg_path).convert('L'))
                    seg[seg == 127] = 1
                    seg[seg == 255] = 2
                    
                    fixed_img = Image.fromarray(img)
                    fixed_seg = Image.fromarray(seg)
                    
                    fixed_img.save(pjoin(target_path, "ct", split, img_name), "PNG")
                    fixed_seg.save(pjoin(target_path, "seg", split, seg_name), "PNG")
                    
                    
def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)  


# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
# if __name__ == '__main__':
# # local_path = '/home/meetshah1995/datasets/VOCdevkit/VOC2012/'
# bs = 4
# augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
# dst = pascalVOCLoader(root=local_path, is_transform=True, augmentations=augs)
# trainloader = data.DataLoader(dst, batch_size=bs)
# for i, data in enumerate(trainloader):
# imgs, labels = data
# imgs = imgs.numpy()[:, ::-1, :, :]
# imgs = np.transpose(imgs, [0,2,3,1])
# f, axarr = plt.subplots(bs, 2)
# for j in range(bs):
# axarr[j][0].imshow(imgs[j])
# axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
# plt.show()
# a = raw_input()
# if a == 'ex':
# break
# else:
# plt.close()
