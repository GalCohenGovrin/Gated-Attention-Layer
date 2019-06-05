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


class GALoader(data.Dataset):

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
        
        for split in ["train", "val", "test"]:
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
        all_seg_path = pjoin(self.root, "fixed_data", "all_seg", self.split, "seg" + im_name)
        mask_seg_path = pjoin(self.root, "fixed_data", "mask_seg", self.split, "seg" + im_name)
        
        im = Image.open(im_path)
        all_seg = Image.open(all_seg_path)
        mask_seg = Image.open(mask_seg_path)
        
        im = torch.from_numpy(np.array(im)/255.)
        im = torch.unsqueeze(im, 0).float()
        all_seg = torch.from_numpy(np.array(all_seg)).long()
        all_seg = torch.unsqueeze(all_seg, 0)
        mask_seg = torch.from_numpy(np.array(mask_seg)).long()
        mask_seg = torch.unsqueeze(mask_seg, 0)
        
        if self.augmentations is not None:
            im, all_seg, mask_seg = self.augmentations(im, all_seg, mask_seg)
            
        if self.is_transform:
            im, all_seg, mask_seg = self.transform(im, all_seg, mask_seg)
            
        return im, all_seg, mask_seg

    def transform(self, img, all_seg, mask_seg):
        #if self.img_size == ("same", "same"):
         #   pass
        #else:
         #   img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        #    all_seg = all_seg.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        #all_seg = torch.from_numpy(np.array(all_seg)).long()
        
        return img, all_seg, mask_seg

    def setup_annotations(self):

        target_path = pjoin(self.root, "fixed_data")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            
            os.makedirs(pjoin(target_path, "ct"))
            os.makedirs(pjoin(target_path, "all_seg"))
            os.makedirs(pjoin(target_path, "mask_seg"))
            
            os.makedirs(pjoin(target_path, "ct", "train"))
            os.makedirs(pjoin(target_path, "ct", "val"))
            os.makedirs(pjoin(target_path, "ct", "test"))
            
            os.makedirs(pjoin(target_path, "all_seg", "train"))
            os.makedirs(pjoin(target_path, "all_seg", "val"))
            os.makedirs(pjoin(target_path, "all_seg", "test"))
            
            os.makedirs(pjoin(target_path, "mask_seg", "train"))
            os.makedirs(pjoin(target_path, "mask_seg", "val"))
            os.makedirs(pjoin(target_path, "mask_seg", "test"))

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
                    all_seg = np.array(Image.open(seg_path).convert('L'))
                    mask_seg = np.zeros_like(all_seg)
                    
                    all_seg[all_seg == 127] = 1
                    all_seg[all_seg == 255] = 2
                    
                    mask_seg[all_seg == 1] = 1
                    mask_seg[all_seg == 2] = 1
                    
                    fixed_img = Image.fromarray(img)
                    fixed_all_seg = Image.fromarray(all_seg)
                    fixed_mask_seg = Image.fromarray(mask_seg)
                    
                    fixed_img.save(pjoin(target_path, "ct", split, img_name), "PNG")
                    fixed_all_seg.save(pjoin(target_path, "all_seg", split, seg_name), "PNG")
                    fixed_mask_seg.save(pjoin(target_path, "mask_seg", split, seg_name), "PNG")
                    
            for ii in tqdm(self.files["test"]):
                img_name = "ct" + ii
                img_path = pjoin(self.root, "test", img_name)

                img = np.array(Image.open(img_path).convert('L'))
                fixed_img = Image.fromarray(img)

                fixed_img.save(pjoin(target_path, "ct", "test", img_name), "PNG")
                    
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
