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
from augmentations.augmentations import *
from torchvision.transforms import Lambda


class GALoader(data.Dataset):

    def __init__(
        self,
        root = '/content/Data/',
        split="train",
        is_train=True,
        is_transform=True,
        data_mul = 5,
        img_size=512,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.data_mul = data_mul
        self.is_train = is_train
        self.test_mode = test_mode
        self.n_classes = 3
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        if self.is_train:
            self.augmentations = EnhancedCompose([
#                 Merge(),
#                 RandomRotate(),
#                 ElasticTransform(),
#                 Split([0, 1], [1, 2], [2,3]),
                [NormalizeNumpyImage(), CreateSeg(), CreateMask()]
                # for non-pytorch usage, remove to_tensor conversion
#                 [Lambda(to_float_tensor), Lambda(to_long_tensor),Lambda(to_long_tensor)]
            ])
        else:
            self.augmentations = EnhancedCompose([
#                 Merge(),
# #                 augmentations.RandomRotate(),
# #                 augmentations.ElasticTransform(),
#                 Split([0, 1], [1, 2], [2,3]),
                [NormalizeNumpyImage(), CreateSeg(), CreateMask()]
                # for non-pytorch usage, remove to_tensor conversion
#                 [Lambda(to_float_tensor), Lambda(to_long_tensor),Lambda(to_long_tensor)]
            ])
        
        for split in ["train", "val"]:
                path = pjoin(self.root, split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list

        if not self.test_mode:
            self.setup_annotations()
            path = pjoin(self.root,  "aug_train.txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files["train"] = file_list
        

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
        
#         im = np.expand_dims(np.array(Image.open(im_path)), axis=-1)
#         all_seg = np.expand_dims(np.array(Image.open(all_seg_path)), axis=-1)
#         mask_seg = np.expand_dims(np.array(Image.open(mask_seg_path)), axis=-1)
        
#         im = torch.from_numpy(np.array(im)/255.)
#         im = torch.unsqueeze(im, 0).float()
#         all_seg = torch.from_numpy(np.array(all_seg)).long()
#         all_seg = torch.unsqueeze(all_seg, 0)
#         mask_seg = torch.from_numpy(np.array(mask_seg)).long()
#         mask_seg = torch.unsqueeze(mask_seg, 0)
        im = Image.open(im_path)
        all_seg = Image.open(all_seg_path)
        mask_seg = Image.open(mask_seg_path)
        
        im = torch.from_numpy(np.array(im)/255.)
        im = torch.unsqueeze(im, 0).float()
        all_seg = torch.from_numpy(np.array(all_seg)).long()
        all_seg = torch.unsqueeze(all_seg, 0)
        mask_seg = torch.from_numpy(np.array(mask_seg)).long()
        mask_seg = torch.unsqueeze(mask_seg, 0)
        
# #         if self.augmentations is not None:
#         im1, all_seg1, mask_seg1 = self.augmentations([im, all_seg, mask_seg])
#         im1 = torch.from_numpy(im1.transpose((2, 0, 1))).float()
#         all_seg1 = torch.from_numpy(all_seg1.transpose((2, 0, 1))).long()
#         mask_seg1 = torch.from_numpy(mask_seg1.transpose((2, 0, 1))).long()
#         im = torch.unsqueeze(im, 0)
#         all_seg = torch.unsqueeze(all_seg, 0)
#         mask_seg = torch.unsqueeze(mask_seg, 0)
#             im, all_seg, mask_seg = self.augmentations(im, all_seg)
            
        if self.is_transform:
            im, all_seg = self.transform(im, all_seg)#, mask_seg)
            
#         return im1, all_seg1, mask_seg1
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
        aug_transforms = EnhancedCompose([
            Merge(),
#             RandomRotate(),
            ElasticTransform(),
            Split([0, 1], [1, 2], [2,3])
            [, CreateSeg(), CreateMask()]
            # for non-pytorch usage, remove to_tensor conversion
    #                 [Lambda(to_float_tensor), Lambda(to_long_tensor),Lambda(to_long_tensor)]
        ])

        target_path = pjoin(self.root, "fixed_data")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            
            os.makedirs(pjoin(target_path, "ct"))
            os.makedirs(pjoin(target_path, "all_seg"))
            os.makedirs(pjoin(target_path, "mask_seg"))
            
            os.makedirs(pjoin(target_path, "ct", "train"))
            os.makedirs(pjoin(target_path, "ct", "val"))
            
            os.makedirs(pjoin(target_path, "all_seg", "train"))
            os.makedirs(pjoin(target_path, "all_seg", "val"))
            
            os.makedirs(pjoin(target_path, "mask_seg", "train"))
            os.makedirs(pjoin(target_path, "mask_seg", "val"))

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
                    
#                     target_path = pjoin(self.root, "fixed_data")
                    fixed_img.save(pjoin(target_path, "ct", split, img_name), "PNG")
                    fixed_all_seg.save(pjoin(target_path, "all_seg", split, seg_name), "PNG")
                    fixed_mask_seg.save(pjoin(target_path, "mask_seg", split, seg_name), "PNG")
                    if split == "train":
                        all_seg[all_seg == 1] = 127
                        all_seg[all_seg == 2] = 255

                        mask_seg[all_seg == 127] = 255
                        mask_seg[all_seg == 255] = 255
                        for i in range(self.data_mul):
                            aug_img, aug_all_seg , aug_mask_seg = aug_transforms([im, all_seg, mask_seg])
                            fixed_img = Image.fromarray(aug_img)
                            fixed_all_seg = Image.fromarray(aug_all_seg)
                            fixed_mask_seg = Image.fromarray(aug_mask_seg)
                            img_name = "ct" + str(i) + ii
                            seg_name = "seg" + str(i)+ ii
                            fixed_img.save(pjoin(target_path, "ct", split, img_name), "PNG")
                            fixed_all_seg.save(pjoin(target_path, "all_seg", split, seg_name), "PNG")
                            fixed_mask_seg.save(pjoin(target_path, "mask_seg", split, seg_name), "PNG")
                        
                    
                    
                    
                    
                    
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
