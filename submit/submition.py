import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils import data
from PIL import Image
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt

def save_predictions(preds, name, root = '/content/Data/'):
    np_preds = preds.cpu().detach().numpy()
    np_preds = np.squeeze(np_preds, axis=0)
    
    pred_mask = np.zeros_like(np_preds, dtype=int)
    pred_mask[np_preds.max(axis=0,keepdims=1) == np_preds] = 1
    
    pred_bg = pred_mask[0, :, :]
    pred_liver = pred_mask[1, :, :]
    pred_lesion = pred_mask[2, :, :]
    
    final_mask = np.zeros_like(pred_bg, dtype=int)
    
    final_mask[pred_liver == 1] = 127
    final_mask[pred_lesion == 1] = 255
    
    img_name = "seg" + name[0]
    path = pjoin(root, "fixed_data", "seg", "test",  img_name)
    fixed_all_seg = Image.fromarray(final_mask)
    fixed_all_seg.save(pjoin(path), "PNG")
    
  

