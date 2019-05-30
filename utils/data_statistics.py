import torch
from torch.utils.data import DataLoader

def online_statistics(loader):
    """Compute in an online fashion the following:
        - images mean and sd 
        - groundtruth labels statistics
    

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)
    
    bg_cnt = 0
    liver_cnt = 0
    lesion_cnt = 0
    for data, lbl in loader:

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        current_lbl, cnt_lbl = torch.unique(lbl, return_counts=True)
        for i in range(len(cnt_lbl)):
            if current_lbl[i].item() == 0:
                bg_cnt += cnt_lbl[i].item()
            elif current_lbl[i].item() == 1:
                liver_cnt += cnt_lbl[i].item()
            else:
                lesion_cnt += cnt_lbl[i].item()

        cnt += nb_pixels
        
    bg_cnt = 1./(bg_cnt/cnt)
    liver_cnt = 1./(liver_cnt/cnt)
    lesion_cnt = 1./(lesion_cnt/cnt)

    return fst_moment.item(), torch.sqrt(snd_moment - fst_moment ** 2).item(), bg_cnt, liver_cnt, lesion_cnt

    

#dataset = MyDataset()
#loader = DataLoader(
 #   dataset,
 #   batch_size=1,
 #   num_workers=1,
 #   shuffle=False
#)

#mean, std = online_mean_and_sd(loader)
