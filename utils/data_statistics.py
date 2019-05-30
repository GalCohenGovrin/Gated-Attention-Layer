import torch
from torch.utils.data import DataLoader

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for data, _ in loader:

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment.item(), torch.sqrt(snd_moment - fst_moment ** 2).item()

    

#dataset = MyDataset()
#loader = DataLoader(
 #   dataset,
 #   batch_size=1,
 #   num_workers=1,
 #   shuffle=False
#)

#mean, std = online_mean_and_sd(loader)
