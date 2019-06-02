import torch
import torch.nn.functional as F
import torch.nn as nn
from loss.loss import cross_entropy2d

def trainGUnet(model, tLoader, vLoader, optimizer, scheduler, scores, weights, nm_epochs):
  epoch = 0
  mask_loss = nn.BCEWithLogitsLoss()
  while epoch <= nm_epochs:
    scores.reset()
    i = 0
    total_loss = 0
    for (images, all_seg, mask_seg) in tLoader:
            
            i += 1
            
            model.train()
            images = images.float().cuda()
            all_seg = all_seg.cuda()
            mask_seg = mask_seg.cuda()
            optimizer.zero_grad()
            seg_out, mask_out = model(images)

            ce_loss = cross_entropy2d(input=seg_out, target=all_seg, weight=weights)
            bce_loss = mask_loss(mask_out, mask_seg)
            
            loss = ce_loss + bce_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_seg = all_seg.squeeze()
            scores.update(seg_out, all_seg)
            
            
  metrics_results = scores.get_scores()
  print('dice_bg:\t',metrics_results[0])
  print('precision_bg:\t',metrics_results[1])
  print('recall_bg:\t',metrics_results[2],'\n')
  print('dice_liver:\t',metrics_results[3])
  print('precision_liver:\t',metrics_results[4])
  print('recall_liver:\t',metrics_results[5],'\n')
  print('dice_lesion:\t',metrics_results[6])
  print('precision_lesion:\t',metrics_results[7])
  print('recall_lesion:\t',metrics_results[8],'\n')
  print('avg loss:\t',total_loss/i)
  print('epoch:\t', epoch, '\n')
  
  scores.reset()
  i = 0
  total_loss = 0
  with torch.no_grad():
    for (images, all_seg, mask_seg) in vLoader:
      i += 1

      model.eval()
      images = images.float().cuda()
      all_seg = all_seg.cuda()
      mask_seg = mask_seg.cuda()

      seg_out, mask_out = model(images)

      loss = cross_entropy2d(input=seg_out, target=all_seg, weight=balance_weight)

      total_loss += loss.item()
      all_seg = all_seg.squeeze()
      scores.update(seg_out, all_seg)
      
  metrics_results = scores.get_scores()
  print('----------VAL-------------')
  print('dice_bg:\t',metrics_results[0])
  print('precision_bg:\t',metrics_results[1])
  print('recall_bg:\t',metrics_results[2],'\n')
  print('dice_liver:\t',metrics_results[3])
  print('precision_liver:\t',metrics_results[4])
  print('recall_liver:\t',metrics_results[5],'\n')
  print('dice_lesion:\t',metrics_results[6])
  print('precision_lesion:\t',metrics_results[7])
  print('recall_lesion:\t',metrics_results[8],'\n')
  print('avg loss:\t',total_loss/i)
  print('epoch:\t', epoch, '\n')
  
  scheduler.step(metrics_results[7])
  
  epoch += 1
