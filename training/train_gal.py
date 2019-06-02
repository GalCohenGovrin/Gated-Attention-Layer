import torch
import torch.nn.functional as F
from loss.loss import cross_entropy2d

def trainUnet(model, tLoader, vLoader, optimizer, scheduler, scores, weights, nm_epochs):
  epoch = 0
  while epoch <=nm_epochs:
    scores.reset()
    i = 0
    total_loss = 0
    for (images, labels) in tLoader:
            
            i += 1
            
            model.train()
            images = images.float().cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)

            loss = cross_entropy2d(input=outputs, target=labels, weight=weights)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            labels = labels.squeeze()
            scores.update(outputs, labels)
            
            
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
    for (images, labels) in vLoader:
      i += 1

      model.eval()
      images = images.float().cuda()
      labels = labels.cuda()

      outputs = model(images)

      loss = cross_entropy2d(input=outputs, target=labels, weight=balance_weight)

      total_loss += loss.item()
      labels = labels.squeeze()
      scores.update(outputs, labels)
      
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
