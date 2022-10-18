import os
import random 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import matplotlib
import matplotlib.pyplot as plt 
import sys
from model import ResNet14
from Loader.superschedule_loader import SuperScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME

if __name__ == "__main__":
    f = open("trainlog.txt",'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = ResNet14(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
    net = net.to(device)
    #net.load_state_dict(torch.load('./resnet.pth'))
  
    criterion = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(net.parameters(), lr=1e-4)    
    
    SparseMatrix_Dataset = SparseMatrixDataset('./TrainingData/train.txt')
    train_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    SparseMatrix_Dataset_Valid = SparseMatrixDataset('./TrainingData/validation.txt')
    valid_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset_Valid, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for epoch in range(80) :
      # Train
      net.train()
      train_loss = 0
      train_loss_cnt = 0 
      for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(train_SparseMatrix) :
        torch.cuda.empty_cache()
        torch.save(net.state_dict(), "resnet.pth")
       
        SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0]) # Get rid of runtime<1000
        train_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
        shapes = shapes.to(device)
        
        SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
        for schedule_batchidx, (schedule, runtime) in enumerate(train_SuperSchedule) :
          if (schedule.shape[0] < 2) : break
          schedule, runtime = schedule.to(device), runtime.to(device)
          optimizer.zero_grad()
          query_feature = net.embed_sparse_matrix(SparseMatrix, shapes)
          query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
          predict = net.forward_after_query(query_feature, schedule)

          #HingeRankingLoss
          iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
          pred1, pred2 = predict[iu[0]], predict[iu[1]]
          true1, true2 = runtime[iu[0]], runtime[iu[1]]
          sign = (true1-true2).sign()
          loss = criterion(pred1, pred2, sign)
          train_loss += loss.item()
          train_loss_cnt += 1

          loss.backward()
          optimizer.step()
         
          if (sparse_batchidx % 100 == 0 and schedule_batchidx == 0) :
            print("Epoch: ", epoch, ", MTX: ", mtx_names[0], " " , shapes, "(", sparse_batchidx, "), Schedule : ", schedule_batchidx, ", Loss: ", loss.item())
            print("\tPredict : ", predict.detach()[:5,0])
            print("\tGT      : ", runtime.detach()[:5,0])
            print("\tQuery   : ", query_feature.detach()[0,:5])

          break
      
      #Validation
      net.eval()
      with torch.no_grad() :
        valid_loss = 0
        valid_loss_cnt = 0
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(valid_SparseMatrix) :
          torch.cuda.empty_cache()
          SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0]) # Get rid of runtime<1000
          valid_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
          shapes = shapes.to(device)
          
          SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
          for schedule_batchidx, (schedule, runtime) in enumerate(valid_SuperSchedule) :
            if (schedule.shape[0] < 6) : break
            schedule, runtime = schedule.to(device), runtime.to(device)
            query_feature = net.embed_sparse_matrix(SparseMatrix, shapes)
            query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
            predict = net.forward_after_query(query_feature, schedule)

            #HingeRankingLoss
            iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
            pred1, pred2 = predict[iu[0]], predict[iu[1]]
            true1, true2 = runtime[iu[0]], runtime[iu[1]]
            sign = (true1-true2).sign()
            loss = criterion(pred1, pred2, sign)
            valid_loss += loss.item()
            valid_loss_cnt += 1
           
            if (sparse_batchidx % 100 == 0 and schedule_batchidx == 0) :
              print("ValidEpoch: ", epoch, ", MTX: ", mtx_names[0], " " , shapes, "(", sparse_batchidx, "), Schedule : ", schedule_batchidx, ", Loss: ", loss.item())
              print("\tValidPredict : ", predict.detach()[:5,0])
              print("\tValidGT      : ", runtime.detach()[:5,0])
              print("\tValidQuery   : ", query_feature.detach()[0,:5])

            break
      
      print ("--- Epoch {} : Train {} Valid {} ---".format(epoch, train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
      f.write("--- Epoch {} : Train {} Valid {} ---\n".format(epoch, train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
      f.flush()

