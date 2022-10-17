import numpy as np
import torch
import torch.nn as nn
from model import ResNet14
from Loader.superschedule_loader import TrainingScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME
import hnswlib
import time
import os
import sys

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  schedules = TrainingScheduleDataset("./TrainingData/total.txt")
  schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)
  
  net = ResNet14(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
  net = net.to(device)
  net.load_state_dict(torch.load('resnet.pth'))
  net.eval()
 
  waco_prefix = os.getenv("WACO_HOME")
  if waco_prefix is None : 
    print("Err : environment variable WACO_HOME is not defined")
    quit() 
  np.savetxt(waco_prefix+'/hnswlib/WACO_COSTMODEL/weight0.txt', net.final[0].weight.detach().cpu().numpy().flatten(), fmt='%.6f' )
  np.savetxt(waco_prefix+'/hnswlib/WACO_COSTMODEL/weight1.txt', net.final[2].weight.detach().cpu().numpy().flatten(), fmt='%.6f')
  np.savetxt(waco_prefix+'/hnswlib/WACO_COSTMODEL/weight2.txt', net.final[4].weight.detach().cpu().numpy().flatten(), fmt='%.6f' )
  np.savetxt(waco_prefix+'/hnswlib/WACO_COSTMODEL/bias0.txt', net.final[0].bias.detach().cpu().numpy().flatten(), fmt='%.6f' )
  np.savetxt(waco_prefix+'/hnswlib/WACO_COSTMODEL/bias1.txt', net.final[2].bias.detach().cpu().numpy().flatten(), fmt='%.6f' )
  np.savetxt(waco_prefix+'/hnswlib/WACO_COSTMODEL/bias2.txt', net.final[4].bias.detach().cpu().numpy().flatten(), fmt='%.6f' )

  start = time.time()  
  names = []
  embeddings = [] 
  for batch_idx, (data, string) in enumerate(schedule_loader) :
    data = data.to(device)
    embedding = net.embed_super_schedule(data)
    embeddings.extend(embedding.detach().cpu().tolist())
    names.extend(string)
  embeddings = np.array(embeddings)
  print("Calculate Embedding : ", time.time()-start)

  dim = embeddings.shape[1] 
  num_elements = embeddings.shape[0]
  p = hnswlib.Index(space = 'l2', dim = dim) 
  p.init_index(max_elements = num_elements, ef_construction = 200, M = 32)
  start = time.time()
  p.add_items(embeddings, np.arange(num_elements))
  print("Gen Index : ", time.time()-start)
  p.save_index("hnsw_schedule.bin")
  

