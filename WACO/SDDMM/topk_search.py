import numpy as np
import torch
import torch.nn as nn
from model import ResNet14
from Loader.superschedule_loader import SuperScheduleDataset,TrainingScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME
import hnswlib
import time

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  schedules = TrainingScheduleDataset("./TrainingData/total.txt")
  schedule_loader = torch.utils.data.DataLoader(schedules, batch_size=128, shuffle=False, num_workers=0)
  
  net = ResNet14(in_channels=1, out_channels=1, D=2) # D : 2D Tensor
  net = net.to(device)
  net.load_state_dict(torch.load('resnet.pth'))
  net.eval()
  
  names = []
  for batch_idx, (data, string) in enumerate(schedule_loader) :
    data = data.to(device)
    names.extend(string)

  dim = 128
  num_elements = schedules.__len__()
 
  p = hnswlib.Index(space='ip', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
  p.load_index("hnsw_schedule.bin", max_elements = num_elements)
  p.set_ef(200) # ef should always be > k
  
  SparseMatrix_Dataset = SparseMatrixDataset("./TrainingData/test.txt")
  train_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
  for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(train_SparseMatrix) :
    torch.cuda.empty_cache()
    SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
    shapes = shapes.to(device)

    query = net.embed_sparse_matrix(SparseMatrix, shapes)
    labels, distances = p.knn_query(query.cpu().detach().numpy()[0], k=20)
    
    with open("./topk/"+mtx_names[0]+".txt", 'w') as f :
      f.write('\n'.join(list(np.array(names)[labels[0]])))
