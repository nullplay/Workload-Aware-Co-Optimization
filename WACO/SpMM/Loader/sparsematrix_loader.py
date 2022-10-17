import torch
import numpy as np
import MinkowskiEngine as ME
import os

def from_csr(filename) :
  waco_prefix = os.getenv("WACO_HOME")
  if waco_prefix is None : 
    print("Err : environment variable WACO_HOME is not defined")
    return 

  csr = np.fromfile(waco_prefix+"/dataset/"+filename+".csr", dtype='<i4')
  num_row,num_col,nnz = csr[0],csr[1],csr[2]
  coo = np.zeros((nnz,2),dtype=int)
  coo[:,1] = csr[3+num_row+1:]
  bins = np.array(csr[4:num_row+4]) - np.array(csr[3:num_row+3])
  coo[:,0] = np.repeat(range(num_row), bins)
  return num_row, num_col, nnz, coo

def collate_fn(list_data):
    coords_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
    )

    mtxnames_batch = [d["mtxname"] for d in list_data]
    shapes_batch = torch.stack([d["shape"] for d in list_data]) 

    return mtxnames_batch, coords_batch, features_batch, shapes_batch

class SparseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
      waco_prefix = os.getenv("WACO_HOME")
      if waco_prefix == None : 
        print("Err : environment variable WACO_HOME is not defined")
        quit() 
      with open(filename) as f :
        self.names = f.read().splitlines() 
      # Preparing Data
      self.standardize = {}
      self.normalize = {}
      with open("./TrainingData/train.txt") as f :
        total_rows, total_cols, total_nnzs = [], [], []
        for filename in f.read().splitlines() :
          csr = np.fromfile(waco_prefix+"/dataset/"+filename+".csr", count=3, dtype='<i4')
          total_rows.append(csr[0])
          total_cols.append(csr[1])
          total_nnzs.append(csr[2])
        self.standardize["mean_rows"] = np.mean(total_rows)
        self.standardize["mean_cols"] = np.mean(total_cols)
        self.standardize["mean_nnzs"] = np.mean(total_nnzs)
        self.standardize["std_rows"] = np.std(total_rows)
        self.standardize["std_cols"] = np.std(total_cols)
        self.standardize["std_nnzs"] = np.std(total_nnzs)
    
    def __len__(self):
      return len(self.names)

    def __getitem__(self, idx):
      filename = self.names[idx]
      num_row, num_col, nnz, coo = from_csr(filename)
      
      # standardize
      num_row = (num_row - self.standardize["mean_rows"])/self.standardize["std_rows"]
      num_col = (num_col - self.standardize["mean_cols"])/self.standardize["std_cols"]
      nnz     = (nnz - self.standardize["mean_nnzs"])/self.standardize["std_nnzs"]
      
      # To ME Sparse Tensor
      coordinates = torch.from_numpy(coo).to(torch.int32)
      features = torch.ones((len(coo),1)).to(torch.float32)
      label = torch.tensor([[0]]).to(torch.float32)
      shape = torch.tensor([num_row, num_col, nnz]).to(torch.float32)

      return {
        "mtxname" : filename, 
        "coordinates" : coordinates, 
        "features" : features, 
        "label" : label, 
        "shape" : shape 
      }



