import torch
import torch.nn as nn
import numpy as np
import itertools
from itertools import permutations, product

class SuperScheduleDataset(torch.utils.data.Dataset):
    def __init__(self, name):
      with open("./TrainingData/CollectedData/"+name+".txt") as f:
        lines = f.read().splitlines()
        lines = [line.split() for line in lines]

      split_ = [1<<p for p in range(17)]
      index_ = ['i1', 'i0', 'j1', 'j0', 'k1', 'k0']
      format_ = [0, 1] #(C,U)
      parnum_ = [48]
      parchunk_ = [1<<p for p in range(6)] #[1,256]

      schedules = []
      runtimes = []

      for idx, line in enumerate(lines) :
        i0s = split_.index(int(line[0]))
        j0s = split_.index(int(line[1]))
        k0s = split_.index(int(line[2]))

        order = line[3:9]
        perm = np.zeros((len(index_), len(index_)))
        perm[index_.index(order[0]),0] = 1 #i1
        perm[index_.index(order[1]),1] = 1 #i0
        perm[index_.index(order[2]),2] = 1 #k1
        perm[index_.index(order[3]),3] = 1 #k0
        perm[index_.index(order[4]),4] = 1 #j1
        perm[index_.index(order[5]),5] = 1 #j0
        perm = perm.flatten()

        i1f = format_.index(int(line[9]))
        i0f = format_.index(int(line[10]))
        j1f = format_.index(int(line[11]))
        j0f = format_.index(int(line[12]))

        p1 = index_.index(line[13])
        p2 = parnum_.index(int(line[14]))
        p3 = parchunk_.index(int(line[15]))

        concat = np.array([i0s,j0s,k0s,
                           perm,
                           i1f,i0f,j1f,j0f,
                           p1,p2,p3], dtype=object)
        concat = np.hstack(concat)
        runtime = float(line[-1])
          
        if (runtime < 1000) :
          schedules.append(concat)
          runtimes.append([runtime])

      schedules = np.stack(schedules, axis=0)
      runtimes = np.stack(runtimes, axis=0)
      self.schedules = schedules.astype(np.float32)
      self.runtimes = runtimes.astype(np.float32)
      
      # Normalize
      self.runtimes = self.runtimes / 1000.0

      # To TorchTensor
      self.schedules = torch.from_numpy(self.schedules)
      self.runtimes = torch.from_numpy(self.runtimes)
        
     
    def __len__(self):
      return len(self.schedules)

    def __getitem__(self, idx):
      return self.schedules[idx], self.runtimes[idx] 

class TrainingScheduleDataset(torch.utils.data.Dataset):
    def __init__(self, filename, extend=False):
      split_ = [1<<p for p in range(17)]
      index_ = ['i1', 'i0', 'j1', 'j0', 'k1', 'k0']
      format_ = [0, 1] #(C,U)
      parnum_ = [48]
      parchunk_ = [1<<p for p in range(6)]

      schedules = []
      schedules_str = []

      with open(filename) as f :
        names = f.read().splitlines()
        uniqstr = set()
      
      for name in names : 
        with open("./TrainingData/CollectedData/"+name+".txt") as f:
          lines = f.read().splitlines()
          lines = [line.split() for line in lines]
        
        for idx, line in enumerate(lines) :
          if (" ".join(line[:-2]) in uniqstr) : continue
          uniqstr.add(" ".join(line[:-2]))          

          i0s = split_.index(int(line[0]))
          j0s = split_.index(int(line[1]))
          k0s = split_.index(int(line[2]))

          order = line[3:9]
          perm = np.zeros((len(index_), len(index_)))
          perm[index_.index(order[0]),0] = 1
          perm[index_.index(order[1]),1] = 1
          perm[index_.index(order[2]),2] = 1
          perm[index_.index(order[3]),3] = 1
          perm[index_.index(order[4]),4] = 1
          perm[index_.index(order[5]),5] = 1
          perm = perm.flatten()

          i1f = format_.index(int(line[9]))
          i0f = format_.index(int(line[10]))
          j1f = format_.index(int(line[11]))
          j0f = format_.index(int(line[12]))

          p1 = index_.index(line[13])
          p2 = parnum_.index(int(line[14]))
          p3 = parchunk_.index(int(line[15]))

          concat = np.array([i0s,j0s,k0s,
                             perm,
                             i1f,i0f,j1f,j0f,
                             p1,p2,p3], dtype=object)
          concat = np.hstack(concat)
          schedules.append(concat)
          schedules_str.append(" ".join(line[:-2]))

      schedules = np.stack(schedules, axis=0)
      self.schedules = schedules.astype(np.float32)
      self.schedules_str = schedules_str
       
    def __len__(self):
      return len(self.schedules)

    def __getitem__(self, idx):
      return self.schedules[idx], self.schedules_str[idx] 


