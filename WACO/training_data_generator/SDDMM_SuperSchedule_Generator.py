import random 
import os
import itertools
import time
import sys
import math
import numpy as np

result=""
currinput = ""
num_row = 0
num_col = 0
num_nonzero = 0
def filter(l,k) :
  i1 = l.index('i1')
  k1 = l.index('k1')
  j1 = l.index('j1')
  i0 = l.index('i0')
  k0 = l.index('k0')
  j0 = l.index('j0')
  if k<8 : 
    lastk = k1==4 and k0==5
  else :
    lastk = k0==5
  return (i1<i0) and (k1<k0) and (j1<j0) and lastk 

def first(l) :
  for idx in l :
    if (idx == 'i1' or idx == 'j1') :
      return idx

if __name__ == '__main__':
  waco_prefix = os.getenv("WACO_HOME")
  if waco_prefix == None : 
    print("Err : environment variable WACO_HOME is not defined")
    quit() 

  with open(sys.argv[1]) as f :
    lines = f.read().splitlines()

  for idx, mtx in enumerate(lines) :
    csr = np.fromfile(waco_prefix + "/dataset/{0}.csr".format(mtx), count=3, dtype='<i4')
    num_row, num_col, num_nonzero = csr[0], csr[1], csr[2]

    cfgs = set()
    cfg = {}
    while (len(cfgs) < 32) :
      cfg['isplit'] = random.choice([1<<p for p in range(int(math.log(num_row,2)))]) 
      cfg['jsplit'] = random.choice([1<<p for p in range(int(math.log(num_col,2)))]) 
      cfg['ksplit'] = random.choice([1<<p for p in range(int(math.log(256,2)))]) 
      cfg['rankorder'] = random.choice([p for p in list(itertools.permutations(['i1','i0','k1','k0','j1','j0'])) if filter(p,cfg['ksplit'])]) 
      cfg['i1f'] = random.choice([0,1])
      cfg['i0f'] = random.choice([0,1])
      cfg['j1f'] = random.choice([0,1])
      cfg['j0f'] = random.choice([0,1])
      if (cfg['i1f'] == 1 and cfg['i0f'] == 1 and cfg['j1f'] == 1 and cfg['j0f'] == 1) : continue
      cfg['paridx'] = first(cfg['rankorder']) #random.choice(['i1','j1'])
      cfg['parnum'] = random.choice([48])
      cfg['parchunk'] = random.choice([1<<p for p in range(6)])
      isplit, ksplit, jsplit = cfg['isplit'], cfg['ksplit'], cfg['jsplit']
      rankorder= " ".join(cfg['rankorder'])
      i1f, i0f, j1f, j0f = cfg['i1f'], cfg['i0f'], cfg['j1f'], cfg['j0f']
      paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
      cfgs.add("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n".format(isplit, jsplit, ksplit, rankorder, i1f, i0f, j1f, j0f, paridx, parnum, parchunk))


    while (len(cfgs) < 100) :
      cfg['isplit'] = random.choice([1,2,4,8,16,32])  #random.choice([1<<p for p in range(int(math.log(num_row,2)))]) 
      cfg['jsplit'] = random.choice([1,2,4,8,16,32])  #random.choice([1<<p for p in range(int(math.log(num_col,2)))]) 
      cfg['ksplit'] = random.choice([1<<p for p in range(int(math.log(256,2)))]) 
      cfg['rankorder'] = random.choice([['i1','j1','k1','i0','j0','k0'], ['j1','i1','k1','j0','i0','k0'], ['i1','j1','k1','j0','i0','k0'], ['j1','i1','k1','i0','j0','k0']])
      cfg['i1f'] = random.choice([0,1])
      cfg['i0f'] = 1 #random.choice([0,1])
      cfg['j1f'] = random.choice([0,1])
      cfg['j0f'] = 1 #random.choice([0,1])
      if (cfg['i1f'] == 1 and cfg['i0f'] == 1 and cfg['j1f'] == 1 and cfg['j0f'] == 1) : continue
      cfg['paridx'] = first(cfg['rankorder']) #random.choice(['i1', 'j1'])
      cfg['parnum'] = random.choice([48])
      cfg['parchunk'] = random.choice([1<<p for p in range(6)])
      isplit, ksplit, jsplit = cfg['isplit'], cfg['ksplit'], cfg['jsplit']
      rankorder= " ".join(cfg['rankorder'])
      i1f, i0f, j1f, j0f = cfg['i1f'], cfg['i0f'], cfg['j1f'], cfg['j0f']
      paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
      cfgs.add("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n".format(isplit, jsplit, ksplit, rankorder, i1f, i0f, j1f, j0f, paridx, parnum, parchunk))
    
    f = open("./config/{}.txt".format(mtx), 'w')
    for sched in cfgs : f.write(sched)
    f.close()
