import random 
import itertools
import time
import sys
import random
import os
import math
import numpy as np

result=""
currinput = ""
num_row = 0
num_col = 0
num_nonzero = 0
def filter(l) :
  i1 = l.index('i1')
  k1 = l.index('k1')
  i0 = l.index('i0')
  k0 = l.index('k0')
  return (i1<i0) and (k1<k0) 

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
    while (len(cfgs) < 100) :
      cfg['isplit'] = random.choice([1<<p for p in range(int(math.log(num_row,2)))]) 
      cfg['ksplit'] = random.choice([1<<p for p in range(int(math.log(num_col,2)))]) 
      cfg['rankorder'] = random.choice([p for p in list(itertools.permutations(['i1','i0','k1','k0'])) if filter(p)]) 
      cfg['i1f'] = random.choice([0,1])
      cfg['i0f'] = random.choice([0,1])
      cfg['k1f'] = random.choice([0,1])
      cfg['k0f'] = random.choice([0,1])
      cfg['paridx'] = random.choice(['i1'])
      cfg['parnum'] = random.choice([48])
      cfg['parchunk'] = random.choice([1<<p for p in range(9)])
      isplit, ksplit = cfg['isplit'], cfg['ksplit']
      rankorder= " ".join(cfg['rankorder'])
      i1f, i0f, k1f, k0f = cfg['i1f'], cfg['i0f'], cfg['k1f'], cfg['k0f']
      paridx, parnum, parchunk = cfg['paridx'], cfg['parnum'], cfg['parchunk']
      cfgs.add("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(isplit, ksplit, rankorder, i1f, i0f, k1f, k0f, paridx, parnum, parchunk))
    
    f = open("./config/{}.txt".format(mtx), 'w')
    for sched in cfgs : f.write(sched)
    f.close()
