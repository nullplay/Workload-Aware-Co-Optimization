# WACO
Workload-Aware Co-Optimization for a sparse tensor program.

This repository includes an artifact for ["WACO: Learning workload-aware co-optimization of the format and schedule of a sparse tensor program"](https://dl.acm.org/doi/10.1145/3575693.3575742)

## Requirement
You can compile a generated code from TACO with `gcc` with OpenMP but we ***highly recommend*** to use Intel C++ Compiler Classic (`icc`, `icpc`) to compile a generated kernel from TACO for better performance.

You can download from https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.fyw7ne . It is important to install **Intel C++ Compiler Classic (`icc` and `icpc`)**, not an OneAPI compiler.

## Installation
#### 0. clone the repo and set `WACO_HOME` as working directory.
```
git clone https://github.com/nullplay/Workload-Aware-Co-Optimization.git
cd Workload-Aware-Co-Optimization
export WACO_HOME=`pwd`
```  
#### 1. If you want to train the cost model from scratch or use the [pre-trained](https://github.com/nullplay/Workload-Aware-Co-Optimization/edit/main/README.md#pretrained-model) model, you need a system that has a GPU with [PyTorch](https://pytorch.org/get-started/locally/) and [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) installed.

- Please follow the instructions in https://github.com/NVIDIA/MinkowskiEngine#requirements to install dependencies needed.  
#### 2. Install hnswlib with a Python binding
```
cd $WACO_HOME/hnswlib
pip install .
```
#### 3. Install [TACO](https://github.com/tensor-compiler/taco)
```
cd $WACO_HOME/code_generator/taco
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```
#### 4. Make code_generator
```
cd $WACO_HOME/code_generator
make icc # 'make gcc' if you use gcc
```

## How to use
This repository includes WACO for SpMV, SpMM, and SDDMM. While this README focuses on co-optimizing SpMM, you can easily apply these steps to SpMV and SDDMM as well.
Basically, WACO is consisted of four stages and you can walk through each stage by following instructions. 

**1. Generate a training dataset.**
```
cd $WACO_HOME/WACO/training_data_generator
python SpMM_SuperSchedule_Generator.py ../SpMM/TrainingData/total.txt
```
Then you can see randomly sampled SuperSchedules in `./config/` for each sparse matrix in `../SpMM/TrainingData/total.txt`.
Note that all the sparse matrices should placed in `$WACO_HOME/dataset/`.

Once we generate randomly sampled SuperSchedules, user needs to collect a runtime of (sparse matrix, SuperSchedule). To do this :
```
cd $WACO_HOME/WACO/code_generator
./spmm ../dataset/"YourSparseMatrixName".csr ../WACO/training_data_generator/config/"YourSparseMatrixName".txt
(Example) ./spmm ../dataset/nemspmm1_16x4_0.csr ../WACO/training_data_generator/config/nemspmm1_16x4_0.txt
```
You can see the code_generator running sampled SuperSchedules(`../WACO/training_data_generator/config/nemspmm1_16x4_0.txt`) for a given sparse matrix(`../dataset/nemspmm1_16x4_0.csr`) and printing runtimes at stdout. 

Once you collect all the runtimes from stdout, save training data into `$WACO_HOME/WACO/SpMM/TrainingData/CollectedData/"YourSparseMatrixName".txt`. We put some of examples in `$WACO_HOME/WACO/SpMM/TrainingData/CollectedData/`. 
```
(Example of training dataset for EX1_8x8_4)
cat $WACO_HOME/WACO/SpMM/TrainingData/CollectedData/EX1_8x8_4.txt
```

The cost model needs a lot of collected runtimes to accurately predicts the runtime for an arbitrary sparse matrix. In the paper, we collected runtimes for 7,000 - 20,000 sparse matrices and 100 SuperSchedules for each sparse matrix. It took 3-4days with 10 machines. We already put some samples(10) of training dataset, so you can proceed into next step without collecting runtimes for your machine.

**2. Training a cost model and building a KNN graph.**
```
cd $WACO_HOME/WACO/SpMM
python train.py      ## This trains the model for 80 epochs
python build_hnsw.py ## This build the KNNGraph using hnswlib
```

`python train.py` will create a trained model : `resnet.pth`  
`python build_hnsw.py` will build a KNNgraph : `hnsw_schedule.bin`

**3. Search the Top-20 SuperSchedules for a given input sparse matrix.**
```
cd $WACO_HOME/WACO/SpMM
python topk_search.py
cd topk
```
You can see the Top-20 SuperSchedules for each sparse matrix that WACO found.   
`python topk_search.py` does a search the best SuperSchedule for every sparse matrices in `$WACO_HOME/WACO/SpMM/TrainingData/test.txt`.  
Note that all the sparse matrices are stored in `$WACO_HOME/dataset`

**4. Test performances of SuperSchedules found by WACO.**
```
cd $WACO_HOME/code_generator
./spmm ../dataset/"YourSparseMatrixName".csr $WACO_HOME/WACO/SpMM/topk/"YourSparseMatrixName".txt
(Example) ./spmm ../dataset/bcsstk38.csr $WACO_HOME/WACO/SpMM/topk/bcsstk38.txt
```

## Pretrained model
Collecting runtimes and training a cost model need a lot of time (approximately 1-2 weeks). For a quick evaluation, we provide pre-trained cost models which are trained on ***Intel(R) Xeon(R) CPU E5-2680 v3 with an icc-compiled*** TACO kernel. 

Top-20 Schedules, collected Runtimes, built KNNGraphs, and pre-trained cost models that we've used in the paper can be found at : 
https://www.dropbox.com/s/mos4jtma4jmqkje/pretrained.zip?dl=0


```
pretrained
├── SDDMM
│   ├── TrainingData
│   │   ├── CollectedData  [19262 entries]
│   │   ├── test.txt
│   │   ├── total.txt
│   │   ├── train.txt
│   │   └── validation.txt
│   ├── hnsw_schedule.bin
│   ├── resnet.pth
│   └── topk  [975 entries]
├── SpMM
│   ├── TrainingData
│   │   ├── CollectedData  [21481 entries]
│   │   ├── test.txt
│   │   ├── total.txt
│   │   ├── train.txt
│   │   └── validation.txt
│   ├── hnsw_schedule.bin
│   ├── resnet.pth
│   └── topk  [975 entries]
├── SpMV
│   ├── TrainingData
│   │   ├── CollectedData  [21481 entries]
│   │   ├── test.txt
│   │   ├── total.txt
│   │   ├── train.txt
│   │   └── validation.txt
│   ├── hnsw_schedule.bin
│   ├── resnet.pth
│   └── topk  [975 entries]
└── dataset  [975 entries]
```

`dataset` directory includes 975 sparse matrices from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) in which a file format are converted into our custom .csr format. 

To test a pretrained cost model for SpMM,
```
cp -r pretrained/SpMM/* $WACO_HOME/WACO/SpMM/
cp -r pretrained/dataset/* $WACO_HOME/dataset/
cd $WACO_HOME/code_generator
./spmm ../dataset/"YourSparseMatrixName".csr $WACO_HOME/WACO/SpMM/topk/"YourSparseMatrixName".txt
(Example) ./spmm ../dataset/bcsstk38.csr $WACO_HOME/WACO/SpMM/topk/bcsstk38.txt
```

To test a pretrained cost model for all 975 test matrices,
```
while read mtx; do echo $mtx &&  ./spmm ../dataset/$mtx.csr $WACO_HOME/WACO/SpMM/topk/$mtx.txt ; done < $WACO_HOME/WACO/SpMM/TrainingData/test.txt
```

Note that WACO's cost model predicts a runtime of a sparse tensor program assuming running on a specific architecture. Therefore, you may not observe a speedup if the characteristic of your system is significantly different from the system that a pretrained model assumes.
