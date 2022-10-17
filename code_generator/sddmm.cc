#include <iostream>
#include <string>
#include <random>
#include <set>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <map>
#include <immintrin.h>
#include <vector>
#include <fstream>
#include <limits>
#include <cmath>
#include <iomanip>
#include "format_schedule.hpp"
#include "execution_manager.hpp"
#include "time.hpp"
using namespace std;

default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
uniform_real_distribution<float> uniform(-1.0, 1.0);

int roundup(int num, int multiple) {
  return ((num+multiple-1)/multiple) * multiple;
}

int main(int argc, char* argv[]) {
  if (argc != 3) { cout << "Wrong arguments" << endl; exit(-1); }
  string mtx_name(argv[1]);

  //////////////////////////
  // Reading CSR A from file
  //////////////////////////
  int num_row, num_col, num_nonzero;
  fstream csr(argv[1]); 
  csr.read((char*)&num_row, sizeof(int));
  csr.read((char*)&num_col, sizeof(int));
  csr.read((char*)&num_nonzero, sizeof(int));
  vector<int> A_crd(num_nonzero);
  vector<int> A_pos(num_row+1);
  vector<float> A_val(num_nonzero);
  for (int i=0; i<num_row+1; i++) {
    int data;
    csr.read((char*)&data, sizeof(data));
    A_pos[i] = data;
  }
  for (int i=0; i<num_nonzero; i++) {
    int data;
    csr.read((char*)&data, sizeof(data));
    A_crd[i] = data;
    A_val[i] = 1.0; 
  }
  csr.close();
  cout << "NUMROW : " << num_row << " / NUMCOL : " << num_col << " / NNZ : " << num_nonzero << " "  ;
  cout << endl;

  vector<pair<uint64_t, float>> coo(num_nonzero);
  vector<pair<uint64_t, float>> coo_out(num_nonzero);
  int col_digit2 = (int)(ceil(log2(num_col)));
  #pragma omp parallel for schedule(dynamic, 128)
  for (int i = 0; i<num_row; i++) {
    for(int j = A_pos[i]; j<A_pos[i+1]; j++) {
      uint64_t i_ = (((uint64_t)i) << col_digit2);
      uint64_t t = i_ | A_crd[j];
      coo[j] = {t, A_val[j]};
      coo_out[j] = {t,0};
    }
  }


  ////////////////////////////
  // Generating Random Dense B,D
  ////////////////////////////
  int N = 256; // number of k in C(i,j) = A(i,j) * B(i,k) * D(k,j)
  vector<float> B = vector<float>(roundup(num_row,1024)*N,1);
  vector<float> D = vector<float>(roundup(num_col,1024)*N,1);
  
  ExecutionManager M;
  vector<FormatInfo> TensorC, TensorA, TensorB, TensorD;
  TensorC.push_back({"i", num_row, UNCOMPRESSED});
  TensorC.push_back({"j", num_col, COMPRESSED});
  TensorA.push_back({"i", num_row, UNCOMPRESSED});
  TensorA.push_back({"j", num_col, COMPRESSED});
  TensorB.push_back({"i", num_row, UNCOMPRESSED});
  TensorB.push_back({"k", N, UNCOMPRESSED});
  TensorD.push_back({"j", num_col, UNCOMPRESSED});
  TensorD.push_back({"k", N, UNCOMPRESSED});

  M.add_tensor("C", TensorC, coo_out, true);
  M.add_tensor("A", TensorA, coo, false); 
  M.add_tensor("B", TensorB, B, false);
  M.add_tensor("D", TensorD, D, false);
  cout << "Use " << NUMCORE << " Threads" << endl; 

  M.init_all();
  M.parallelize("i", 48, 32);
  M.pack_all();
  M.compile();
  stringstream fixedCSR;
  fixedCSR << "FixedCSR : " << M.run(10, 50, false) << " ms" << endl;

  string arg(argv[2]);
  fstream arg_file(arg);
  string schedule;
  string bestSuperSchedule;
  float bestTime=1000000000;
  for (; getline(arg_file, schedule) ;) {
    stringstream ss(schedule);
    int isplit, ksplit, jsplit;
    vector<string> r(6);
    vector<int> f(4);
    string pidx;
    int pnum, pchunk;
    ss >> isplit >> jsplit >> ksplit;    
    ss >> r[0] >> r[1] >> r[2] >> r[3] >> r[4] >> r[5];  
    ss >> f[0] >> f[1] >> f[2] >> f[3];  
    ss >> pidx >> pnum >> pchunk;
    
    // Erase Split Size = 1 in loop order "r"
    if (isplit == 1) {
      r.erase(find(r.begin(), r.end(), "i0"));
      auto itr = find(r.begin(), r.end(), "i1");
      *itr = "i";
    } if (ksplit == 1) {
      r.erase(find(r.begin(), r.end(), "k0"));
      auto itr = find(r.begin(), r.end(), "k1");
      *itr = "k";
    } if (jsplit == 1) {
      r.erase(find(r.begin(), r.end(), "j0"));
      auto itr = find(r.begin(), r.end(), "j1");
      *itr = "j";
    }

    // Extract format order of A
    vector<string> rA;
    for (string idx : r) {
      if (idx[0] == 'i' || idx[0] == 'j') {
        rA.push_back(idx);
      }
    }
    
    try {
      M.init_all();
      if (isplit != 1) { M.fsplit("A", "i", "i1", "i0", isplit);}
      if (ksplit != 1) { M.fsplit("A", "k", "k1", "k0", ksplit);}
      if (jsplit != 1) { M.fsplit("A", "j", "j1", "j0", jsplit);}
      M.lreorder(r);
      M.freorder("A", rA);
      M.freorder("C", rA);
      if (isplit != 1) { 
        M.fmode("A", "i1", f[0]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("A", "i0", f[1]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("C", "i1", f[0]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("C", "i0", f[1]==0 ? COMPRESSED : UNCOMPRESSED);
        if (pidx[0] == 'i') M.parallelize(pidx, pnum, pchunk);
      } else { 
        M.fmode("A", "i", f[0]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("C", "i", f[0]==0 ? COMPRESSED : UNCOMPRESSED);
        if (pidx[0] == 'i') M.parallelize("i", pnum, pchunk);
      } 
      if (jsplit != 1) { 
        M.fmode("A", "j1", f[2]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("A", "j0", f[3]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("C", "j1", f[2]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("C", "j0", f[3]==0 ? COMPRESSED : UNCOMPRESSED);
        if (pidx[0] == 'j') M.parallelize(pidx, pnum, pchunk);
      } else { 
        M.fmode("A", "j", f[2]==0 ? COMPRESSED : UNCOMPRESSED)
         .fmode("C", "j", f[2]==0 ? COMPRESSED : UNCOMPRESSED);
        if (pidx[0] == 'j') M.parallelize("j", pnum, pchunk);
      }           
      M.pack_all();
      M.compile();
      float avgtime = M.run(5,30,false);
      cout << "Testing Candidate SuperSchedules... " << schedule << " = " << avgtime << " ms" << endl;
      if (bestTime > avgtime) {
        bestTime = avgtime;
        bestSuperSchedule = schedule;
      }
    } catch(...) {
    }
  }
  cout << endl;
  cout << "SuperSchedule found by WACO : " << bestSuperSchedule << endl; 
  cout << "WACO : " << bestTime << " ms" << endl;
  cout << fixedCSR.str() << endl;


  return 0;
}  
