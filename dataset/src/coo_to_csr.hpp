#include <iostream>
#include <random>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
using namespace std;

void coo_to_csr(vector<pair<int,int>>& coo, int num_row, int num_col, string file_name) {
  int num_nonzero = coo.size();
  vector<int> uncompressed(num_row+1);
  for (pair<int, int>& coor : coo) {
    uncompressed[coor.first+1]++;
  }
  for (int i = 0; i < num_row; i++) {
    uncompressed[i+1] += uncompressed[i];
  }

  ofstream out("./" + file_name+".csr", ios::binary | ios::out);
  out.write((char*)&num_row, sizeof(int));
  out.write((char*)&num_col, sizeof(int));
  out.write((char*)&num_nonzero, sizeof(int));

  for (int i = 0; i<num_row+1; i++) {
    int src = uncompressed[i];
    out.write((char*)&src, sizeof(src));
  }

  for (int i = 0; i<num_nonzero; i++) {
    int dst = coo[i].second;
    out.write((char*)&dst, sizeof(dst));
  }
  out.close();
}
