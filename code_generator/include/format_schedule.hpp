#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <parallel/algorithm>
#include "time.hpp"
using namespace std;

typedef enum { COMPRESSED, UNCOMPRESSED } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;


struct FormatInfo {
  string var;
  int dimension;
  taco_mode_t mode;
  int startbit;
  int lenbit;
};

class FormatScheduler {
  protected :
    vector<FormatInfo> TensorFormat;
    vector<FormatInfo> TensorFormat_init;
    vector<pair<uint64_t, float>> coo;
    vector<vector<int>> T_pos, T_crd;
    vector<float> T_vals;
    taco_tensor_t *T=NULL;
    bool is_dense; // All ranks are Uncompressed
    bool is_dense_fix; // All ranks are Uncompressed

  protected :
    inline int get_digit2(int x) { return (int)ceil(log2(x)); }
    inline int extract(uint64_t coords, int start, int len) {
      uint64_t result = (coords >> start);
      result = result & ((1<<len) - 1);
      return (int)result;  
    }
    inline uint64_t extractuppercoords(uint64_t coords, int start) {
      return coords >> start;
    }
    
    void init_taco_tensor_t() {
      int num_rank = TensorFormat.size();
      T = (taco_tensor_t *) new taco_tensor_t[1];
      T->order = num_rank;
      T->dimensions = (int32_t *) new int32_t[num_rank];
      T->mode_types = (taco_mode_t *) new taco_mode_t[num_rank];
      T->indices = (uint8_t ***) new uint8_t**[num_rank];
      for (int32_t rank = 0; rank < num_rank; rank++) {
        T->dimensions[rank] = TensorFormat[rank].dimension;
        T->mode_types[rank] = TensorFormat[rank].mode;
        T->indices[rank] = (uint8_t **) new uint8_t*[2];
        switch (TensorFormat[rank].mode) {
          case COMPRESSED:
            T->indices[rank][0] = (uint8_t*)(T_pos[rank].data());
            T->indices[rank][1] = (uint8_t*)(T_crd[rank].data());
            break;
        }
      }
      T->vals = (uint8_t*)(T_vals.data());
    }

    void destroy_taco_tensor_t() {
      //cout << "Destroy " << flush;
      //cout << T->order << flush;
      int num_rank = T->order;
      delete[] T->dimensions;
      delete[] T->mode_types;
      for (int rank=0; rank<num_rank; rank++) {
        delete[] T->indices[rank];
      }
      delete[] T->indices;
      delete[] T;
    }


  public :
    vector<FormatInfo>& get_format() {return TensorFormat;}
    vector<float>& get_vals() {return T_vals;}
    void init() { TensorFormat = TensorFormat_init; }
    void set_coo(vector<pair<uint64_t, float>>& coo_) {coo = coo_;}
    ~FormatScheduler() {if(T!=NULL) destroy_taco_tensor_t(); }

    bool is_var_exist(string var) {
      for (int rank = 0; rank<TensorFormat.size(); rank++) {
        if (TensorFormat[rank].var == var) { return true; }
      }
      return false;
    }

    string print_format() {
      stringstream ss;
      ss << "Rank\tVar\tDim\tMode\tStartb\tLenb" << endl;
      int rank = 0;
      for (FormatInfo& format : TensorFormat) {
        ss << rank++ << "\t";
        ss << format.var << "\t";
        ss << format.dimension << "\t";
        ss << (format.mode ? "U" : "C") << "\t";
        ss << format.startbit << "\t";
        ss << format.lenbit;
        ss << endl;
      }
      return ss.str();
    }

    // Rank order in TensorFormat starts from the highest rank.
    // e.g.) A[i,k](U,C) => TensorFormat = [{"i",U}, {"k",C}]
    //       Then COO : {"i", "k"}
    FormatScheduler(vector<pair<uint64_t, float>>& coo, vector<FormatInfo>& init) : coo(coo), TensorFormat(init) {
      int num_rank = TensorFormat.size();
      TensorFormat[num_rank-1].startbit = 0; // Lowest Rank first
      TensorFormat[num_rank-1].lenbit = get_digit2(TensorFormat[num_rank-1].dimension);
      for (int rank = num_rank-2; rank >= 0; rank--) {
        TensorFormat[rank].startbit = TensorFormat[rank+1].startbit + TensorFormat[rank+1].lenbit; 
        TensorFormat[rank].lenbit = get_digit2(TensorFormat[rank].dimension);
      }
      TensorFormat_init = TensorFormat;
      is_dense_fix = false;
    }

    FormatScheduler(vector<float>& dense, vector<FormatInfo>& init) : TensorFormat(init), TensorFormat_init(init) {
      is_dense_fix = true;
      is_dense = true;
      T_vals = dense;
    }

    // |<-----i----->|
    // |<-i1->|<-i0->|
    FormatScheduler& split(string var, string outer_var, string inner_var, int split_size) {
      for (int rank = 0; rank < TensorFormat.size(); rank++) {
        if (TensorFormat[rank].var == var) {
          FormatInfo inner_rank;
          inner_rank.var = inner_var;
          inner_rank.dimension = min(split_size, TensorFormat[rank].dimension);
          inner_rank.mode = TensorFormat[rank].mode;
          inner_rank.startbit = TensorFormat[rank].startbit; // in bit
          inner_rank.lenbit = get_digit2(inner_rank.dimension); // in bit

          TensorFormat[rank].var = outer_var;
          TensorFormat[rank].dimension = (TensorFormat[rank].dimension + inner_rank.dimension - 1)/inner_rank.dimension;
          TensorFormat[rank].mode = TensorFormat[rank].mode;
          TensorFormat[rank].startbit = TensorFormat[rank].startbit + get_digit2(split_size);
          TensorFormat[rank].lenbit = get_digit2(TensorFormat[rank].dimension);

          TensorFormat.insert(TensorFormat.begin()+rank+1, inner_rank);
          break; 
        }
      }
      return *this;
    }

    FormatScheduler& reorder(vector<string> reordered_vars) {
      if (reordered_vars.size() != TensorFormat.size()) return *this;
      vector<FormatInfo> ReorderedTensorFormat;
      for (string& var : reordered_vars) {
        for (int rank = 0; rank < TensorFormat.size(); rank++) {
          if (TensorFormat[rank].var == var) {
            ReorderedTensorFormat.push_back(TensorFormat[rank]);
          }
        }
      }
      TensorFormat = ReorderedTensorFormat;
      return *this;
    }

    FormatScheduler& mode(string var, taco_mode_t mode) {
      is_dense = true;
      for (int rank = 0; rank < TensorFormat.size(); rank++) {
        if (TensorFormat[rank].var == var) {
          TensorFormat[rank].mode = mode; 
        }
        is_dense &= (TensorFormat[rank].mode == UNCOMPRESSED);
      }
      return *this;
    }
    
    void pack() {
      if (is_dense&&is_dense_fix) {
        return;
      }
      //cout << "Pack1 " << flush;
      if (T != NULL) destroy_taco_tensor_t();
      //cout << "Pack2 " << flush;
      auto t1 = Clock::now();
      int nnz = coo.size();
      int num_rank = TensorFormat.size();
      vector<int> newstartbit(num_rank);
      vector<pair<uint64_t, float>> pack_coo(nnz);
      vector<uint64_t> uniq_coords(num_rank, -1);
      T_pos = vector<vector<int>>(num_rank, vector<int>());
      T_crd = vector<vector<int>>(num_rank, vector<int>());
      T_vals = vector<float>();
      //cout << "Pack3 " << flush;

      int prefixsum = 0;
      for (int rank = num_rank-1; rank>=0; rank--) {
        newstartbit[rank] = prefixsum;
        prefixsum += TensorFormat[rank].lenbit;
      }

      #pragma omp parallel for
      for (int i = 0; i<nnz; i++) {
        uint64_t coords = coo[i].first;
        uint64_t pack_coords = 0;
        for (int rank = num_rank-1; rank>=0; rank--) {
          uint64_t rank_coord = extract(coords, TensorFormat[rank].startbit, TensorFormat[rank].lenbit);
          pack_coords |= (rank_coord << newstartbit[rank]); 
        }
        pack_coo[i] = {pack_coords, coo[i].second};
      }
      
      __gnu_parallel::sort(pack_coo.begin(), pack_coo.end());
      
      //cout << "Pack4 " << flush;
      int limit = nnz*20; // 5e8;
      for (int i = 0; i < nnz; i++) {
        uint64_t coords = pack_coo[i].first;
        long pos_idx = 0;
        for (int rank = 0; rank < num_rank; rank++) {
          int rank_coord = extract(coords, newstartbit[rank], TensorFormat[rank].lenbit);
          uint64_t upper_coords = extractuppercoords(coords, newstartbit[rank]); 
          switch (TensorFormat[rank].mode) {
            case UNCOMPRESSED:
              pos_idx = pos_idx * TensorFormat[rank].dimension + rank_coord;
              break;
            case COMPRESSED:
              if (upper_coords != uniq_coords[rank]) {
                uniq_coords[rank] = upper_coords;
                T_crd[rank].push_back(rank_coord);
                if (T_pos[rank].size() <= pos_idx+1) {
                  if (pos_idx > limit) {
                    T=NULL;
                    throw std::invalid_argument("Pack Too Large");
                  }
                  T_pos[rank].resize((pos_idx+1000000), 0);
                }
                T_pos[rank][pos_idx+1]++;
              }
              pos_idx = T_crd[rank].size()-1;
              break;
          }
        }
        // case VALUEARRAY: 
        if (T_vals.size() <= pos_idx) {
          if (pos_idx > limit) {
            T=NULL;
            throw std::invalid_argument("Pack Too Large");
          }
          T_vals.resize((pos_idx+1000000), 0);
        }
        T_vals[pos_idx] = pack_coo[i].second;
      }
      //cout << "Pack5 " << flush;

      int format_size = 0;
      long pos_size = 1;
      for (int rank = 0; rank < num_rank; rank++) {
        switch (TensorFormat[rank].mode) {
          case UNCOMPRESSED :
            format_size += 1;
            pos_size *= TensorFormat[rank].dimension;
            break;
          case COMPRESSED :
            if (T_pos[rank].size() < pos_size) {
              if (pos_size > limit) {
                T=NULL;
                throw std::invalid_argument("Pack Too Large");
              }
              T_pos[rank].resize(pos_size+1, 0);
            }
            for (int i = 0; i < pos_size; i++)  
              T_pos[rank][i+1] += T_pos[rank][i];
            format_size += pos_size + T_crd[rank].size();
            pos_size = T_crd[rank].size();
            break;
        }
      }
      format_size += pos_size; //value arr
      if (T_vals.size() < pos_size) {
        //cout << pos_size << endl;
        if (pos_size > limit) {
          T=NULL;
          throw std::invalid_argument("Pack Too Large");
        }
        T_vals.resize(pos_size+1, 0);
      }
     
      //cout << "Format Conversion Time (ms) : " << compute_clock(Clock::now(), t1) << " ms " << endl;
      //cout << "Converted Format Size  (MB) : " << format_size*4/1e6 << " MB " << endl;
    }

    taco_tensor_t* get_taco_tensor() {
      init_taco_tensor_t();
      return T;
    }
};

