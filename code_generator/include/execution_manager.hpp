#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstdlib>

extern string suffix="";
using namespace std;
using namespace std::chrono_literals;

//typedef int (*compute)(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B, float* C_vals, float* A_vals, float* B_vals);
//typedef int (*compute2)(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B, taco_tensor_t* D, float* C_vals, float* A_vals, float* B_vals, float* D_vals);
typedef int (*compute)(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B);
typedef int (*compute2)(taco_tensor_t* C, taco_tensor_t* A, taco_tensor_t* B, taco_tensor_t* D);

class ExecutionManager{
  private :
    map<string, FormatScheduler*> tensor_lhs;
    map<string, FormatScheduler*> tensor_rhs;
    compute func;
    compute2 func2;
    void* lib_handle=NULL;
    vector<float> ref;
    string parallel_var;
    bool is_parallel=false;
    vector<string> reorder_var;
    float timeout=0;
    
    
    string gen_command() {
      string kernel = "\"";
      string format = "";
      string precision = "";
      for (auto& it : tensor_lhs) {
        format += " -f=";
        format += it.first;
        format += ":";
        kernel += it.first;
        kernel += "(";
        precision += " -t=" + it.first + ":float";
        vector<FormatInfo> TensorFormat = it.second->get_format();
        for (int rank = 0; rank<TensorFormat.size(); rank++) {
          if (TensorFormat[rank].mode == UNCOMPRESSED) {
            format += "d";
          } else if (TensorFormat[rank].mode == COMPRESSED) {
            format += "s";
          }
          kernel += TensorFormat[rank].var;
          if (rank==TensorFormat.size()-1) kernel += ")";
          else kernel += ",";
        }
        format += ":";
        for (int rank = 0; rank<TensorFormat.size(); rank++) {
          format += to_string(rank) + ",";
        }
        format.pop_back();
      }
      kernel += "=";
      for (auto& it : tensor_rhs) {
        format += " -f=";
        format += it.first;
        format += ":";
        kernel += it.first;
        kernel += "(";
        precision += " -t=" + it.first + ":float";
        vector<FormatInfo> TensorFormat = it.second->get_format();
        for (int rank = 0; rank<TensorFormat.size(); rank++) {
          if (TensorFormat[rank].mode == UNCOMPRESSED) {
            format += "d";
          } else if (TensorFormat[rank].mode == COMPRESSED) {
            format += "s";
          }
          kernel += TensorFormat[rank].var;
          if (rank==TensorFormat.size()-1) kernel += ")";
          else kernel += ",";
        }
        format += ":";
        for (int rank = 0; rank<TensorFormat.size(); rank++) {
          format += to_string(rank) + ",";
        }
        format.pop_back();
        kernel += "*";
      }
      kernel.pop_back();
      kernel+="\"";

      return kernel + format + precision;
    }
   
    bool IsPowerOfTwo(int x){
      return (x & (x - 1)) == 0;
    }

    string gen_sched() {
      map<string, bool> vars;
      map<string, int> dims;
      for (auto& it : tensor_lhs) {
        FormatScheduler* t = it.second;
        for(auto& rank : t->get_format()) {
          if (vars.find(rank.var) == vars.end()) {
            vars[rank.var] = rank.mode==UNCOMPRESSED;
            dims[rank.var] = rank.dimension;
          } else {
            vars[rank.var] &= (rank.mode==UNCOMPRESSED);
          }
        }
      }
      for (auto& it : tensor_rhs) {
        FormatScheduler* t = it.second;
        for(auto& rank : t->get_format()) {
          if (vars.find(rank.var) == vars.end()) {
            vars[rank.var] = rank.mode==UNCOMPRESSED;
            dims[rank.var] = rank.dimension;
          } else {
            vars[rank.var] &= (rank.mode==UNCOMPRESSED);
          }
        }
      }

      string schedule = "";
      for (auto& it : vars) {
        if (it.second == true) {
          schedule += " -s=\"bound(" + it.first + "," + it.first + "b," + to_string(dims[it.first]) + ", MaxExact)\"";
        }
      }

      // Reorder
      if (reorder_var.size() > 0) {
        schedule += " -s=\"reorder(";
        for (int i = 0; i<reorder_var.size(); i++) {
          string var = reorder_var[i];
          schedule += var + (vars[var] ? "b" : "");
          if (i<reorder_var.size()-1) {
            schedule += ",";
          } else {
            schedule += ")\"";
          }
        }
      } 
      
      // Parallelize
      if (is_parallel) {
        bool is_reduction = true;
        for (auto t : tensor_lhs) {
          FormatScheduler* t_format = t.second;
          is_reduction &= !(t_format->is_var_exist(parallel_var));
        }
        schedule += " -s=\"parallelize(" + parallel_var + (vars[parallel_var] ? "b" : "") + ",CPUThread," + (is_reduction ? "Atomics" : "NoRaces") +")\"";
      }

      return schedule;
    }


  public :
    ExecutionManager() {}
    ExecutionManager(vector<float>& ref):ref(ref) {}
    ExecutionManager(vector<float>& ref, float timeout):ref(ref), timeout(timeout) {}
    ~ExecutionManager() {
      for (auto& it: tensor_lhs) {delete it.second;}
      for (auto& it: tensor_rhs) {delete it.second;}
      if (lib_handle) {dlclose(lib_handle); } 
    }
    

    void mod_tensor(string tensorname, vector<pair<uint64_t, float>>& coo) {
      auto t = tensor_lhs.find(tensorname);
      if (t != tensor_lhs.end()) { t->second->set_coo(coo);} 
      else {
        t = tensor_rhs.find(tensorname);
        if (t != tensor_rhs.end()) { t->second->set_coo(coo);}
      }
    }

    void add_tensor(string tensorname, vector<FormatInfo> tensorformat, vector<pair<uint64_t, float>>& coo, bool lhs) {
      FormatScheduler* T = new FormatScheduler(coo, tensorformat);
      if (lhs) {tensor_lhs[tensorname] = T;}
      else {tensor_rhs[tensorname] = T;}    
    }

    void add_tensor(string tensorname, vector<FormatInfo> tensorformat, vector<float>& dense, bool lhs) {
      FormatScheduler* T = new FormatScheduler(dense, tensorformat);
      if (lhs) {tensor_lhs[tensorname] = T;}
      else {tensor_rhs[tensorname] = T;}
    }

    FormatScheduler& get_tensor(string tensorname) {
      auto t = tensor_lhs.find(tensorname);
      if (t != tensor_lhs.end()) { return *(t->second);} 
      else {
        t = tensor_rhs.find(tensorname);
        if (t != tensor_rhs.end()) { return *(t->second);}
      }
    }
    
    ExecutionManager& fsplit(string tensorname, string var, string outer_var, string inner_var, int split_size) {
      for (auto& it : tensor_lhs) { it.second->split(var,outer_var,inner_var,split_size); } 
      for (auto& it : tensor_rhs) { it.second->split(var,outer_var,inner_var,split_size); } 
      return *this;
    }
 
    ExecutionManager& lreorder(vector<string> reordered_vars) {
      reorder_var = reordered_vars;
      return *this; 
    }   
    
    ExecutionManager& freorder(string tensorname, vector<string> reordered_vars) {
      auto t = tensor_lhs.find(tensorname);
      if (t != tensor_lhs.end()) {
        t->second->reorder(reordered_vars);
      } else {
        t = tensor_rhs.find(tensorname);
        if (t != tensor_rhs.end()) {
          t->second->reorder(reordered_vars);
        }
      }
      return *this;
    }

    ExecutionManager& fmode(string tensorname, string var, taco_mode_t mode) {
      auto t = tensor_lhs.find(tensorname);
      if (t != tensor_lhs.end()) {
        t->second->mode(var, mode);
      } else {
        t = tensor_rhs.find(tensorname);
        if (t != tensor_rhs.end()) {
          t->second->mode(var, mode);
        }
      }
      return *this;
    }

    ExecutionManager& parallelize(string var, int num_thread, int chunk_size) {
      // Test Variable Existence
      bool is_exist = false;
      for (auto t : tensor_lhs) {
        FormatScheduler* t_format = t.second;
        is_exist |= (t_format->is_var_exist(var));
      } for (auto t : tensor_rhs) {
        FormatScheduler* t_format = t.second;
        is_exist |= (t_format->is_var_exist(var));
      }
      if (is_exist == false) {
        cerr << "[Paralleize] There is no " << var << " in notation" << endl;
        exit(-1);
      }
      //omp_set_num_threads(num_thread);
      omp_set_num_threads(NUMCORE);
      omp_set_schedule(omp_sched_dynamic, chunk_size);
      is_parallel = num_thread > 1;
      parallel_var = var;
      return *this;
    }

    void pack_all() {
      for (auto& it : tensor_lhs) { it.second->pack(); }
      for (auto& it : tensor_rhs) { it.second->pack(); }
    }

    void print_all() {
      for (auto& it : tensor_lhs) { cout << it.second->print_format(); }
      for (auto& it : tensor_rhs) { cout << it.second->print_format(); }
    }
    
    string print_tensor(string tensorname) {
      auto t = tensor_lhs.find(tensorname);
      if (t != tensor_lhs.end()) {
        return t->second->print_format();
      } else {
        t = tensor_rhs.find(tensorname);
        if (t != tensor_rhs.end()) {
          return t->second->print_format();
        }
      }
    }

    void init_all() {
      reorder_var.clear();
      parallel_var = "";
      is_parallel=false;
      for (auto& it : tensor_lhs) { it.second->init(); }
      for (auto& it : tensor_rhs) { it.second->init(); }
    }
 
    void compile() {
      char* env_val = getenv("WACO_HOME");
      if (env_val == NULL) { 
        std::cout << "ERR : Environment variable WACO_HOME not defined" << std::endl; 
        exit(1);
      }
      std::string waco_prefix = env_val;

      string taco_command = waco_prefix+"/code_generator/taco/build/bin/taco ";
      string kernel = gen_command();
      string source = " -write-compute=taco_kernel"+suffix+".c";
      string schedule = gen_sched();
      taco_command += kernel + source + schedule ;
      string header = "#include <stdint.h> \\n" 
                      "typedef enum { COMPRESSED, UNCOMPRESSED } taco_mode_t;\\n"
                      "typedef struct {\\n"
                      "  int32_t      order;\\n" 
                      "  int32_t*     dimensions;\\n" 
                      "  int32_t      csize;\\n" 
                      "  int32_t*     mode_ordering;\\n" 
                      "  taco_mode_t* mode_types;\\n" 
                      "  uint8_t***   indices;\\n" 
                      "  uint8_t*     vals;\\n" 
                      "  int32_t      vals_size;\\n" 
                      "} taco_tensor_t;\\n";
      string taco_header_command = "sed -i '1s/^/" + header + "/' taco_kernel" + suffix + ".c";
      int taco_compile = system(taco_command.c_str());
      int taco_add_header = system(taco_header_command.c_str());
      if (tensor_rhs.size() == 3) {
        string patch_command = "python " + waco_prefix+"/code_generator/include/sddmm_patch.py ./taco_kernel"+suffix+".c";
        system(patch_command.c_str());
      }

      if (is_parallel) {
        #ifdef ICC
        string gcc_command = "icc -march=native -mtune=native -O3 -ffast-math -qopenmp -fPIC -shared taco_kernel" + suffix + ".c -o taco_kernel" + suffix + ".so -lm";
        #elif GCC
        string gcc_command = "gcc -march=native -mtune=native -O3 -fopenmp -ffast-math -fPIC -shared taco_kernel" + suffix + ".c -o taco_kernel" + suffix + ".so -lm";
        #endif
        int gcc_compile = system(gcc_command.c_str());
      }

      if (lib_handle) {dlclose(lib_handle);}
      string taco_kernel = "./taco_kernel"+suffix+".so";
      lib_handle = dlopen(taco_kernel.c_str(), RTLD_NOW|RTLD_LOCAL);
      if (!lib_handle) {cout << "DLOPEN - " << dlerror() << endl;}
      
      if (tensor_rhs.size() == 2) 
        func  = (compute)dlsym(lib_handle, "compute");
      else if (tensor_rhs.size() == 3) {
        func2 = (compute2)dlsym(lib_handle, "compute");
      }
      
      if (dlerror() != NULL) {cout << "DLSYM ERROR" << endl;}
    }

    double run(int warm=3, int round=50, bool verify=false) {
      vector<taco_tensor_t*> T;
      for (auto& it : tensor_lhs) { T.push_back(it.second->get_taco_tensor()); }
      for (auto& it : tensor_rhs) { T.push_back(it.second->get_taco_tensor()); }
      double elapsed_time;
      if (tensor_rhs.size() == 2) {
        for(int r=0; r<warm; r++) {
          auto t1 = Clock::now();
          //func(T[0], T[1], T[2], (float*)(T[0]->vals), (float*)(T[1]->vals), (float*)(T[2]->vals));
          func(T[0], T[1], T[2]);
          double tt = compute_clock(Clock::now(), t1);
          if (tt > 100) return tt;
        }
        vector<double> elapsed;

        for(int r=0; r<round; r++) {
          auto t1 = Clock::now();
          func(T[0], T[1], T[2]);
          double tt = compute_clock(Clock::now(), t1);
          elapsed.push_back(tt);
        }
        
        sort(elapsed.begin(), elapsed.end());
        elapsed_time = elapsed[elapsed.size()/2];
        
        if (verify) {
          for (auto& it : tensor_lhs) {
            FormatScheduler* t = it.second;
            vector<float>& res = t->get_vals();
            fill(res.begin(), res.end(), 0);
            func(T[0], T[1], T[2]);
            #pragma omp parallel for
            for(int i=0; i<ref.size(); i++) { 
              if (abs(ref[i]-res[i])>0.01) {
                cout << "Wrong " << i << " " << ref[i] << " " << res[i] << endl;
                exit(-1);
              }
            }
          }
        }
      } else if (tensor_rhs.size() == 3) { //MTTKRP
        for(int r=0; r<warm; r++) {func2(T[0], T[1], T[2], T[3]);}
        vector<double> elapsed;
        for(int r=0; r<round; r++) {
          auto t1 = Clock::now();
          //func2(T[0], T[1], T[2], T[3], (float*)(T[0]->vals), (float*)(T[1]->vals), (float*)(T[2]->vals), (float*)(T[3]->vals));
          func2(T[0], T[1], T[2], T[3]);
          double tt = compute_clock(Clock::now(), t1);
          if (tt > 1000) return tt;
          elapsed.push_back(tt);
        }
        sort(elapsed.begin(), elapsed.end());
        elapsed_time = elapsed[round/2];
        
        if (verify) {
          for (auto& it : tensor_lhs) {
            FormatScheduler* t = it.second;
            vector<float>& res = t->get_vals();
            fill(res.begin(), res.end(), 0);
            func2(T[0], T[1], T[2], T[3]);
            for (int i = 0; i<res.size(); i++) {
              if (abs(ref[i]-res[i])>0.01) {
                cout << "Wrong " << i << " " << ref[i] << " " << res[i] << endl;
                exit(-1);
              }
            }
          }
        }
      }

      return elapsed_time;
    }

};
