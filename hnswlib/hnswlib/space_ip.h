#pragma once
#include "hnswlib.h"
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>

float weight0[256*128];
float weight1[128*64];
float weight2[64*1];
float bias0[128];
float bias1[64];
float bias2[1];

const int embedsz = 256; //SpMV

namespace hnswlib {

    static float
    InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        size_t dim = *((size_t *) qty_ptr);

        //Concatenate
        float in[embedsz];
        for (int i = 0; i<embedsz/2; i++) {
          in[i] = ((float*) pVect1)[i]; //Query
          in[i+embedsz/2] = ((float*) pVect2)[i]; //Item
        }

        // 1. Linear(256,128)+BN+ReLU
        float out0[128];
        for (int o = 0; o<128; o++) {
          out0[o] = bias0[o];
          for (int i = 0; i<embedsz; i++) {
            out0[o] += in[i] * weight0[o*embedsz+i]; 
          }
          // ReLU
          out0[o] = std::max(out0[o], (float)0.0); 
        }

        // 2. Linear(128,64)+ReLU
        float out1[64];
        for (int o = 0; o<64; o++) {
          out1[o] = bias1[o];
          for (int i = 0; i<128; i++) {
            out1[o] += out0[i] * weight1[o*128+i]; 
          }
          out1[o] = std::max(out1[o], (float)0.0); //ReLU
        }
        
        // 3. Linear(64,1)
        float res = bias2[0];
        for (int i = 0; i<64; i++) {
          res += out1[i]*weight2[i]; 
        }
        
        return res;
    }

    class InnerProductSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpace(size_t dim) {
            fstdistfunc_ = InnerProduct;
            dim_ = dim;
            data_size_ = dim * sizeof(float);

            std::ifstream weight0_, weight1_, weight2_;
            char* weight0_path, weight1_path, weight2_path;
            char* bias0_path, bias1_path, bias2_path;
            char* env_val = getenv("WACO_HOME");
            if (env_val == NULL) { 
              std::cout << "ERR : Environment variable WACO_HOME not defined" << std::endl; 
              exit(1);
            }
            
            std::string waco_prefix = env_val;
            weight0_.open(waco_prefix+"/hnswlib/WACO_COSTMODEL/weight0.txt");
            weight1_.open(waco_prefix+"/hnswlib/WACO_COSTMODEL/weight1.txt");
            weight2_.open(waco_prefix+"/hnswlib/WACO_COSTMODEL/weight2.txt");
            for (int i = 0; i<256*128; i++) weight0_ >> weight0[i];
            for (int i = 0; i<128*64; i++) weight1_ >> weight1[i];
            for (int i = 0; i<64*1; i++) weight2_ >> weight2[i];

            std::ifstream bias0_, bias1_, bias2_;
            bias0_.open(waco_prefix+"/hnswlib/WACO_COSTMODEL/bias0.txt");
            bias1_.open(waco_prefix+"/hnswlib/WACO_COSTMODEL/bias1.txt");
            bias2_.open(waco_prefix+"/hnswlib/WACO_COSTMODEL/bias2.txt");
            for (int i = 0; i<128; i++) bias0_ >> bias0[i];
            for (int i = 0; i<64; i++) bias1_ >> bias1[i];
            for (int i = 0; i<1; i++) bias2_ >> bias2[i];

            weight0_.close();
            weight1_.close();
            weight2_.close();
            bias0_.close();
            bias1_.close();
            bias2_.close();
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

    ~InnerProductSpace() {}
    };


}
