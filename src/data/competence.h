//
// Created by revo on 4/18/19.
//
#pragma once

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/types.h"

#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
#include "3rd_party/cnpy/cnpy.h"
#include "common/options.h"

#include <math.h>
#include <string>

namespace marian {
namespace data {

class Competence {
private:
  std::unordered_map<Word, float> data_;
  Ptr<Options> options_;
  Ptr<Vocab> srcVocab_;
  struct cdf_data {
    float* data;
    std::vector<unsigned int> shape;
  };

  cdf_data cdf_data_;
  cdf_data base_data_;

  float T = 50000;
  float c0 = 0.01;


  // bool IsModTraining_ = false;
  // size_t converge_avg_mod = 29322;

  void load(const std::string& fname) {
    io::InputFileStream in(fname);
    std::string src;
    float prob;
    while(in >> src >> prob) {
      // @TODO: change this to something safer other than NULL
      if(src == "NULL")
        continue;
      Word sId = (*srcVocab_)[src];
      data_[sId] = prob;
    }
  }

  void cdf_load(const std::string& fname) {
    auto cdf = cnpy::npz_load(fname, "cdf");
    float* data = (float*)cdf->data();
    auto shape = cdf->shape;
    cdf_data_ = {new float[shape[0]], shape};
    std::copy(data, data + shape[0], cdf_data_.data);
    // base_data
    auto base = cnpy::npz_load(fname, "base");
    data = (float*)base->data();
    shape = base->shape;
    base_data_ = {new float[shape[0]], shape};
    std::copy(data, data + shape[0], base_data_.data);
    // test
    // for (size_t i = 0; i < cdf_data_.shape[0]; i++)
    //  std::cerr << cdf_data_.data[i] << " ";
    // std::cerr << std::endl;
  }

public:
  std::string schedule = "cl";
  float init_value = 0.;
  float mutiple_T = 1.;
  bool first_time_set_T = true;
  // option
  bool IsDW = false;
  bool IsNorm = false;
  bool Israrity = false;

  float d_ratio = 0.5;

  Competence(Ptr<Options> options, Ptr<Vocab> srcVocab, bool IsModTraining = false)
      : options_(options), srcVocab_(srcVocab) {
    std::vector<std::string> vals = options_->get<std::vector<std::string>>("sr-freq-file");

    std::string SR_fname = vals[0];
    std::string CDF_fname = vals[1];
    if(vals.size() > 2)
      T = std::atof(vals[2].c_str());
    if(vals.size() > 3)
      c0 = std::atof(vals[3].c_str());
    if(vals.size() > 4)
      schedule = vals[4].c_str();
    if(vals.size() > 5) {
      // mod options
      std::string mod_option = vals[5].c_str();
      for (size_t i=0; i < mod_option.size(); i++){
        if (mod_option[i] == 'd'){
          IsDW = true;
          d_ratio = std::atof(vals[6].c_str());
        }
        if (mod_option[i] == 'n')
          IsNorm = true;
        if (mod_option[i] == 'r')
          Israrity = true;
      }
    }
    //if(vals.size() > 6){
    //  init_value = std::atof(vals[6].c_str());
    //  set_T(init_value);
    //}

    LOG(info,
        "[CL] Loading sentence rarity file as: {}, cdf file: {}, T: {}, C0: {}, schedule: {}",
        SR_fname,
        CDF_fname,
        T,
        c0,
        schedule);

    if (schedule == "mod"){
      mutiple_T = T;
      T = 50000;
    }

    load(SR_fname);
    cdf_load(CDF_fname);
  }

  void set_T(float T_value){
    if (first_time_set_T){
      T = (mutiple_T - 1) * T_value;
      LOG(info, "[CL] Set T:{}, mutiple: {}", T, mutiple_T);
      first_time_set_T = false;
    }
  }

  float cal_sentence_rarity(std::vector<Word> sent) {
    float sum = 0.0;
    for(Word w : sent) {
      float p = data_[w];
      if(schedule == "cl" || Israrity) {
        // rarity
        float p_log = std::log(p);
        if(p)
          sum += p_log;
      } else {
        // MOD
        float p_log = p;
        if(p)
          sum += p_log;
      }
      // std::cerr << p << " " << p_log << std::endl;
    }
    // rarity
    if(schedule == "cl" || Israrity)
      sum = -1 * sum;
    if (IsNorm)
      // MOD=>with normalize
      sum = sum / (float)sent.size();
    // FIND HISTOGRAM
    size_t HISTORGRAM_SIZE = base_data_.shape[0];
    size_t index = HISTORGRAM_SIZE - 2;
    for(size_t i = 0; i < HISTORGRAM_SIZE - 1; i++) {
      // std::cerr << "sum:" << sum << " " << base_data_.data[i] << std::endl;
      if(sum < base_data_.data[i]) {
        index = i;
        break;
      }
    }
    // index--;
    auto cdf_score = cdf_data_.data[index];
    // std::cerr << "GOT INDEX:" << index << " " << cdf_score << std::endl;
    return cdf_score;
  }

  template <class T>
  const T& min(const T& a, const T& b) {
    return !(b < a) ? a : b;  // or: return !comp(b,a)?a:b; for version (2)
  }

  float cal_compentence(unsigned int t) {
    float tmp = t * ((1 - c0 * c0) / T) + c0 * c0;
    float c_sqrt = std::pow(tmp, 0.5);
    return min(float(1.0), c_sqrt);
  }

  void debug() {
    c0 = 0.01;
    for(unsigned int i = 1; i < 500; i++)
      std::cerr << cal_compentence(i) << " ";
    std::cerr << std::endl;
  }
};
}
};
