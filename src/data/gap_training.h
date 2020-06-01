//
// Created by revo on 08/02/2019.
//
#pragma once

#include <math.h>
#include <string>
#include "common/config.h"
#include "common/options.h"

#include <training/validator.h>
#include "tensors/tensor_operators.h"
#include "data/text_input.h"
#include "vocab.h"
//
#include "revo_stub.h"

namespace marian {
namespace data {

class GapTraining : public DataTrainingBase {
private:
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<Ptr<Scorer>> scorers_;
  // for toBatch function
  Corpus data_;
  // hyp-prarameter
  float u_ = 12;
  size_t b_ = 3;
  // random
  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());
  std::uniform_real_distribution<> dis{0.0, 1.0};
  // auto randfun = std::bind(dis, gen);
  //
  std::mutex mutex_;
  ThreadPool threadPool_;
  int batch_num_ = 0;

  float decay(float epoch) { return u_ / (u_ + expf(epoch / u_)); }

  // Update document-wide sufficient statistics for BLEU with single sentence n-gram stats.
  template <typename T>
  void updateStats(std::vector<float>& stats,
                   const std::vector<T>& cand,
                   const std::vector<T>& ref) {
    std::map<std::vector<T>, size_t> rgrams;
    for(size_t i = 0; i < ref.size(); ++i) {
      // template deduction for std::min<T> seems to be weird under VS due to
      // macros in windows.h hence explicit type to avoid macro parsing.
      for(size_t l = 1; l <= std::min<size_t>(4ul, ref.size() - i); ++l) {
        std::vector<T> ngram(l);
        std::copy(ref.begin() + i, ref.begin() + i + l, ngram.begin());
        rgrams[ngram]++;
      }
    }

    std::map<std::vector<T>, size_t> tgrams;
    for(size_t i = 0; i < cand.size() - 1; ++i) {
      for(size_t l = 1; l <= std::min<size_t>(4ul, cand.size() - 1 - i); ++l) {
        std::vector<T> ngram(l);
        std::copy(cand.begin() + i, cand.begin() + i + l, ngram.begin());
        tgrams[ngram]++;
      }
    }

    for(auto& ngramcount : tgrams) {
      size_t l = ngramcount.first.size();
      size_t tc = ngramcount.second;
      size_t rc = rgrams[ngramcount.first];

      stats[2 * l - 2] += std::min<size_t>(tc, rc);
      stats[2 * l - 1] += tc;
    }

    stats[8] += ref.size();
  }

  // Extract matching target reference from batch and pass on to update BLEU stats
  void updateStats(std::vector<float>& stats,
                   const Words& cand,
                   const Ptr<data::Batch> batch,
                   size_t no,
                   Word eos) {
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
    auto subBatch = corpusBatch->back();

    size_t size = subBatch->batchSize();
    size_t width = subBatch->batchWidth();

    Words ref;  // fill ref
    for(size_t i = 0; i < width; ++i) {
      Word w = subBatch->data()[i * size + no];
      if(w == eos)
        break;
      ref.push_back(w);
    }

    updateStats(stats, cand, ref);
  }

  float calcBLEU(const std::vector<float>& stats) {
    float logbleu = 0;
    for(int i = 0; i < 8; i += 2) {
      if(stats[i] == 0.f)
        return 0.f;
      logbleu += std::log(stats[i] / stats[i + 1]);
    }

    logbleu /= 4.f;

    float brev_penalty = 1.f - std::max(stats[8] / stats[1], 1.f);
    return std::exp(logbleu + brev_penalty) * 100;
  }

  void gap_sampling(Samples& batch,
                    const std::vector<float>& BLEUS,
                    const std::vector<Words>& BLEUS_WORDS,
                    size_t e = 0) {
    for(int j = 0; j < batch.size(); ++j) {
      auto tgt = &batch[j].back();
      size_t best_id = j * b_;
      float bleu_t = 0;
      for(int k = j * b_; k < j * b_ + b_; ++k)
        if(BLEUS[k] > bleu_t) {
          bleu_t = BLEUS[k];
          best_id = k;
        }
      Words mt_tgt = BLEUS_WORDS[best_id];
      // real sampling
      for(int l = 0; l < mt_tgt.size(); ++l) {
        float r = dis(gen);
        if(r > decay(e))
          tgt->operator[](l) = mt_tgt[l];
      }
      //batch[j].back() = tgt;
    }
  }

  void debug(Words& s, std::string msg) {
    std::cerr << msg;
    // for (auto &word: s)
    //  std::cerr << word << ",";
    std::cerr << vocabs_[1]->decode(s);
    std::cerr << '\n';
  }

  // void resort_his(Samples& batch, std::vector<Ptr<History>>& all_his){
  //  size_t num_split = batch.size() / graphs_.size();
  //  std::vector<Ptr<History>> new_his;
  //  for (int j = 0; j < batch.size(); j += num_split) {
  //    size_t sent_id = batch[j].getId();
  //    for (int k = 0; k < all_his.size(); k += num_split)
  //      if (all_his[k]->GetLineNum() == sent_id)
  //        for (int l = 0; l < num_split; ++l)
  //          new_his.emplace_back(all_his[k + l]);
  //  }
  //  all_his = new_his;
  //}

  void resort_his(std::vector<std::tuple<size_t, Histories>>& all_his, Histories& histories) {
    for(int j = 0; j < all_his.size(); ++j) {
      for(auto& one : all_his) {
        size_t id;
        Histories his;
        std::tie(id, his) = one;
        if(id == j) {
          for(int k = 0; k < his.size(); ++k)
            histories.emplace_back(his[k]);
          break;
        }
      }
    }
  }

public:
  GapTraining(Ptr<Options> options,
              std::vector<Ptr<Vocab>> vocabs,
              const std::vector<Ptr<ExpressionGraph>>& graphs)
      : DataTrainingBase(options, vocabs), graphs_(graphs), data_(options) {
    std::vector<std::string> vals = options_->get<std::vector<std::string>>("gap-training");
    if(vals.size() > 0)
      u_ = std::atoi(vals[0].c_str());
    if(vals.size() > 1)
      b_ = std::atoi(vals[1].c_str());
    LOG(info, "[Gap-Training] u:{}, beam_size:{}", u_, b_);
    // refresh options for beam size
    auto opt = New<Options>();
    opt->merge(options_);
    opt->set("beam-size", b_);
    opt->set("inference", true);
    opt->set("force-decoding", true);
    options_ = opt;
    // create scorer
    auto model = options_->get<std::string>("model");
    //auto scorers = createScorers(options_);
    for(size_t i=0; i < graphs_.size(); ++i) {
      auto builder = models::from_options(options_, models::usage::translation);
      Ptr<Scorer> scorer = New<ScorerWrapper>(builder, "", 1.0f, model);
      // set to inference
      //graphs_[i]->setInference(true);
      //scorer->init(graphs_[i]);
      scorers_.push_back(scorer);
    }
    threadPool_.reserve(graphs.size());
  }

  void run(Samples& batch, size_t e) override {
    // force-decoding setup
    auto realBatch = data_.toBatch(batch);
    size_t MAX_BATCH_SIZE = 800 / realBatch->widthTrg();
    //if (realBatch->widthTrg() > 130)
    //  MAX_BATCH_SIZE = 3;
    size_t num_batch = batch.size() / MAX_BATCH_SIZE;
    size_t single_batch_size = MAX_BATCH_SIZE;
    // split into batches
    auto batches = realBatch->split2Corpus(num_batch);
    //for(auto& s : batches)
    //  std::cerr << s->size() << ",";
    // pre-create leng
    std::vector<std::vector<size_t>> batch_lengs;
    size_t runner = 0;
    for(int k = 0; k < batches.size(); k ++) {
      std::vector<size_t> leng;
      for(int j = 0; j < batches[k]->size(); ++j)
        leng.emplace_back(batch[runner++].back().size());
      batch_lengs.emplace_back(leng);
    }

    LOG(info,
        "[Gap-Training] BATCH_ID:{}, NUM OF SMALL-BATCHES:{}, SMALL-BATCH_SIZE:{}, Width:[{}, {}], epoch:{}",
        batch_num_++,
        batches.size(),
        single_batch_size,
        realBatch->width(),
        realBatch->widthTrg(),
        e);
    //return;
    // search
    for(auto graph : graphs_)
      graph->setInference(true);
    size_t sentenceId = 0;
    std::vector<std::tuple<size_t, Histories>> all_his;
    {
      //threadPool_.reserve(graphs_.size());
      //TaskBarrier taskBarrier;
      ThreadPool threadPool_2(graphs_.size(), graphs_.size());
      for(auto small_batch : batches) {
        auto task = [=, &all_his, &batch_lengs](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local Ptr<Scorer> scorer;
          if(!graph) {
            graph = graphs_[id % graphs_.size()];
            scorer = scorers_[id % graphs_.size()];
          }
          auto search = New<BeamSearch>(options_,
                                        std::vector<Ptr<Scorer>>{scorer},
                                        vocabs_[1]->getEosId(),
                                        vocabs_[1]->getUnkId());
          auto leng = batch_lengs[id];
          search->SetTargetLen(leng);
          auto histories = search->search(graph, small_batch);
          //std::lock_guard<std::mutex> statsLock(mutex_);
          all_his.emplace_back(std::make_tuple(id, histories));
        };
        threadPool_2.enqueue(task, sentenceId);
        //taskBarrier.push_back(threadPool_.enqueue(task, sentenceId));
        sentenceId++;
      }
    }
    // resort histories
    Histories histories;
    resort_his(all_his, histories);
    for(auto graph : graphs_)
      graph->setInference(false);
    // BLEU calcs
    std::vector<float> BLEUS;
    std::vector<Words> BLEUS_WORDS;
    size_t no = 0;
    for(auto& history : histories) {
      // each beam calculate BLEU
      auto nbest = history->NBest(b_);
      for(int j = 0; j < b_; ++j) {
        // early stopped
        float score = 0.0;
        if(j < nbest.size()) {
          auto result = nbest[j];
          std::vector<float> stats(9, 0.f);
          const auto& words = std::get<0>(result);
          updateStats(stats, words, realBatch, no, vocabs_.back()->getEosId());
          score = calcBLEU(stats);
          //
          BLEUS_WORDS.emplace_back(words);
        }else{
          Words tgt = batch[no].back();
          BLEUS_WORDS.emplace_back(tgt);
        }
        BLEUS.emplace_back(score);
      }
      no++;
    }
    //// sampling
    //// debug(batch[0].back(), "ORIIII: ");
    gap_sampling(batch, BLEUS, BLEUS_WORDS, e);
    //debug(batch[0].back(), "GAPPED: ");
  }

  //float get_init_value() override { return 0.;}

  //void get_mod_from_emb() override {}
};

class ModTraining : public DataTrainingBase {
private:
    std::vector<Ptr<ExpressionGraph>> graphs_;
    Ptr<Scorer> scorer_;
    Ptr<ExpressionGraph> graph;
    size_t runner_ = 0;
    float init_mod = 0.;
public:
    ModTraining(Ptr<Options> options,
                std::vector<Ptr<Vocab>> vocabs,
                const std::vector<Ptr<ExpressionGraph>>& graphs)
            : DataTrainingBase(options, vocabs), graphs_(graphs) {
      //std::string model = options_->get<std::string>("model");
      //auto builder = models::from_options(options_, models::usage::translation);
      //scorer_ = New<ScorerWrapper>(builder, "", 1.0f, model);
      //graph = New<ExpressionGraph>();
      //graph->setDevice({0, DeviceType::gpu});
      //graph->reserveWorkspaceMB(10);
      if (options_->has("mod_init_value") && options_->get<float>("mod_init_value")) {
        init_mod = options_->get<float>("mod_init_value");
        LOG(info, "[CL-MOD] initial_mod:{}", init_mod);
      }
    }

    float get_mod_from_graph(size_t step) override {
      // ?? WTF
      // 1. graph run all batches then come to here
      step -= 1;
      if (graphs_.empty() || graphs_[0]->params()->size() == 0)
        return 10.;
      float mode_value = graphs_[0]->mod_values[0];
      if (init_mod == 0.){
        init_mod = mode_value - 10;
        LOG(info,"[CL-MOD] step:{}, initial_mod:{}", step, init_mod);
      }else{
        auto vs = graphs_[0]->mod_values;
        //while (step > vs.size())
        //  sleep(1);
        mode_value = graphs_[0]->mod_values[step];
      }
      if (runner_ != step) {
        LOG(info, "[CL-MOD] step:{}, enc_emb_mod_avg:{}", step, mode_value);
        runner_ = step;
      }
      return mode_value - init_mod;
    }

    float get_mod_from_emb(size_t step) override {
      // graph is not been created, using mini value
      if (graphs_.empty() || graphs_[0]->params()->size() == 0)
        return 10.;
      int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[0];
      int dimEmb = options_->get<int>("dim-emb");

      //auto enc_emb = graphs_[0]->param("encoder_Wemb", {dimSrcVoc, dimEmb}, inits::glorot_uniform);
      auto enc_emb = graphs_[0]->get("encoder_Wemb");
      std::vector<float> values;
      if (!enc_emb->val())
        graphs_[0]->forward();
      enc_emb->val()->get(values);
      graph = New<ExpressionGraph>();
      graph->setDevice(graphs_[0]->getDeviceId());
      auto new_enc_emb = graph->param("encoder_Wemb" + std::to_string(runner_++), {dimSrcVoc, dimEmb}, inits::from_vector(values));
      auto enc_mod = square(new_enc_emb);
      enc_mod = sum(enc_mod, -1);
      enc_mod = sqrt(enc_mod, 0);
      enc_mod = sum(enc_mod, 0);
      //enc_mod->debug("enc_mod sqrt");
      graph->forward();
      //std::string str = enc_mod->graphviz();
      values.clear();
      enc_mod->val()->get(values);
      if (init_mod == 0.){
        init_mod = values[0] - 10;
        LOG(info,"[CL-MOD] step:{}, initial_mod:{}", step, init_mod);
      }
      LOG(info,"[CL-MOD] step:{}, enc_emb_mod_avg:{}", step, values[0]);
      return values[0] - init_mod;
    }

    float get_init_value() override {
      return init_mod;
    }

    void run(Samples& batch, size_t e) override{}
};

Ptr<DataTrainingBase> NewGapTraining(Ptr<Options> options,
                                     std::vector<Ptr<Vocab>> vocabs,
                                     const std::vector<Ptr<ExpressionGraph>>& graphs) {
  return New<GapTraining>(options, vocabs, graphs);
}

Ptr<DataTrainingBase> NewModTraining(Ptr<Options> options,
                                     std::vector<Ptr<Vocab>> vocabs,
                                     const std::vector<Ptr<ExpressionGraph>>& graphs) {
  return New<ModTraining>(options, vocabs, graphs);
}


}
};
