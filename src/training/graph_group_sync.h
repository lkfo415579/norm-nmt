#pragma once

#include "training/graph_group.h"
#include "training/communicator.h"
#include "training/exponential_smoothing.h"

namespace marian {

class SyncGraphGroup : public GraphGroup, public ExponentialSmoothing {
  const size_t delay_{ 1 }; // optimizer-delay parameter

  Ptr<ICommunicator> comm_; // [not null] communicator, e.g. NCCLCommunicator
  Ptr<IMPIWrapper> mpi_;    // [not null] all MPI-like communication goes through this (this is a dummy implementation if no MPI run)

  std::vector<DeviceId> devices_;                  // [deviceIndex]
  std::vector<Ptr<models::ModelBase>> builders_;   // [deviceIndex]
  std::vector<Ptr<ExpressionGraph>> graphs_;       // [deviceIndex]

  std::vector<Ptr<OptimizerBase>> shardOpt_;       // [deviceIndex]

  std::vector<Tensor> paramsAvg_;                  // [deviceIndex] exponentially smoothed parameters, sharded
  // @TODO: instead, create an array of ExponentialSmoothing objects, and don't use ExponentialSmoothing as a base class
  std::vector<Ptr<TensorAllocator>> paramsAllocs_; // [deviceIndex] we must hold a reference to the memory until this class dies
  // @TODO: move this nto ExponentialSmoothing, together with paramsAvg_?

  bool first_{ true }; // gets interpreted and cleared by update()

  void initialize(const Ptr<data::Batch>& exampleBatch);
  void initializeAvg();

  bool isMainProcess() const { return mpi_->myMPIRank() == 0; } // (we need this test a few times)
  void barrier() const { mpi_->barrier(); } // (we need this several times)
  void swapParamsAvg() { if (mvAvg_ && paramsAvg_.size() > 0) comm_->swapParams(paramsAvg_); } // note: must call this on all MPI ranks in parallel

  // update cycle
  size_t update_cycle_ = 2;
  size_t iter_ = 0;
  std::vector<Tensor> gradient_acc_;
  // mod learning
  size_t step = 0;

  void lazyInit() {
    if(gradient_acc_.size() == 0) {
      int totalSize = (int)graphs_[0]->params()->vals()->size();
      int shardSize = (int)ceil(totalSize / (float)graphs_.size());

      int pos = 0;
      for(auto graph : graphs_) {
        int __size__ = std::min(shardSize, totalSize);
        //int __size__ = totalSize;

        auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
        paramsAllocs_.push_back(paramsAlloc);

        paramsAlloc->reserveExact(__size__ * sizeof(float));

        Tensor tmp;

        paramsAlloc->allocate(tmp, {1, __size__});
        gradient_acc_.push_back(tmp);

        // move to next shard
        pos += __size__;
        totalSize -= __size__;
      }
    }
  }

public:
  SyncGraphGroup(Ptr<Options> config);

  void setScheduler(Ptr<Scheduler> scheduler) override;

  void update(Ptr<data::Batch> batch) override;

  void load() override;
  void save(bool final = false) override;

  Ptr<data::BatchStats> collectStats();

  std::vector<Ptr<ExpressionGraph>> collectGraphs(){
    return graphs_;
  }

  void calculate_mod(){
    if (iter_)
      return;
    int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[0];
    int dimEmb = options_->get<int>("dim-emb");

    //auto enc_emb = graphs_[0]->param("encoder_Wemb", {dimSrcVoc, dimEmb}, inits::glorot_uniform);
    auto enc_emb = graphs_[0]->get("encoder_Wemb");
    std::vector<float> values;
    if (!enc_emb->val())
      graphs_[0]->forward();
    enc_emb->val()->get(values);
    auto graph = New<ExpressionGraph>();
    graph->setDevice(graphs_[0]->getDeviceId());
    auto new_enc_emb = graph->param("encoder_Wemb", {dimSrcVoc, dimEmb}, inits::from_vector(values));
    auto enc_mod = square(new_enc_emb);
    enc_mod = sum(enc_mod, -1);
    enc_mod = sqrt(enc_mod, 0);
    enc_mod = sum(enc_mod, 0);
    graph->forward();
    values.clear();
    enc_mod->val()->get(values);
    LOG(info, "[MOD-CL] step: {}, mod_value: {}", step, values[0]);
    graphs_[0]->mod_values.push_back(values[0]);
  }
  // @TODO: consider to make this a virtual as well? Currently it is a template dispatch
};
}  // namespace marian
