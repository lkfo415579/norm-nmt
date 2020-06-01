//
// Created by revo on 8/2/19.
//

#pragma once

#include "graph/expression_graph.h"
#include "marian.h"

namespace marian {
namespace data {

class DataTrainingBase {
protected:
    typedef typename CorpusBase::Sample Sample;
    typedef std::vector<Sample> Samples;  // @TODO: type names should be capitalized

    Ptr<Options> options_;
    std::vector<Ptr<Vocab>> vocabs_;
public:
    DataTrainingBase(Ptr<Options> options, std::vector<Ptr<Vocab>> &vocabs) : options_(options), vocabs_(vocabs) {};

    virtual void run(Samples &batch, size_t e) = 0;

    virtual float get_mod_from_emb(size_t step) { return 0.0f;};

    virtual float get_mod_from_graph(size_t step) { return 0.0f;};

    virtual float get_init_value() { return 0.0f;};
};

    Ptr<DataTrainingBase> NewGapTraining(Ptr<Options> options, std::vector<Ptr<Vocab>> vocabs, const std::vector<Ptr<ExpressionGraph>> &graphs);

    Ptr<DataTrainingBase> NewModTraining(Ptr<Options> options, std::vector<Ptr<Vocab>> vocabs, const std::vector<Ptr<ExpressionGraph>> &graphs);
}
}  // namespace marian