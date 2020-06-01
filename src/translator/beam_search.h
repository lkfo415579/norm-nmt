#pragma once
#include <algorithm>

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"

#include "translator/helpers.h"
#include "translator/nth_element.h"

namespace marian {

class BeamSearch {
private:
  Ptr<Options> options_;
  std::vector<Ptr<Scorer>> scorers_;
  size_t beamSize_;
  Word trgEosId_ = (Word)-1;
  Word trgUnkId_ = (Word)-1;
  bool force_decoding_ = false;
  std::vector<size_t> leng_;

public:
  BeamSearch(Ptr<Options> options,
             const std::vector<Ptr<Scorer>>& scorers,
             Word trgEosId,
             Word trgUnkId = -1)
      : options_(options),
        scorers_(scorers),
        beamSize_(options_->has("beam-size") ? options_->get<size_t>("beam-size") : 3),
        trgEosId_(trgEosId),
        trgUnkId_(trgUnkId),
        force_decoding_(options_->has("force-decoding") ? true : false) {}


  Beams toHyps(const std::vector<unsigned int> keys,
               const std::vector<float> pathScores,
               size_t vocabSize,
               const Beams& beams,
               std::vector<Ptr<ScorerState>>& states,
               size_t beamSize,
               bool first,
               Ptr<data::CorpusBatch> batch) {
    Beams newBeams(beams.size());

    std::vector<float> align;
    if(options_->has("alignment"))
      // Use alignments from the first scorer, even if ensemble
      align = scorers_[0]->getAlignment();

    for(size_t i = 0; i < keys.size(); ++i) {
      // Keys contains indices to vocab items in the entire beam.
      // Values can be between 0 and beamSize * vocabSize.
      Word embIdx = (Word)(keys[i] % vocabSize);
      auto beamIdx = i / beamSize;

      // Retrieve short list for final softmax (based on words aligned
      // to source sentences). If short list has been set, map the indices
      // in the sub-selected vocabulary matrix back to their original positions.
      auto shortlist = scorers_[0]->getShortlist();
      if(shortlist)
        embIdx
            = shortlist->reverseMap(embIdx);  // @TODO: should reverseMap accept a size_t or a Word?

      if(newBeams[beamIdx].size() < beams[beamIdx].size()) {
        auto& beam = beams[beamIdx];
        auto& newBeam = newBeams[beamIdx];

        auto hypIdx = (IndexType)(keys[i] / vocabSize);
        float pathScore = pathScores[i];

        auto hypIdxTrans = IndexType((hypIdx / beamSize) + (hypIdx % beamSize) * beams.size());
        if(first)
          hypIdxTrans = hypIdx;

        size_t beamHypIdx = hypIdx % beamSize;
        if(beamHypIdx >= (int)beam.size())
          beamHypIdx = beamHypIdx % beam.size();

        if(first)
          beamHypIdx = 0;

        auto hyp = New<Hypothesis>(beam[beamHypIdx], embIdx, hypIdxTrans, pathScore);

        // Set score breakdown for n-best lists
        if(options_->get<bool>("n-best")) {
          std::vector<float> breakDown(states.size(), 0);
          beam[beamHypIdx]->GetScoreBreakdown().resize(states.size(), 0);
          for(size_t j = 0; j < states.size(); ++j) {
            size_t key = embIdx + hypIdxTrans * vocabSize;
            breakDown[j] = states[j]->breakDown(key) + beam[beamHypIdx]->GetScoreBreakdown()[j];
          }
          hyp->GetScoreBreakdown() = breakDown;
        }

        // Set alignments
        if(!align.empty()) {
          hyp->SetAlignment(
              getAlignmentsForHypothesis(align, batch, (int)beamHypIdx, (int)beamIdx));
        }

        newBeam.push_back(hyp);
      }
    }
    return newBeams;
  }

  int findConsecutive(Words cur_str, Words s) {
    size_t step = cur_str.size();
    s = Words(s.begin() + step, s.end());
    size_t count = 0;
    for (int j = 0; j < s.size(); j += step) {
      Words t = Words(s.begin() + j, s.begin() + j + step);
      if (cur_str == t)
        count++;
      else
        break;
    }
    return count;
  }

  int checkRepeatWords(Beam& beam, History& his) {
    if(his.size() > 5) {
      // print out all top-hyp
      Ptr<Hypothesis> last = his.LastBeam()[0];
      Words s, cur_str;
      size_t count = 0;
      //
      while(last != nullptr) {
        s.push_back(last->GetWord());
        //std::cerr << last->GetWord() << ",";
        last = last->GetPrevHyp();
      }
      //std::cerr << '\n';
      // check consecutive phrase
      std::vector<std::tuple<size_t, Words>> res;
      for(int j = 0; j < s.size(); ++j) {
        cur_str.push_back(s[j]);
        // only search for half of the sent
        if (cur_str.size() > s.size() - cur_str.size())
          break;
        count = findConsecutive(cur_str, s);
        if(count){
          // larger than three words phrase
          if (cur_str.size() > 2){
            res.emplace_back(std::make_tuple(count, cur_str));
            break;
          }
        }
      }
      // retrieved repeat pattern
      if (!res.empty()){
        // suppress the start word of the pattern.
        size_t t;
        Words pattern;
        std::tie(t, pattern) = res.front();
        // LOG
        std::ostringstream oss;
        std::copy(pattern.begin(), pattern.end(), std::ostream_iterator<int>(oss, ","));
        std::ostringstream oss2;
        std::copy(s.begin(), s.end(), std::ostream_iterator<int>(oss2, ","));
        LOG(info, "[Repeat-Detector] count:{}, Words:{}, Re-Words:{}", t, oss2.str(), oss.str());
        // Remove Last repeat pattern
        his.Remove(pattern.size());
        beam = his.LastBeam();
        //
        return pattern.back();
      }
    }
    return 0;
  }

  std::vector<float> getAlignmentsForHypothesis(const std::vector<float> alignAll,
                                                Ptr<data::CorpusBatch> batch,
                                                int beamHypIdx,
                                                int beamIdx) {
    // Let's B be the beam size, N be the number of batched sentences,
    // and L the number of words in the longest sentence in the batch.
    // The alignment vector:
    //
    // if(first)
    //   * has length of N x L if it's the first beam
    //   * stores elements in the following order:
    //     beam1 = [word1-batch1, word1-batch2, ..., word2-batch1, ...]
    // else
    //   * has length of N x L x B
    //   * stores elements in the following order:
    //     beams = [beam1, beam2, ..., beam_n]
    //
    // The mask vector is always of length N x L and has 1/0s stored like
    // in a single beam, i.e.:
    //   * [word1-batch1, word1-batch2, ..., word2-batch1, ...]
    //
    size_t batchSize = batch->size();
    size_t batchWidth = batch->width() * batchSize;
    std::vector<float> align;

    for(size_t w = 0; w < batchWidth / batchSize; ++w) {
      size_t a = ((batchWidth * beamHypIdx) + beamIdx) + (batchSize * w);
      size_t m = a % batchWidth;
      if(batch->front()->mask()[m] != 0)
        align.emplace_back(alignAll[a]);
    }

    return align;
  }

  Beams pruneBeam(const Beams& beams) {
    Beams newBeams;
    for(auto beam : beams) {
      Beam newBeam;
      for(auto hyp : beam) {
        if(hyp->GetWord() != trgEosId_) {
          newBeam.push_back(hyp);
        }
      }
      newBeams.push_back(newBeam);
    }
    return newBeams;
  }

  // main decoding function
  Histories search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    int dimBatch = (int)batch->size();

    Histories histories;
    for(int i = 0; i < dimBatch; ++i) {
      size_t sentId = batch->getSentenceIds()[i];
      auto history = New<History>(
          sentId, options_->get<float>("normalize"), options_->get<float>("word-penalty"));
      histories.push_back(history);
    }

    size_t localBeamSize = beamSize_;  // max over beam sizes of active sentence hypotheses

    auto getNBestList = createGetNBestListFn(localBeamSize, dimBatch, graph->getDeviceId());

    Beams beams(dimBatch);  // [batchIndex][beamIndex] is one sentence hypothesis
    for(auto& beam : beams)
      beam.resize(localBeamSize, New<Hypothesis>());

    bool first = true;
    bool final = false;

    for(int i = 0; i < dimBatch; ++i)
      histories[i]->Add(beams[i], trgEosId_);

    std::vector<Ptr<ScorerState>> states;

    for(auto scorer : scorers_) {
      scorer->clear(graph);
    }

    for(auto scorer : scorers_) {
      states.push_back(scorer->startState(graph, batch));
    }

    // main loop over output tokens
    Expr prevPathScores;
    //size_t repeat_ID = 0;
    do {
      //**********************************************************************
      // create constant containing previous path scores for current beam
      // also create mapping of hyp indices, which are not 1:1 if sentences complete
      std::vector<IndexType> hypIndices;  // [beamIndex * activeBatchSize + batchIndex]
                                          // backpointers, concatenated over beam positions. Used
                                          // for reordering hypotheses
      std::vector<IndexType> embIndices;
      // Expr prevPathScores; // [beam, 1, 1, 1]
      if(first) {
        // no scores yet
        prevPathScores = graph->constant({1, 1, 1, 1}, inits::from_value(0));
      } else {
        std::vector<float> beamScores;

        dimBatch = (int)batch->size();

        for(size_t i = 0; i < localBeamSize; ++i) {
          for(size_t j = 0; j < beams.size(); ++j) {  // loop over batch entries (active sentences)
            auto& beam = beams[j];
            if(i < beam.size()) {
              auto hyp = beam[i];
              hypIndices.push_back((IndexType)hyp->GetPrevStateIndex());  // backpointer
              embIndices.push_back(hyp->GetWord());
              beamScores.push_back(hyp->GetPathScore());
            } else {  // dummy hypothesis
              hypIndices.push_back(0);
              embIndices.push_back(0);  // (unused)
              beamScores.push_back(-9999);
            }
          }
        }

        prevPathScores
            = graph->constant({(int)localBeamSize, 1, dimBatch, 1}, inits::from_vector(beamScores));
      }

      //**********************************************************************
      // prepare scores for beam search
      auto pathScores = prevPathScores;

      for(size_t i = 0; i < scorers_.size(); ++i) {
        states[i] = scorers_[i]->step(
            graph, states[i], hypIndices, embIndices, dimBatch, (int)localBeamSize);

        if(scorers_[i]->getWeight() != 1.f)
          pathScores = pathScores + scorers_[i]->getWeight() * states[i]->getLogProbs();
        else
          pathScores = pathScores + states[i]->getLogProbs();
      }


      // make beams continuous
      if(dimBatch > 1 && localBeamSize > 1)
        pathScores = transpose(pathScores, {2, 1, 0, 3});

      if(first)
        graph->forward();
      else {
        try {
          graph->forwardNext();
        } catch(const std::exception& e) {
          // exceeds maximum reserved MB
          std::cerr << e.what() << "\n";
          for(int i = 0; i < dimBatch; ++i)
            if(!beams[i].empty())
              histories[i]->Add(beams[i], trgEosId_, true, true);
          break;
        }
      }

      //**********************************************************************
      // suppress specific symbols if not at right positions
      if(trgUnkId_ != -1 && options_->has("allow-unk") && !options_->get<bool>("allow-unk"))
        suppressWord(pathScores, trgUnkId_);
      for(auto state : states)
        state->blacklist(pathScores, batch);
      //**********************************************************************
      // force decoindg if len(y') is less than len(y*) produce EOS.
      if (force_decoding_){
        //for(int i = 0; i < dimBatch; ++i) {
        //  if(!beams[i].empty()) {
        //    if (histories[i]->size() < leng_[i]){
        //      //auto len_node = graph->constant({leng_.size()}, inits::from_vector(beamScores));
        //      suppressWordSent(pathScores, trgEosId_, leng_);
        //    }
        //  }else{
        //    leng_.erase(leng_.begin() + i);
        //  }
        //}
        suppressWord(pathScores, trgEosId_);
      }
      //**********************************************************************
      // check if it produces repeat words..
      //if (repeat_ID)
      //  suppressWord(pathScores, repeat_ID);

      //**********************************************************************
      // perform beam search and pruning
      std::vector<unsigned int> outKeys;
      std::vector<float> outPathScores;

      std::vector<size_t> beamSizes(dimBatch, localBeamSize);
      getNBestList(beamSizes, pathScores->val(), outPathScores, outKeys, first);

      int dimTrgVoc = pathScores->shape()[-1];
      beams = toHyps(outKeys, outPathScores, dimTrgVoc, beams, states, localBeamSize, first, batch);

      auto prunedBeams = pruneBeam(beams);
      for(int i = 0; i < dimBatch; ++i) {
        if(!beams[i].empty()) {
          final = final
                  || histories[i]->size() >= options_->get<float>("max-length-factor")
                                                 * batch->front()->batchWidth();
          // force-decoding
          if (force_decoding_)
            if (histories[i]->size() >= leng_[i] - 1)
              // prunedBeam that has been final
              prunedBeams[i] = {};
          // force-decoding
          histories[i]->Add(beams[i], trgEosId_, prunedBeams[i].empty() || final);
        }
      }
      beams = prunedBeams;

      //repeat_ID = checkRepeatWords(beams[0], *histories[0]);

      // determine beam size for next sentence, as max over still-active sentences
      if(!first) {
        size_t maxBeam = 0;
        for(auto& beam : beams)
          if(beam.size() > maxBeam)
            maxBeam = beam.size();
        localBeamSize = maxBeam;
      }
      first = false;

    } while(localBeamSize != 0 && !final);  // end of main loop over output tokens

    return histories;
  }

  void SetTargetLen(std::vector<size_t> &leng){
    leng_ = leng;
  }
};
}  // namespace marian
