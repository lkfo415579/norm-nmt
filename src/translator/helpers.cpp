/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <limits>

#include "data/types.h"
#include "tensors/tensor.h"
#include "translator/helpers.h"

namespace marian {

namespace cpu {

void SetColumn(Tensor in_, size_t col, float value) {
  int nRows = in_->shape().elements() / in_->shape()[-1];
  int nColumns = in_->shape()[-1];

  float* in = in_->data();
  for(int rowNumber = 0; rowNumber < nRows; ++rowNumber) {
    auto index = col + rowNumber * nColumns;
    in[index] = value;
  }
}

void suppressWord(Expr logProbs, Word id) {
  SetColumn(logProbs->val(), id, std::numeric_limits<float>::lowest());
}

void SetColumnId(Tensor in_, size_t col, float value, size_t sent_id) {
  int nColumns = in_->shape()[-1];

  float* in = in_->data();
  auto index = col + sent_id * nColumns;
  in[index] = value;
}

}  // namespace cpu

void suppressWord(Expr logProbs, Word id) {
  if(logProbs->val()->getBackend()->getDeviceId().type == DeviceType::cpu) {
    cpu::suppressWord(logProbs, id);
  }
#ifdef CUDA_FOUND
  else {
    gpu::suppressWord(logProbs, id);
  }
#endif
}
// force decoding
void suppressWordSent(Expr logProbs, Word id, std::vector<size_t> sent_ids) {
  gpu::SetColumnId(logProbs->val(), id, std::numeric_limits<float>::lowest());
  //cpu::SetColumnId(logProbs->val(), id, std::numeric_limits<float>::lowest(), sent_id);
}

}  // namespace marian
