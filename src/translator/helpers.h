/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "graph/expression_graph.h"

namespace marian {

namespace cpu {

void suppressWord(Expr logProbs, Word id);
}

namespace gpu {

void suppressWord(Expr logProbs, Word id);

void SetColumnId(Tensor in_, size_t col, float value);
}

void suppressWord(Expr logProbs, Word id);

void suppressWordSent(Expr logProbs, Word id, std::vector<size_t> sent_ids);


}  // namespace marian
