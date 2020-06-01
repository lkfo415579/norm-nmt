//#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;


void Multiple_forward(){
  Config::seed = 1234;
  auto device = DeviceType::gpu;

  auto graph = New<ExpressionGraph>();
  graph->setDevice({0, device});
  graph->reserveWorkspaceMB(16);

  std::vector<float> values;
  std::vector<float> grad_values;

  //SECTION("scalar multiplication")
  graph->clear();
  values.clear();
  std::vector<float> vB({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  std::vector<float> vC({10, 20, 30});
  //for (int i = 0; i < values.size(); ++i) {
  //  std::cerr << values[i] << ",";
  //}
  //std::cerr << '\n';
}

void tests(DeviceType device) {
  //auto floatApprox = [](float x, float y) { return x == Approx(y); };
  Config::seed = 1234;

  auto graph = New<ExpressionGraph>();
  graph->setDevice({0, device});
  graph->reserveWorkspaceMB(16);

  std::vector<float> values;
  std::vector<float> grad_values;

  graph->clear();
  values.clear();
  std::vector<float> vB({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  //std::vector<float> vB({1, 2, 3, 4, 5, 6});
  std::vector<float> vC({10, 20});
  std::vector<uint32_t > labels({1, 0});


  auto B = graph->param("B", {2, 4}, inits::glorot_uniform);
  auto C = graph->param("C", {2, 1}, inits::from_vector(vC));
  C->setTrainable(false);
  auto B2 = B * C;
  //
  Expr relu_output = relu(B2);
  relu_output->debug("Relu");
  // error
  auto idx = graph->constant({2}, inits::from_vector(labels), Type::uint32);
  auto ce = cross_entropy(relu_output, idx);
  //auto cost = mean(sum(ce, /*axis=*/2), /*axis=*/0);
  ce->debug("CE");
  //
  graph->forward();
  graph->backward();

  B2->val()->get(values);
  relu_output->grad()->get(grad_values);

}


int main(int argc, char** argv) {
  tests(DeviceType::cpu);
}