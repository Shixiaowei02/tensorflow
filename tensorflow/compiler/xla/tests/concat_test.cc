/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

using ConcatTest = ClientLibraryTestBase;
using ConcatTestHlo = HloTestBase;
using ::testing::HasSubstr;

XLA_TEST_F(ConcatTest, Concat_1x1024_With_1x1024_InDim0) {
  Array2D<float> lhs(1, 1024);
  Array2D<float> rhs(1, 1024);
  for (int i = 0; i < 1024; ++i) {
    lhs(0, i) = i;
    rhs(0, i) = i + 1024;
  }

  XlaBuilder builder(TestName());
  auto a = ConstantR2FromArray2D<float>(&builder, lhs);
  auto b = ConstantR2FromArray2D<float>(&builder, rhs);
  ConcatInDim(&builder, {a, b}, 0);

  Array2D<float> expected(2, 1024);
  for (int i = 0; i < 1024; ++i) {
    expected(0, i) = i;
    expected(1, i) = i + 1024;
  }
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla
