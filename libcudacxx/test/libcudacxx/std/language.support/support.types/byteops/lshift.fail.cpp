//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <test_macros.h>

// template <class IntegerType>
//    constexpr byte operator <<(byte b, IntegerType shift) noexcept;
// These functions shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.

int main(int, char**)
{
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b2 = b1 << 2.0f;

  return 0;
}
