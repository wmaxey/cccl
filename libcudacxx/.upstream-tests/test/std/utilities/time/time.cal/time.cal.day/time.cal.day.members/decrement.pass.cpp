//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class day;

//  constexpr day& operator--() noexcept;
//  constexpr day operator--(int) noexcept;


#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename D>
__host__ __device__
constexpr bool testConstexpr()
{
    D d1{10};
    if (static_cast<unsigned>(--d1) != 9) return false;
    if (static_cast<unsigned>(d1--) != 9) return false;
    if (static_cast<unsigned>(d1)   != 8) return false;
    return true;
}

int main(int, char**)
{
    using day = cuda::std::chrono::day;
    ASSERT_NOEXCEPT(--(cuda::std::declval<day&>())  );
    ASSERT_NOEXCEPT(  (cuda::std::declval<day&>())--);

    ASSERT_SAME_TYPE(day , decltype(  cuda::std::declval<day&>()--));
    ASSERT_SAME_TYPE(day&, decltype(--cuda::std::declval<day&>()  ));

    static_assert(testConstexpr<day>(), "");

    for (unsigned i = 10; i <= 20; ++i)
    {
        day day(i);
        assert(static_cast<unsigned>(--day) == i - 1);
        assert(static_cast<unsigned>(day--) == i - 1);
        assert(static_cast<unsigned>(day)   == i - 2);
    }

  return 0;
}
