//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/optional>

// constexpr optional(const T& v);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Z
{
  Z(int) {}
  Z(const Z&)
  {
    TEST_THROW(6);
  }
};

void test_exceptions()
{
  typedef Z T;
  try
  {
    const T t(3);
    optional<T> opt(t);
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  {
    typedef int T;
    constexpr T t(5);
    constexpr optional<T> opt(t);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == 5, "");

    struct test_constexpr_ctor : public optional<T>
    {
      __host__ __device__ constexpr test_constexpr_ctor(const T&) {}
    };
  }
  {
    typedef double T;
    constexpr T t(3);
    constexpr optional<T> opt(t);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == 3, "");

    struct test_constexpr_ctor : public optional<T>
    {
      __host__ __device__ constexpr test_constexpr_ctor(const T&) {}
    };
  }

  {
    const int x = 42;
    optional<const int> o(x);
    assert(*o == x);
  }
  {
    typedef TestTypes::TestType T;
    T::reset();
    const T t(3);
    optional<T> opt = t;
    assert(T::alive() == 2);
    assert(T::copy_constructed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
  {
    typedef ExplicitTestTypes::TestType T;
    static_assert(!cuda::std::is_convertible<T const&, optional<T>>::value, "");
    T::reset();
    const T t(3);
    optional<T> opt(t);
    assert(T::alive() == 2);
    assert(T::copy_constructed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }

  {
    typedef ConstexprTestTypes::TestType T;
    constexpr T t(3);
    constexpr optional<T> opt = {t};
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(opt.value().value == 3, "");

    struct test_constexpr_ctor : public optional<T>
    {
      __host__ __device__ constexpr test_constexpr_ctor(const T&) {}
    };
  }
  {
    typedef ExplicitConstexprTestTypes::TestType T;
    static_assert(!cuda::std::is_convertible<const T&, optional<T>>::value, "");
    constexpr T t(3);
    constexpr optional<T> opt(t);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(opt.value().value == 3, "");

    struct test_constexpr_ctor : public optional<T>
    {
      __host__ __device__ constexpr test_constexpr_ctor(const T&) {}
    };
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
