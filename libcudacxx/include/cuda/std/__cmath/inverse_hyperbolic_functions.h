// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_INVERSE_HYPERBOLIC_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_INVERSE_HYPERBOLIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/common.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// acosh

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float acosh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSHF)
  return _CCCL_BUILTIN_ACOSHF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSHF ^^^ / vvv !_CCCL_BUILTIN_ACOSHF vvv
  return ::acoshf(__x);
#endif // !_CCCL_BUILTIN_ACOSHF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float acoshf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSHF)
  return _CCCL_BUILTIN_ACOSHF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSHF ^^^ / vvv !_CCCL_BUILTIN_ACOSHF vvv
  return ::acoshf(__x);
#endif // !_CCCL_BUILTIN_ACOSHF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double acosh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSH)
  return _CCCL_BUILTIN_ACOSH(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSH ^^^ / vvv !_CCCL_BUILTIN_ACOSH vvv
  return ::acosh(__x);
#endif // !_CCCL_BUILTIN_ACOSH
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double acosh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSHL)
  return _CCCL_BUILTIN_ACOSHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSHL ^^^ / vvv !_CCCL_BUILTIN_ACOSHL vvv
  return ::acoshl(__x);
#  endif // !_CCCL_BUILTIN_ACOSHL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double acoshl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSHL)
  return _CCCL_BUILTIN_ACOSHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSHL ^^^ / vvv !_CCCL_BUILTIN_ACOSHL vvv
  return ::acoshl(__x);
#  endif // !_CCCL_BUILTIN_ACOSHL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half acosh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::acoshf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 acosh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::acoshf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double acosh(_Integer __x) noexcept
{
  return _CUDA_VSTD::acosh((double) __x);
}

// asinh

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float asinh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINHF)
  return _CCCL_BUILTIN_ASINHF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINHF ^^^ / vvv !_CCCL_BUILTIN_ASINHF vvv
  return ::asinhf(__x);
#endif // !_CCCL_BUILTIN_ASINHF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float asinhf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINHF)
  return _CCCL_BUILTIN_ASINHF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINHF ^^^ / vvv !_CCCL_BUILTIN_ASINHF vvv
  return ::asinhf(__x);
#endif // !_CCCL_BUILTIN_ASINHF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double asinh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINH)
  return _CCCL_BUILTIN_ASINH(__x);
#else // ^^^ _CCCL_BUILTIN_ASINH ^^^ / vvv !_CCCL_BUILTIN_ASINH vvv
  return ::asinh(__x);
#endif // !_CCCL_BUILTIN_ASINH
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double asinh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINHL)
  return _CCCL_BUILTIN_ASINHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINHL ^^^ / vvv !_CCCL_BUILTIN_ASINHL vvv
  return ::asinhl(__x);
#  endif // !_CCCL_BUILTIN_ASINHL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double asinhl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINHL)
  return _CCCL_BUILTIN_ASINHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINHL ^^^ / vvv !_CCCL_BUILTIN_ASINHL vvv
  return ::asinhl(__x);
#  endif // !_CCCL_BUILTIN_ASINHL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half asinh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::asinhf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 asinh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::asinhf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double asinh(_Integer __x) noexcept
{
  return _CUDA_VSTD::asinh((double) __x);
}

// atanh

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float atanh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANHF)
  return _CCCL_BUILTIN_ATANHF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANHF ^^^ / vvv !_CCCL_BUILTIN_ATANHF vvv
  return ::atanhf(__x);
#endif // !_CCCL_BUILTIN_ATANHF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float atanhf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANHF)
  return _CCCL_BUILTIN_ATANHF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANHF ^^^ / vvv !_CCCL_BUILTIN_ATANHF vvv
  return ::atanhf(__x);
#endif // !_CCCL_BUILTIN_ATANHF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double atanh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANH)
  return _CCCL_BUILTIN_ATANH(__x);
#else // ^^^ _CCCL_BUILTIN_ATANH ^^^ / vvv !_CCCL_BUILTIN_ATANH vvv
  return ::atanh(__x);
#endif // !_CCCL_BUILTIN_ATANH
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double atanh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANHL)
  return _CCCL_BUILTIN_ATANHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANHL ^^^ / vvv !_CCCL_BUILTIN_ATANHL vvv
  return ::atanhl(__x);
#  endif // !_CCCL_BUILTIN_ATANHL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double atanhl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANHL)
  return _CCCL_BUILTIN_ATANHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANHL ^^^ / vvv !_CCCL_BUILTIN_ATANHL vvv
  return ::atanhl(__x);
#  endif // !_CCCL_BUILTIN_ATANHL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half atanh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::atanhf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 atanh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::atanhf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double atanh(_Integer __x) noexcept
{
  return _CUDA_VSTD::atanh((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_INVERSE_HYPERBOLIC_FUNCTIONS_H
