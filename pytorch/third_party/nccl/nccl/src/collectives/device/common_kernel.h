/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include "reduce_kernel.h"
#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>
#include <assert.h>

#define Pp if(blockDim.x*blockIdx.x+threadIdx.x==0)printf("%s:%d\n",__FILE__,__LINE__);

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

typedef uint64_t PackType;

// unpack x and y to elements of type T and apply FUNC to each element
template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const T c1, const T c2) const;
  __device__ PackType operator()(const PackType x, T y, const int alpha) const;
  __device__ PackType operator()(const PackType x, const PackType y, const T alpha) const;
  __device__ PackType operator()(const PackType x, const PackType y) const;
  __device__ PackType operator()(const PackType x, const T alpha) const;
  __device__ PackType operator()(const PackType x) const;
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const;
  __device__ PackType LAMBWeightUpdate(const PackType w, T ratio, const PackType rLambdaWeight) const;
  __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const T unscaleParameter, const  T beta2) const;
};

struct FuncProd2 {
  template<typename T>
  __device__ T operator()(const T x, const T y) const {
    T z = x * y;
    return z;
  }
};

struct FuncSub2 {
  template<typename T> 
  __device__ T operator()(const T x, const T y) const {
    return x - y;
  }
};

template<typename T> 
struct FuncSum2 {
  __device__ T operator()(const T x, const T y) const {
    T z = x + y;
    return z;
  }
};


struct FuncFMA2 {
  template<typename T> 
  __device__ T operator()(const T x, const T y, const T a) const {
    T z = a * x + y;
    return z;
  }
};


template<class FUNC>
struct MULTI<FUNC, int8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const int8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const int8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const int8_t y, const int alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const int8_t c1, const int8_t c2) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{return 0;}
  __device__ PackType LAMBWeightUpdate(const PackType w, int8_t ratio, const PackType rLambdaWeight) const {return 0;}
  __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const uint8_t unscaleParameter, const  int8_t beta2) {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, uint8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

   __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const uint8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const uint8_t alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const uint8_t y, const int alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const uint8_t c1, const uint8_t c2) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{return 0;}
  __device__ PackType LAMBWeightUpdate(const PackType w, uint8_t ratio, const PackType rLambdaWeight) const {return w;}
    __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const uint8_t unscaleParameter, const  uint8_t beta2) {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, int32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(int32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      int32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const int32_t alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, alpha);
    cr.b = FUNC()(cx.b, alpha);

    return cr.storage;
  }

  
  __device__ PackType operator()(const PackType x, const PackType y, const int32_t alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const int32_t y, const int32_t alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, y, alpha);
    cr.b = FUNC()(cx.b, y, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const int32_t c1, const int32_t c2) {
    converter cx, cy, cz, cr;
    cx.storage = x;
    cy.storage = y;
    cz.storage = z;

    cr.a = FUNC()(cx.a, cy.a, cz.a, c1, c2);
    cr.b = FUNC()(cx.b, cy.b, cz.a, c1, c2);

    return cr.storage;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, int32_t ratio, const PackType rLambdaWeight) const {}
    __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const int32_t unscaleParameter, const  int32_t beta2) {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, uint32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const uint32_t alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, alpha);
    cr.b = FUNC()(cx.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const uint32_t alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const int alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const uint32_t y, const int alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, y, alpha);
    cr.b = FUNC()(cx.b, y, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const uint32_t c1, const uint32_t c2) {
    converter cx, cy, cz, cr;
    cx.storage = x;
    cy.storage = y;
    cz.storage = z;

    cr.a = FUNC()(cx.a, cy.a, cz.a, c1, c2);
    cr.b = FUNC()(cx.b, cy.b, cz.a, c1, c2);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, int32_t ratio, const PackType rLambdaWeight) const {}
    __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const uint32_t unscaleParameter, const  uint32_t beta2) {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, half> {
  static_assert(sizeof(PackType) == 4 * sizeof(half),
      "PackType must be four times the size of half.");

  struct PackHalf2 {
    half2 a, b;
  };
  
  __device__ PackType operator()(const PackType x, const PackType y) const {
    struct PackHalf2 cx, cy, cr;
    cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
    cy = *(reinterpret_cast<const struct PackHalf2*>(&y));

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return *(reinterpret_cast<PackType*>(&cr));
  }

  __device__ PackType operator()(const PackType x, const half alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const half alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const half y, const int alpha) {
    assert(false);
    return x;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const half c1, const half c2) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, half ratio, const PackType rLambdaWeight) const {}
    __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const half unscaleParameter, const  half beta2) {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };


  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const float alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, alpha);
    cr.b = FUNC()(cx.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const float alpha) {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a, alpha);
    cr.b = FUNC()(cx.b, cy.b, alpha);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x) {
     converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a);
    cr.b = FUNC()(cx.b);

    return cr.storage;
  }

  __device__ PackType operator()(const PackType x, const float y, const int alpha) {
    converter cx, cr;
    cx.storage = x;

    cr.a = FUNC()(cx.a, y, alpha);
    cr.b = FUNC()(cx.b, y, alpha);

    return cr.storage;
  }
  
  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const float c1, const float c2) {
    converter cx, cy, cz, cr;
    cx.storage = x;
    cy.storage = y;
    cz.storage = z;

    cr.a = FUNC()(cx.a, cy.a, cz.a, c1, c2);
    cr.b = FUNC()(cx.b, cy.b, cz.b, c1, c2);

    return cr.storage;
  }

  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{
  converter cw;
cw.storage = w;
converter cS3;
cS3.storage = S3;
converter cS4;
cS4.storage = S4;
converter cS5;
cS5.a = FUNC()(cw.a, cS3.a, cS4.a);
cS5.b = FUNC()(cw.b, cS3.b, cS4.b);
return cS5.storage;
}
__device__ PackType LAMBWeightUpdate(const PackType w, float ratio, const PackType rLambdaWeight) const {
  converter cw;
  cw.storage = w;
  converter cS3;
  cS3.storage = rLambdaWeight;
  converter cS5;
  cS5.a = FUNC()(cw.a, ratio, cS3.a);
  cS5.b = FUNC()(cw.b, ratio, cS3.b);
  return cS5.storage;
}

  struct converterhalf{half2 x0;
  __device__ half getx0(){ return __low2half(x0);}
  __device__ half getx1(){ return __high2half(x0);}
};
  __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const float unscaleParameter, const  float beta2) {
    converter cv;
    cv.storage = v;
    converterhalf cS0;
    cS0 = *(reinterpret_cast<const converterhalf*>(&S0));
    converter cS2;
    cS2.a = FUNC()(cv.a, cS0.getx0(), unscaleParameter, beta2);
    cS2.b = FUNC()(cv.b, cS0.getx1(), unscaleParameter, beta2);
    //assert(cS2.FOO.x0 == 2.0f && cS2.FOO.x1 == 2.0f);
    return cS2.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y));
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const double alpha) {
    double rv = FUNC()(__longlong_as_double(x), alpha);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const double alpha) {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y), alpha);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const double y, const int alpha) {
    double rv = FUNC()(__longlong_as_double(x), y, alpha);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const double c1, const double c2) {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y), __longlong_as_double(z), c1, c2);
    return __double_as_longlong(rv);
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, double ratio, const PackType rLambdaWeight) const {}
    __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const double unscaleParameter, const  double beta2) {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, uint64_t> {
  static_assert(sizeof(PackType) == sizeof(uint64_t),
      "PackType must be the same size as uint64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    uint64_t rv = FUNC()(x, y);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const uint64_t alpha) {
    uint64_t rv = FUNC()((uint64_t)x, alpha);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const uint64_t alpha) {
    uint64_t rv = FUNC()((uint64_t)x, (uint64_t)y, alpha);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const uint64_t y, const int alpha) {
    uint64_t rv = FUNC()((uint64_t)x, (uint64_t)y, alpha);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const uint64_t c1, const uint64_t c2) {
    return FUNC()((uint64_t)x, (uint64_t)y, (uint64_t)z, c1, c2);
    return x;
  }

  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, uint64_t ratio, const PackType rLambdaWeight) const {}
    __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const uint64_t unscaleParameter, const  uint64_t beta2) {return 0;}
};

template<class FUNC>
struct MULTI<FUNC, int64_t> {
  static_assert(sizeof(PackType) == sizeof(int64_t),
      "PackType must be the same size as int64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    int64_t rv = FUNC()((int64_t)x, (int64_t)y);
    return rv;
  }

  __device__ PackType operator()(const PackType x, const int64_t alpha) {
    return FUNC()((int64_t)x, alpha);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const int64_t alpha) {
    return FUNC()((int64_t)x, (int64_t)y, alpha);
  }

  __device__ PackType operator()(const PackType x, const int64_t y, const int alpha) {
    return FUNC()((int64_t)x, (int64_t)y, alpha);
  }

  __device__ PackType operator()(const PackType x, const PackType y, const PackType z, const int64_t c1, const int64_t c2) {
    return FUNC()((int64_t)x, (int64_t)y, (int64_t)z, c1, c2);
  }
  
  __device__ PackType operator()(const PackType x) {
    assert(false);
    return x;
  }
  __device__ PackType r(const PackType w, const PackType S3, const PackType S4) const{}
  __device__ PackType LAMBWeightUpdate(const PackType w, int64_t ratio, const PackType rLambdaWeight) const {}
    __device__ uint64_t operator()(const uint64_t v, const uint32_t S0, const int64_t unscaleParameter, const  int64_t beta2) {return 0;}
};

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

#if CUDART_VERSION < 9000
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r.x = ptr->x;
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ptr->x = val.x;
}
#else
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r = ((half*)ptr)[0];
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ((half*)ptr)[0] = val;
}
#endif

typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(Pack128& x, Pack128& y) {
    x.x = MULTI<FUNC, T>()(x.x, y.x);
    x.y = MULTI<FUNC, T>()(x.y, y.y);
  }

  __device__ void operator()(Pack128& x, T alpha) {
    x.x = MULTI<FUNC, T>()(x.x, alpha);
    x.y = MULTI<FUNC, T>()(x.y, alpha);
  }

  __device__ void operator()(Pack128& x, Pack128& y, T alpha) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, alpha);
    x.y = MULTI<FUNC, T>()(x.y, y.y, alpha);
  }

  __device__ void operator()(Pack128& x, Pack128& y, Pack128& z, T alpha) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, z.x, alpha);
    x.y = MULTI<FUNC, T>()(x.y, y.y, z.y, alpha);
  }

  __device__ void operator()(Pack128& x, T beta, int alpha) {
    x.x = MULTI<FUNC, T>()(x.x, beta, alpha);
    x.y = MULTI<FUNC, T>()(x.y, beta, alpha);
  }

   __device__ void operator()(Pack128& x, Pack128& y, T z, int alpha) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, z, alpha);
    x.y = MULTI<FUNC, T>()(x.y, y.y, z, alpha);
  }

   __device__ void operator()(Pack128& x, Pack128& y, Pack128& z, T alpha, T epsilon) {
    x.x = MULTI<FUNC, T>()(x.x, y.x, z.x, alpha, epsilon);
    x.y = MULTI<FUNC, T>()(x.y, y.y, z.y, alpha, epsilon);
  }

  __device__ void r(Pack128& w, Pack128& S3, Pack128& S4, Pack128& S5) {
    S5.x = MULTI<FUNC, T>().r(w.x, S3.x, S4.x);
    S5.y = MULTI<FUNC, T>().r(w.y, S3.y, S4.y);
  }

  __device__ void LAMBWeightUpdate(Pack128& w, T ratio, Pack128& rLambdaWeight, Pack128& S5) {
    S5.x = MULTI<FUNC, T>().LAMBWeightUpdate(w.x, ratio, rLambdaWeight.x);
    S5.y = MULTI<FUNC, T>().LAMBWeightUpdate(w.y, ratio, rLambdaWeight.y);
  }
};

template<class FUNC, typename T1, typename T2>
struct MULTI128TwoTypes {
  // __device__ void operator()(Pack128& x, Pack128& y) {
  //   // x.x = MULTI<FUNC, T>()(x.x, y.x);
  //   // x.y = MULTI<FUNC, T>()(x.y, y.y);
  // }

  // __device__ void operator()(Pack128& x, T2 alpha) {
  //   // x.x = MULTI<FUNC, T1>()(x.x, alpha);
  //   // x.y = MULTI<FUNC, T1>()(x.y, alpha);
  // }

  __device__ void operator()(Pack128& x, uint2 y, T2 unscaleParameter, T2 alpha) {
    x.x = MULTI<FUNC, T2>()(x.x, y.x, unscaleParameter, alpha);
    x.y = MULTI<FUNC, T2>()(x.y, y.y, unscaleParameter, alpha);
  }

  // __device__ void operator()(Pack128& x, Pack128& y, Pack128& z, T2 alpha) {
  //   x.x = MULTI<FUNC, T1>()(x.x, y.x, z.x, alpha);
  //   x.y = MULTI<FUNC, T1>()(x.y, y.y, z.y, alpha);
  // }

  // __device__ void operator()(Pack128& x, T2 beta, int alpha) {
  //   x.x = MULTI<FUNC, T1>()(x.x, beta, alpha);
  //   x.y = MULTI<FUNC, T1>()(x.y, beta, alpha);
  // }

  //  __device__ void operator()(Pack128& x, Pack128& y, T2 z, int alpha) {
  //   x.x = MULTI<FUNC, T1>()(x.x, y.x, z, alpha);
  //   x.y = MULTI<FUNC, T1>()(x.y, y.y, z, alpha);
  // }

  //  __device__ void operator()(Pack128& x, Pack128& y, Pack128& z, T2 alpha, T2 epsilon) {
  //   x.x = MULTI<FUNC, T1>()(x.x, y.x, z.x, alpha, epsilon);
  //   x.y = MULTI<FUNC, T1>()(x.y, y.y, z.y, alpha, epsilon);
  // }

  // __device__ void r(Pack128& w, Pack128& S3, Pack128& S4, Pack128& S5) {
  //   S5.x = MULTI<FUNC, T1>().r(w.x, S3.x, S4.x);
  //   S5.y = MULTI<FUNC, T1>().r(w.y, S3.y, S4.y);
  // }

  // __device__ void LAMBWeightUpdate(Pack128& w, T2 ratio, Pack128& rLambdaWeight, Pack128& S5) {
  //   S5.x = MULTI<FUNC, T1>().LAMBWeightUpdate(w.x, ratio, rLambdaWeight.x);
  //   S5.y = MULTI<FUNC, T1>().LAMBWeightUpdate(w.y, ratio, rLambdaWeight.y);
  // }
};


inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

// template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE>
// __device__ __forceinline__ void ReduceCopyMulti(const int tid, const int nthreads,
//     int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
//     const int offset, const int N) {
//   for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
//     T val = vFetch(srcs[0]+idx);
//     #pragma unroll
//     for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
//     #pragma unroll 1
//     for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
//     if (WEIGHT_UPDATE) {
//       assert(false);
//       #pragma unroll
//       for (int i=0; i<MINDSTS; i++) {
//         T update = FuncSub2()(vFetch(((float*)dsts[i])+idx), (float)val);
//         vStore(dsts[i]+idx, update);
//       } 
//       #pragma unroll 1
//       for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) 
//       {
//         T update = FuncSub2()(vFetch(((float*)dsts[i])+idx), (float)val);
//         vStore(dsts[i]+idx, update);
//       }
//     } else {
//       assert(false);
//       #pragma unroll
//       for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
//       #pragma unroll 1
//       for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
//     }
//   }
// }

template<class FUNC, typename T, typename T2, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE>
__device__ __forceinline__ void ReduceCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    const int offset, const int N, T2 alpha, T2 beta1, T2 beta2, const T2 unscaleParameter,const int epoch, T2* m, T2* v,
    T2* rStorage, T2* floatWeights, const size_t mvStartOffset, int partStartOffset, int partSize, double* weightNorm, double* rNorm, 
    const size_t buffNumElements, int* numOverflows) {
  
  if (threadIdx.x == 0 && sizeof(T) == sizeof(double)) {
    printf("%d\n", __LINE__);
  } 
  double perThreadWeightNorm = 0, perThreadRNorm = 0;
  for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
    T val = vFetch(srcs[0]+idx);
    #pragma unroll
    for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));

    if (WEIGHT_UPDATE) {
      const size_t totalOffset = (mvStartOffset + idx);//%(totalSize/nranks);
      const size_t mOffset = partStartOffset + totalOffset%partSize;
      // size_t mOffset = idx;
      T2 m_ = vFetch(m + mOffset);
      T2 v_ = vFetch(v + mOffset);
      T2 wght_ = vFetch(floatWeights + idx);
      *numOverflows |= FuncIsOverflow<T>()(val);
      m_ = FuncFirstMomentUpdateMP<T, T2>()(m_, val, unscaleParameter, beta1);
      vStore(m + mOffset, m_);

      v_ = FuncSecondMomentUpdateMP<T, T2>()(v_, val, unscaleParameter, beta2);
      vStore(v + mOffset, v_);

      m_ = FuncBiasCorrection<T2>()(m_, beta1, epoch+1);
      v_ = FuncBiasCorrection<T2>()(v_, beta2, epoch+1);

      if (LAMB) {
        perThreadWeightNorm += ((double)(wght_*wght_))/buffNumElements;
        // perThreadWeightNorm += ((double)(wght_*wght_))/1;
        T2 r_ = r<T2>()(wght_, m_, v_);
        perThreadRNorm += ((double)(r_*r_))/buffNumElements;
        vStore(rStorage + mOffset, r_);
      } else {
        wght_ = FuncAdamWeightUpdate<T2>()(wght_, m_, v_, alpha, 1e-6);      
        vStore(floatWeights + idx, wght_);
        val = (T)(wght_);
      }
    } else if (LAMB_SEND_COMPUTE) {
      const size_t totalOffset = (mvStartOffset + idx);//%(totalSize/nranks);
      const size_t mOffset = partStartOffset + totalOffset%partSize;
      T2 wght_ = vFetch(floatWeights + idx);

      double scale = ((*weightNorm > 0) ? (*rNorm > 0 ? *weightNorm/(*rNorm) : 1.0f) : 1.0f)/(*rNorm);
      wght_ = LAMBWeightUpdate<T2>()(wght_, alpha*(T2)scale, *(rStorage + mOffset));
      val = (T)wght_;
      vStore((T*)srcs[0]+idx, val);
    }

    #pragma unroll
    for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
  }

  if (LAMB and WEIGHT_UPDATE) {
    atomicAdd(weightNorm, perThreadWeightNorm);
    atomicAdd(rNorm, perThreadRNorm);
  }
}

union LasF {
  float f1, f2;
  uint64_t l;
};

struct halfToUint64_t {
    half2 h1;
    half2 h2;
};

inline __device__ uint64_t float4ToHalf4(Pack128& v) {
  float2 h1 = *(reinterpret_cast<float2*>(&v.x));
  float2 h2 = *(reinterpret_cast<float2*>(&v.y));
  // assert (h1.x == -1.0f);
  // assert (h1.y == -1.0f);
  // assert (h1. == -1.0f);

  half2 r1 = __floats2half2_rn(h1.x, h1.y);
  half2 r2 = __floats2half2_rn(h2.x, h2.y);

  halfToUint64_t converter;
  converter.h1 = r1;
  converter.h2 = r2;

  return *(reinterpret_cast<uint64_t*>(&converter));
}

template<class FUNC, typename T, typename T2, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE>
__device__ __forceinline__ void ReduceCopy128bMulti( const int w, const int nw, const int t,
    int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
    T2* firstMoment, T2* secondMoment, T2* rStorage, T2* floatWeights,
    const int elemOffset, const int Npack, const T2 alpha, const T2 beta1, const T2 beta2, const T2 unscaleParameter, const int epoch,
    const size_t mvStartOffset, int partStartOffset, int partSize, double* weightNorm, double* rNorm, const size_t buffNumElements, int* numOverflows) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;
  const uint64_t* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const uint64_t*)(s[i]+elemOffset))+offset;
  uint64_t* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((uint64_t*)(d[i]+elemOffset))+offset;
  //Pack128* firstMomentPacked = ((Pack128*)(firstMoment+elemOffset))+offset;
  //Pack128* secondMomentPacked = ((Pack128*)(secondMoment+elemOffset))+offset;
  double perThreadWeightNorm = 0.0f;
  double perThreadRNorm = 0.0f;
  // if (LAMB_SEND_COMPUTE) {
  //   if (threadIdx.x == 0) {
  //           printf("rStorage %p\n", rStorage);
  //         }
  // }
  const int newNPack = (sizeof(T2) != sizeof(T)) ? (Npack*sizeof(T2)/sizeof(T)) : Npack * 2;
  while (offset < newNPack) {
    uint64_t vals[UNROLL];
    // Load and reduce
    if (!LAMB_SEND_COMPUTE) {
      for (int u = 0; u < UNROLL; ++u) {
        vals[u] = *(srcs[0]+u*WARP_SIZE);
        if (!WEIGHT_UPDATE)
          *(const_cast<uint64_t*>(srcs[0]+u*WARP_SIZE)) = 0;
      }

      for (int i=1; i<MINSRCS; i++) {
        uint64_t vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FUNC, T>()(vals[u], vals2[u]);
      }
      #pragma unroll 1
      for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
        uint64_t vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = *(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = MULTI<FUNC, T>()(vals[u], vals2[u]);
      }
    }

    // Store
    if (WEIGHT_UPDATE) {
      if (firstMoment != NULL and secondMoment != NULL) {
        //ADAM
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 wght, m, v;
          uint2 _vals = *(reinterpret_cast<const uint2*>(&vals[u]));
          half2 half2_1 = *(reinterpret_cast<half2*>(&_vals.x));
          half2 half2_2 = *(reinterpret_cast<half2*>(&_vals.y));
          //TODO: Lets inline it because creating a MULTI128 and MULTI functions for all types will take quite some time
          *numOverflows |= (FuncIsOverflow<half>()(half2_1.x) || FuncIsOverflow<half>()(half2_1.y) || 
                            FuncIsOverflow<half>()(half2_2.x) || FuncIsOverflow<half>()(half2_2.y));

          const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T2)));//%(totalSize/nranks);
          const size_t mOffset = partStartOffset + totalOffset%partSize;
          const size_t floatWeightsOffset = (elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T2)));
          Pack128* floatWeightsPacked = (Pack128*)(floatWeights + floatWeightsOffset);
          Pack128* firstMomentPacked = (Pack128*)(firstMoment + mOffset);
          Pack128* secondMomentPacked = (Pack128*)(secondMoment + mOffset);
          Pack128* rStoragePack = (Pack128*)(rStorage + mOffset);

          Fetch128(wght, floatWeightsPacked);
          // float4 wf4 = *(reinterpret_cast<float4*>(&wght));
          // if (threadIdx.x == 0) {
          //     printf("933: wf4.x %f\n", wf4.x);
          // }
          
          Fetch128(m, firstMomentPacked);
          Fetch128(v, secondMomentPacked);
          // Pack128 readm = m;
          // float4 mf4 = *(reinterpret_cast<float4*>(&m));
          // if (mf4.x != 0.0) {
          //     printf("844: mf4.x %f totalOffset %ld mvStartOffset %ld threadIdx.x %d secondMoment %p firstMoment %p buffNumElements %ld\n", mf4.x, totalOffset, mvStartOffset, threadIdx.x, secondMoment, firstMoment, buffNumElements);
          // }
          // if (buffNumElements == 2048 and totalOffset < 31260672) {
          //   printf("845: totalOffset %ld\n", totalOffset);
          // }
          MULTI128TwoTypes<FuncFirstMomentUpdate<T>, T, T2>()(m, _vals, unscaleParameter, beta1);
          Store128(firstMomentPacked, m);
          // Pack128 pm = m;

          MULTI128TwoTypes<FuncSecondMomentUpdate<T>, T, T2>()(v, _vals, unscaleParameter, beta2);
          Store128(secondMomentPacked, v);

          MULTI128<FuncBiasCorrection<T2>, T2>()(m, beta1, epoch+1);

          MULTI128<FuncBiasCorrection<T2>, T2>()(v, beta2, epoch+1);
          uint64_t wghtHalf = 0;

          if (LAMB) {
            float4 f4 = *(reinterpret_cast<float4*>(&wght));
            perThreadWeightNorm += ((double)(f4.x * f4.x))/buffNumElements + ((double)(f4.y * f4.y))/buffNumElements + 
                                   ((double)(f4.z * f4.z))/buffNumElements + ((double)(f4.w * f4.w))/buffNumElements;
            // perThreadWeightNorm += ((double)(f4.x * f4.x))/1 + ((double)(f4.y * f4.y))/1 + 
            //                        ((double)(f4.z * f4.z))/1 + ((double)(f4.w * f4.w))/1;
            // if (threadIdx.x == 0) {
            //   printf("f4.x %f\n", f4.x);
            // }
            Pack128 r_;
            MULTI128<r<T2>, T2>().r(wght, m, v, r_);
            f4 = *(reinterpret_cast<float4*>(&r_));
            
            perThreadRNorm += ((double)(f4.x * f4.x))/buffNumElements + ((double)(f4.y * f4.y))/buffNumElements + 
                              ((double)(f4.z * f4.z))/buffNumElements + ((double)(f4.w * f4.w))/buffNumElements;
            Store128(rStoragePack, r_);
            // float4 rf4 = *(reinterpret_cast<float4*>(&r_));
            // float4 mf4 = *(reinterpret_cast<float4*>(&m));
            // // if (fabs(rf4.x - 2.0)/2.0 > 1e-5) {
            // //     printf("862: rf4.x %f mf4.x %f\n", rf4.x, mf4.x);
            // // }
          } else {
            MULTI128<FuncAdamWeightUpdate<T2>, T2>()(wght, m, v, alpha, (T2)1e-6);
            wghtHalf = float4ToHalf4(wght);
            Store128(floatWeightsPacked, wght);
          }

          // if(epoch == 1) {
          // float4 wf4 = *(reinterpret_cast<float4*>(&wght));
          // float4 vf4 = *(reinterpret_cast<float4*>(&v));
          // float4 mf4 = *(reinterpret_cast<float4*>(&m));
          // float4 pmf4 = *(reinterpret_cast<float4*>(&pm));
          // float4 readmf4 = *(reinterpret_cast<float4*>(&readm));
          // half* gh = (half*)(&_vals);
          // if (threadIdx.x == 0) {
          //     printf("844: mf4.x %f vf4.x %f wf4.x %f gh %f sizeof(T2) %ld beta1 %f pmf4.x %f readmf4 %f\n", mf4.x, vf4.x, wf4.x, (float)gh[0], sizeof(T2), (float)beta1, (float)pmf4.x, readmf4.x);
          // }
          // }
          
           
          //In LAMB this ndsts is 0 (and so is MINDSTS)
          for (int i = 0; i < MINDSTS; i++) {
            *(dsts[i]+u*WARP_SIZE) = wghtHalf;
          }
        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            *(dsts[i]+u*WARP_SIZE) = wghtHalf;
          }
        }
      } else {
        assert(false);
        //SGD
        #if 0
        for (int u = 0; u < UNROLL; ++u) {
          Pack128 val2;
          Fetch128(val2, dsts[0]+u*WARP_SIZE);
          Pack128 _vals = vals[u];
          MULTI128<FuncFMA2, T>()(_vals, val2, alpha);

          for (int i = 0; i < MINDSTS; i++) {
            Store128(dsts[i]+u*WARP_SIZE, _vals);
          }

        #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            Store128(dsts[i]+u*WARP_SIZE, _vals);
          }
        }
        #endif
      }
    } else if (LAMB_SEND_COMPUTE) {
        for (int u = 0; u < UNROLL; ++u) {
          const size_t totalOffset = (mvStartOffset + elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T2)));
          const size_t mOffset = partStartOffset + totalOffset%partSize;
          const size_t floatWeightsOffset = (elemOffset + (offset + u*WARP_SIZE)*(sizeof(Pack128)/sizeof(T2)));
          Pack128* floatWeightsPacked = (Pack128*)(floatWeights + floatWeightsOffset);
          Pack128 rLambdaW;
          Fetch128(rLambdaW, (Pack128*)(rStorage+mOffset));
          double scale = ((*weightNorm > 0) ? (*rNorm > 0 ? *weightNorm/(*rNorm) : 1.0f) : 1.0f)/(*rNorm);
          Pack128 wght;
          Fetch128(wght, floatWeightsPacked);
          Pack128 finalVal;
          MULTI128<LAMBWeightUpdate<T2>, T2>().LAMBWeightUpdate(wght, alpha*(T2)scale, rLambdaW, finalVal);
          // float4 f4 = *(reinterpret_cast<float4*>(&finalVal));
          // if (threadIdx.x == 0) {
          //   printf("f4.x %f\n", f4.x);
          // }
          // float4 rf4 = *(reinterpret_cast<float4*>(&rLambdaW));
          // if (buffNumElements == 31260672 && fabs(f4.x - 0.5)/0.5 > 1e-5) {
          //     printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.x %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.x);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.y - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.y %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.y);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.z - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.z %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.z);
          // }
          // if (buffNumElements == 31260672 && fabs(f4.w - 0.5)/0.5 > 1e-5) {
          //   printf("906: weightNorm %f rNorm %f scale %f alpha %f rLambdaW %f f4.w %f\n", (float)(*weightNorm), (float)(*rNorm), (float)scale, (float)alpha, (float)rf4.x, f4.w);
          // }

          // if (buffNumElements == 2048) {
          //   f4.x = 4.f/3.f;
          //   f4.y = 4.f/3.f;
          //   f4.z = 4.f/3.f;
          //   f4.w = 4.f/3.f;
          //   finalVal = *(reinterpret_cast<Pack128*>(&f4));
          // } else {
          //   f4.x = 0.5f;
          //   f4.y = 0.5f;
          //   f4.z = 0.5f;
          //   f4.w = 0.5f;
          //   finalVal = *(reinterpret_cast<Pack128*>(&f4));
          // }
          // if (buffNumElements == 31260672 && totalOffset >= 31260672) {
          //   printf("f4.x %f totalOffset %ld\n", f4.x, totalOffset);
          // }
          uint64_t finalValHalf = float4ToHalf4(finalVal);
          Store128(floatWeightsPacked, finalVal);
          *((uint64_t*)srcs[0] + u*WARP_SIZE) = finalValHalf;

          for (int i = 0; i < MINDSTS; i++) {
            *(dsts[i]+u*WARP_SIZE) = finalValHalf;
          }
          #pragma unroll 1
          for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
            *(dsts[i]+u*WARP_SIZE) = finalValHalf;
          }
        }
    } else {
      for (int i = 0; i < MINDSTS; i++) {
        for (int u = 0; u < UNROLL; ++u) {
          *(dsts[i]+u*WARP_SIZE) = vals[u];
        }
      }
      #pragma unroll 1
      for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
        for (int u = 0; u < UNROLL; ++u) {
          *(dsts[i]+u*WARP_SIZE) = vals[u];
        }
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    // firstMomentPacked += inc;
    // secondMomentPacked += inc;
    offset += inc;
  }

  if (LAMB and WEIGHT_UPDATE) {
    atomicAdd(weightNorm, perThreadWeightNorm);
    atomicAdd(rNorm, perThreadRNorm);
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

// Try to limit consecutive load/stores to 8.
// Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
#define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))

template<int UNROLL, class FUNC, typename T, typename T2, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int WEIGHT_UPDATE, int LAMB, int LAMB_SEND_COMPUTE>
__device__ __forceinline__ void ReduceOrCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    T2* firstMoment, T2* secondMoment, T2* rStorage, T2* floatWeights,
    int N, const T2 alpha, const T2 beta1, const T2 beta2, const T2 unscaleParameter, const int epoch, const size_t mvStartOffset, int partStartOffset, 
    int partSize, double* weightNorm, double* rNorm, const size_t buffNumElements, int* numOverflows) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int alignDiff = 0;
  int align = ptrAlign128(srcs[0]);
  #pragma unroll
  for (int i=1; i<MINSRCS; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  if (firstMoment)
    alignDiff |= (align ^ ptrAlign128(firstMoment));
  if (secondMoment)
    alignDiff |= (align ^ ptrAlign128(secondMoment));  
  if (rStorage)
    alignDiff |= (align ^ ptrAlign128(rStorage));
  if (floatWeights)
    alignDiff |= (align ^ ptrAlign128(floatWeights));

  int Npreamble = alignDiff ? Nrem :
    N < alignof(Pack128) ? N :
    (alignof(Pack128) - align) % alignof(Pack128);

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment

  if (Npreamble) {
    ReduceCopyMulti<FUNC, T, T2, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble, alpha, beta1, beta2, unscaleParameter,epoch, firstMoment, secondMoment, rStorage, floatWeights, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements, numOverflows);
    Nrem -= Npreamble;
    if (Nrem == 0) return;
  }
  int offset = Npreamble;

  // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 128-bit alignable.
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  const int packFactor = sizeof(Pack128) / sizeof(T);
  // stage 2a: main loop
  int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
      * (AUTOUNROLL * WARP_SIZE); // round down
  int Nelem2a = Npack2a * packFactor;
 
  ReduceCopy128bMulti<FUNC, T, T2, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>(w, nw, t, nsrcs, srcs, ndsts, dsts, firstMoment, secondMoment, rStorage, floatWeights, offset, Npack2a, alpha, beta1, beta2, unscaleParameter, epoch, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements, numOverflows);

  Nrem -= Nelem2a;
  if (Nrem == 0) return;
  offset += Nelem2a;

  // stage 2b: slightly less optimized for section when we don't have full
  // unrolling

  int Npack2b = Nrem / packFactor;
  int Nelem2b = Npack2b * packFactor;

  ReduceCopy128bMulti<FUNC, T, T2, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>(w, nw, t, nsrcs, srcs, ndsts, dsts, firstMoment, secondMoment, rStorage, floatWeights, offset, Npack2b, alpha, beta1, beta2, unscaleParameter, epoch, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements, numOverflows);

  Nrem -= Nelem2b;
  if (Nrem == 0) return;
  offset += Nelem2b;

  // stage 2c: tail
  ReduceCopyMulti<FUNC, T, T2, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, WEIGHT_UPDATE, LAMB, LAMB_SEND_COMPUTE>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem, alpha, beta1, beta2, unscaleParameter, epoch, firstMoment, secondMoment, rStorage, floatWeights, mvStartOffset, partStartOffset, partSize, weightNorm, rNorm, buffNumElements, numOverflows);
}

#endif // COMMON_KERNEL_H_