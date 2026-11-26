// Error-checking macros for CUDA runtime, cuFFT and NPP calls. Any failure
// prints where it happened and ends the program.

#ifndef SRC_CUDA_CHECK_H_
#define SRC_CUDA_CHECK_H_

#include <cuda_runtime.h>
#include <cufft.h>
#include <npp.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    const cudaError_t cuda_check_status = (call);                         \
    if (cuda_check_status != cudaSuccess) {                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                   cudaGetErrorName(cuda_check_status), __FILE__,         \
                   __LINE__, cudaGetErrorString(cuda_check_status));      \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define CUFFT_CHECK(call)                                                 \
  do {                                                                    \
    const cufftResult cufft_check_status = (call);                        \
    if (cufft_check_status != CUFFT_SUCCESS) {                            \
      std::fprintf(stderr, "cuFFT error %d at %s:%d\n",                   \
                   static_cast<int>(cufft_check_status), __FILE__,        \
                   __LINE__);                                             \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// NPP returns negative codes for errors and positive codes for warnings.
#define NPP_CHECK(call)                                                   \
  do {                                                                    \
    const NppStatus npp_check_status = (call);                            \
    if (npp_check_status < NPP_NO_ERROR) {                                \
      std::fprintf(stderr, "NPP error %d at %s:%d\n",                     \
                   static_cast<int>(npp_check_status), __FILE__,          \
                   __LINE__);                                             \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#endif  // SRC_CUDA_CHECK_H_
