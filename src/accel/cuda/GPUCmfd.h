/**
 * @file GPUCmfd.h
 * @brief A derived Cmfd class for the GPU.
 * @date August 2, 2020
 * @author Guillaume Giudicelli, MIT, Course 22 (g_giud@mit.edu)
 */

#ifndef GPUCMFD_H_
#define GPUCMFD_H_

#ifdef __cplusplus
#define _USE_MATH_DEFINES
#ifdef SWIG
#include "Python.h"
#endif
#include "Cmfd.h"
#endif

/**
 * @class GPU Cmfd GPUCmfd.h "src/accel/cuda/GPUCmfd.h"
 * @brief A class for Coarse Mesh Finite Difference (CMFD) acceleration on GPUs.
 */
class GPUCmfd {

private:

  void collapseXS() override;
  void constructMatrices() override;
  void rescaleFlux() override;
  void updateMOCFlux() override;

public:

  void GPUCmfd();
}

#endif /* GPUCMFD_H_ */
