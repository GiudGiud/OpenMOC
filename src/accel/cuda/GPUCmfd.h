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
#include "../../Cmfd.h"
#endif

#define PySys_WriteStdout printf

#include <thrust/copy.h>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include "GPUHelper_functions.h"
#include "clone.h"
#include "dev_exponential.h"
#include "GPUQuery.h"
#include <cusparse.h>
#include <cuda_runtime.h>

/* Indexing macros for the 1D arrays */
#define volume_tally(i,e) volume_tally[i*num_cmfd_groups + e]
#define reaction_tally(i,e) reaction_tally[i*num_cmfd_groups + e]
#define diffusion_tally(i,e) diffusion_tally[i*num_cmfd_groups + e]
#define old_flux(i,e) old_flux[i*num_cmfd_groups + e]
#define new_flux(i,e) new_flux[i*num_cmfd_groups + e]
#define old_dif_surf_corr(i,s,e) old_dif_surf_corr[i*num_cmfd_groups*NUM_FACES + e*NUM_FACES + s]
#define surface_currents(i,s,e) surface_currents[i*num_cmfd_groups*NUM_FACES + s*num_cmfd_groups + e]
#define dA_csrOffsets(i,e) dA_csrOffsets[i*num_cmfd_groups + e]
#define dM_csrOffsets(i,e) dM_csrOffsets[i*num_cmfd_groups + e]
#define sin_thetas(a,p) sin_thetas[a*num_polar/2 + p]
#define polar_weights(a,p) sin_thetas[a*num_polar/2 + p]

/**
 * @class GPU Cmfd GPUCmfd.h "src/accel/cuda/GPUCmfd.h"
 * @brief A class for Coarse Mesh Finite Difference (CMFD) acceleration on GPUs.
 */
class GPUCmfd : public Cmfd {

private:

  /** The number of thread blocks */
  int _B;

  /** The number of threads per thread block */
  int _T;

  dev_material* _materials;
  int* _FSR_materials;
  dev_material* _dev_fsr_materials;
  thrust::device_vector<CMFD_PRECISION> _new_flux;
  thrust::device_vector<CMFD_PRECISION> _old_flux;
  thrust::device_vector<FP_PRECISION> _FSR_fluxes;
  thrust::device_vector<CMFD_PRECISION> _new_source;
  thrust::device_vector<CMFD_PRECISION> _old_source;
  thrust::device_vector<FP_PRECISION> _volumes;
  thrust::device_vector<FP_PRECISION> _FSR_volumes;
  thrust::device_vector<CMFD_PRECISION> _volume_tally;
  thrust::device_vector<CMFD_PRECISION> _reaction_tally;
  thrust::device_vector<CMFD_PRECISION> _diffusion_tally;
  thrust::device_vector<CMFD_PRECISION> _old_dif_surf_corr;

  /** Array of surface currents for each CMFD cell */
  thrust::device_vector<CMFD_PRECISION> _surface_currents;

  thrust::device_vector<long> _cmfd_cells_fsrs_index;
  thrust::device_vector<long> _cmfd_cells_fsrs;

  thrust::device_vector<double> _dev_cell_widths_x;
  thrust::device_vector<double> _dev_cell_widths_y;
  thrust::device_vector<double> _dev_cell_widths_z;

  /* A matrix storage */
  int _A_num_nnz;
  thrust::device_vector<int> dev_dA_csrOffsets;
  int   *dA_columns;
  float *dA_values;

  /* M matrix storage */
  cusparseSpMatDescr_t _M;

  void initializeMaterials();
  void setFSRVolumes(FP_PRECISION* FSR_volumes);

  void collapseXS() override;
  void constructMatrices() override;
  double solveEigenvalueProblem() override;
  void rescaleFlux() override;
  void updateMOCFlux() override;
  void setQuadrature(Quadrature* _quadrature) override;

  double computeResidual();

public:

  GPUCmfd();
  ~GPUCmfd();

  void copySurfaceCurrents(thrust::device_vector<CMFD_PRECISION> surface_currents);
  void setFSRFluxes(FP_PRECISION* scalar_flux, bool from_device);
  void setFSRMaterials(int* FSR_materials, dev_material* dev_materials);
};

#endif /* GPUCMFD_H_ */
