/**
 * @file Solver.h
 * @brief The Solver class.
 * @date February 7, 2012
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#ifdef __cplusplus
#define _USE_MATH_DEFINES
#ifdef SWIG
#include "Python.h"
#endif
#include "constants.h"
#include "Timer.h"
#include "Quadrature.h"
#include "TrackGenerator3D.h"
#include "Cmfd.h"
#include "Progress.h"
#include "ExpEvaluator.h"
#include "segmentation_type.h"
#include <math.h>
#endif

/** Indexing macro for the scalar flux in each FSR and energy group */
#define _scalar_flux(r,e) (_scalar_flux[(r)*_num_groups + (e)])

/** Indexing macro for the old scalar flux in each FSR and energy group */
#define _old_scalar_flux(r,e) (_old_scalar_flux[(r)*_num_groups + (e)])

/** Indexing macro for the reference scalar flux in each FSR and energy group */
#define _reference_flux(r,e) (_reference_flux[(r)*_num_groups + (e)])

/** Indexing macro for the reference scalar flux in each FSR and energy group */
#define reference_flux(r,e) (reference_flux[(r)*_num_groups + (e)])

/** Indexing macro for the stabailizing flux in each FSR and energy group */
#define _stabalizing_flux(r,e) (_stabalizing_flux[(r)*_num_groups + (e)])

/** Indexing macro for the total source divided by the total cross-section
 *  (\f$ \frac{Q}{\Sigma_t} \f$) in each FSR and energy group */
#define _reduced_sources(r,e) (_reduced_sources[(r)*_num_groups + (e)])

/** Indexing macro for the angular fluxes for each polar angle and energy
 *  group for the outgoing reflective track for both the forward and
 *  reverse direction for a given track */
#define _boundary_flux(i,j,pe) (_boundary_flux[(i)*2*_fluxes_per_track \
                                                + (j)*_fluxes_per_track \
                                               + (pe)])
#define _start_flux(i,j,pe) (_start_flux[(i)*2*_fluxes_per_track \
                                                + (j)*_fluxes_per_track \
                                               + (pe)])

#define _track_flux_buffer(i,pe) (_track_flux_buffer[(i)*_fluxes_per_track \
                                               + (pe)])

/** Indexing scheme for fixed sources for each FSR and energy group */
#define _fixed_sources(r,e) (_fixed_sources[(r)*_num_groups + (e)])

/** Indexing scheme for the total fission source (\f$ \nu\Sigma_f\Phi \f$)
 *  for each FSR and energy group */
#define fission_sources(r,e) (fission_sources[(r)*_num_groups + (e)])

/** Indexing scheme for the total in-scatter source (\f$ \Sigma_s\Phi \f$)
 *  for each FSR and energy group */
#define scatter_sources(r,e) (scatter_sources[(r)*_num_groups + (e)])


/**
 * @enum solverMode
 * @brief The solution mode used by the MOC solver.
 */
enum solverMode {

  /** The forward flux distribution */
  FORWARD,

  /** The adjoint flux distribution */
  ADJOINT,
};


/**
 * @enum residualType
 * @brief The type of residual used for the convergence criterion.
 */
enum residualType {

  /** A residual on the scalar flux distribution */
  SCALAR_FLUX,

  /** A residual on the fission source distribution */
  FISSION_SOURCE,

  /** A residual on the total source distribution */
  TOTAL_SOURCE,
};


/**
 * @enum stabalizationType
 * @brief The type of stabalization to use on source iteration
 */
enum stabalizationType {

  /** General diagonal stabalization */
  DIAGONAL,

  /** Yamamoto's groupwise stabalization */
  YAMAMOTO,

  /** Global damping on the scalar flux update */
  GLOBAL,
};


/**
 * @class Solver Solver.h "src/Solver.h"
 * @brief This is an abstract base class which different Solver subclasses
 *        implement for different architectures or source iteration algorithms.
 */
class Solver {

protected:

  /** The number of azimuthal angles */
  int _num_azim;

  /** The number of energy groups */
  int _num_groups;

  /** The number of flat source regions */
  long _num_FSRs;

  /** The number of fissionable flat source regions */
  long _num_fissionable_FSRs;

  /** The FSR "volumes" (i.e., areas) indexed by FSR UID */
  FP_PRECISION* _FSR_volumes;

  /** The FSR Material pointers indexed by FSR UID */
  Material** _FSR_materials;

  /** Material to be used for calculating the initial flux guess from chi */
  Material* _chi_spectrum_material;

  /** A pointer to a TrackGenerator which contains Tracks */
  TrackGenerator* _track_generator;

  /** A pointer to a Geometry with initialized FSR offset maps */
  Geometry* _geometry;

  /** The number of Materials */
  int _num_materials;

  /** A pointer to a polar quadrature */
  Quadrature* _quad;

  /** The number of polar angles */
  int _num_polar;

  /** The number of flux varies stored per track in each direction */
  int _fluxes_per_track;

  /** A pointer to the array of Tracks */
  Track** _tracks; //FIXME

  //FIXME
  std::vector<int> _limit_xs_materials;
  int _reset_iteration;
  bool _limit_xs;
  std::map<int, Material*> _original_materials;
  std::map<int, Material*> _limit_materials;

  /** A pointer to an array with the number of Tracks per azimuthal angle */
  int*** _tracks_per_stack;

  /** Boolean for whether to solve in 3D (true) or 2D (false) */
  bool _solve_3D;

  /** Boolean for whether to correct unphysical cross-sections */
  bool _correct_xs;

  /** Boolean for whether to apply the stabalizing correction to the source
    * iteration (transport sweep) process */
  bool _stabalize_transport;

  /** Boolean for whether to print verbose iteration reports */
  bool _verbose;
  
  /** Boolean for whether to perform a spectrum calculation for the initial 
   *  flux guess */
  bool _calculate_initial_spectrum;
  
  /** Convergence threshold for the initial spectrum calculation */
  double _initial_spectrum_thresh;
  
  /** Boolean for whether to load initial FSR flux profile from file */
  bool _load_initial_FSR_fluxes;
  
  /** Boolean for whether to calculate residuals from reference flux */
  bool _calculate_residuals_by_reference;

  /** File to load initial FSR fluxes from */
  std::string _initial_FSR_fluxes_file;
  
  /** File to load reference FSR fluxes from */
  std::string _reference_file;

  /** The log level for outputting cross-section inconsitencies */
  logLevel _xs_log_level;

  /** Determines the type of track segmentation to use for 3D MOC */
  segmentationType _segment_formation;

  /** The total number of Tracks */
  long _tot_num_tracks;

  /** The weights for each azimuthal angle */
  double* _azim_spacings;

  /** The weights for each polar angle in the polar angle quadrature */
  double** _polar_spacings;

  /** The angular fluxes for each Track for all energy groups, polar angles,
   *  and azimuthal angles. This array stores the boundary fluxes for a
   *  a Track along both "forward" and "reverse" directions. */
  float* _boundary_flux;
  float* _start_flux;

  /* Buffer for the track angular fluxes to avoid double saving them */
  std::vector<float*> _track_flux_buffer;

  /* Whether buffer has been updated with starting flux on track */
  std::vector<bool*> _buffer_updated;

  /** The angular leakages for each Track. This array stores the weighted
    * outgoing angular fluxes for use in non-CMFD eigenvalue calculations. */
  float* _boundary_leakage;

  /** The scalar flux for each energy group in each FSR */
  FP_PRECISION* _scalar_flux;

  /** The old scalar flux for each energy group in each FSR */
  FP_PRECISION* _old_scalar_flux;
  
  /** The reference scalar flux for each energy group in each FSR */
  FP_PRECISION* _reference_flux;
  
  /** The stabalizing flux for each energy group in each FSR */
  FP_PRECISION* _stabalizing_flux;

  /** Optional user-specified fixed sources in each FSR and energy group */
  FP_PRECISION* _fixed_sources;

  /** Temporary scratch pads for intermediate storage of computing steps */
  std::vector<FP_PRECISION*> _groupwise_scratch;
  double* _regionwise_scratch;

  /** A mapping of fixed sources keyed by the pair (FSR ID, energy group) */
  std::map< std::pair<int, int>, FP_PRECISION > _fix_src_FSR_map;

  /** A mapping of fixed sources keyed by the pair (Cell*, energy group) */
  std::map< std::pair<Cell*, int>, FP_PRECISION > _fix_src_cell_map;

  /** A mapping of fixed sources keyed by the pair (Material*, energy group) */
  std::map< std::pair<Material*, int>, FP_PRECISION > _fix_src_material_map;

  /** Ratios of source to total cross-section for each FSR and energy group */
  FP_PRECISION* _reduced_sources;

  /** The current iteration's approximation to k-effective */
  double _k_eff;

  /** The number of source iterations needed to reach convergence */
  int _num_iterations;

  /** The tolerance for converging the source/flux */
  double _converge_thresh;

  /** The factor applied to the source iteration stabilization */
  double _stabalization_factor;
  
  /** The type of source iteration stabilization */
  stabalizationType _stabalization_type;

  /** A matrix of ExpEvaluators to compute exponentials in the transport
    * equation. The matrix is indexed by azimuthal index and polar index */
  ExpEvaluator*** _exp_evaluators;

  /** The number of exponential evaluators in the azimuthal direction */
  int _num_exp_evaluators_azim;

  /** The number of exponential evaluators in the polar direction */
  int _num_exp_evaluators_polar;

  /** A timer to record timing data for a simulation */
  Timer* _timer;

  /** A pointer to a Coarse Mesh Finite Difference (CMFD) acceleration object */
  Cmfd* _cmfd;

  /** A string indicating the type of source apporximation */
  std::string _source_type;

  /**
   * @brief Initializes Track boundary angular flux and leakage and
   *        FSR scalar flux arrays.
   */
  virtual void initializeFluxArrays() =0;

  /**
   * @brief Allocates memory for FSR source arrays.
   */
  virtual void initializeSourceArrays() =0;

  virtual void initializeExpEvaluators();
  virtual void initializeFSRs();
  virtual void initializeTrackFluxBuffer();
  void countFissionableFSRs();
  void checkXS();
  virtual void initializeCmfd();
  void calculateInitialSpectrum(double threshold);

  /**
   * @brief Zero each Track's boundary fluxes for each energy group and polar
   *        angle in the "forward" and "reverse" directions.
   */
  virtual void zeroTrackFluxes() =0;

  /**
   * @brief Set the scalar flux for each FSR and energy group to some value.
   * @param value the value to assign to each FSR scalar flux
   */
  virtual void flattenFSRFluxes(FP_PRECISION value) =0;

  /**
   * @brief Set the scalar flux for each FSR to a chi spectrum.
   */
  virtual void flattenFSRFluxesChiSpectrum() =0;

  /**
   * @brief Stores the current scalar fluxes in the old scalar flux array.
   */
  virtual void storeFSRFluxes() =0;

  /**
   * @brief Normalizes all FSR scalar fluxes and Track boundary angular
   *        fluxes to the total fission source (times \f$ \nu \f$).
   */
  virtual double normalizeFluxes() =0;

  /**
   * @brief Computes the stabalizing flux for transport stabalization
   */
  virtual void computeStabalizingFlux() =0;
  
  /**
   * @brief Adjusts the scalar flux for transport stabalization
   */
  virtual void stabalizeFlux() =0;

  /**
   * @brief Computes the total source (fission, scattering, fixed) for
   *        each FSR and energy group.
   */
  virtual void computeFSRSources(int iteration) =0;

  /**
   * @brief Computes the residual between successive flux/source iterations.
   * @param res_type the type of residual (FLUX, FISSIOn_SOURCE, TOTAL_SOURCE)
   * @return the total residual summed over FSRs and energy groups
   */
  virtual double computeResidual(residualType res_type) =0;

  /**
   * @brief Compute \f$ k_{eff} \f$ from total fission and absorption rates
   *        in each FSR and energy group.
   */
  virtual void computeKeff() =0;

  /**
   * @brief Add the source term contribution in the transport equation to
   *        the FSR scalar flux.
   */
  virtual void addSourceToScalarFlux() =0;

  /**
   * @brief This method performs one transport sweep of all azimuthal angles,
   *        Tracks, segments, polar angles and energy groups.
   */
  virtual void transportSweep() =0;

  void clearTimerSplits();

  //FIXME
  bool _OTF_transport;

public:
  Solver(TrackGenerator* track_generator=NULL);
  virtual ~Solver();

  virtual void setGeometry(Geometry* geometry);

  Geometry* getGeometry();
  TrackGenerator* getTrackGenerator();
  FP_PRECISION getFSRVolume(long fsr_id);
  int getNumPolarAngles();
  int getNumIterations();
  double getTotalTime();
  double getKeff();
  double getConvergenceThreshold();
  FP_PRECISION getMaxOpticalLength();
  bool isUsingDoublePrecision();
  bool isUsingExponentialInterpolation();

  virtual void initializeFixedSources();

  void printFissionRates(std::string fname, int nx, int ny, int nz);
  virtual void printInputParamsSummary();

  void setResidualByReference(std::string fname);
  void dumpFSRFluxes(std::string fname);
  void loadInitialFSRFluxes(std::string fname);
  void loadFSRFluxes(std::string fname, bool assign_k_eff=false, double tolerance=0.01);

  virtual double getFlux(long fsr_id, int group);
  virtual void getFluxes(FP_PRECISION* out_fluxes, int num_fluxes) = 0;
  double getFSRSource(long fsr_id, int group);

  virtual void setTrackGenerator(TrackGenerator* track_generator);
  virtual void setConvergenceThreshold(double threshold);
  virtual void setFixedSourceByFSR(long fsr_id, int group, double source);
  void setFixedSourceByCell(Cell* cell, int group, double source);
  void setFixedSourceByMaterial(Material* material, int group,
                                double source);
  void setMaxOpticalLength(FP_PRECISION max_optical_length);
  void setExpPrecision(double precision);
  void useExponentialInterpolation();
  void useExponentialIntrinsic();
  void correctXS();
  void stabalizeTransport(double stabalization_factor,
                          stabalizationType stabalization_type=DIAGONAL);
  void setInitialSpectrumCalculation(double threshold);
  void setCheckXSLogLevel(logLevel log_level);
  void setChiSpectrumMaterial(Material* material);

  void computeFlux(int max_iters=1000, bool only_fixed_source=true);
  void computeSource(int max_iters=1000, double k_eff=1.0,
                     residualType res_type=TOTAL_SOURCE);
  void computeEigenvalue(int max_iters=1000,
                         residualType res_type=FISSION_SOURCE);
  void printBGQMemory();

  bool* getBufferUpdateArray(int tid);

 /**
  * @brief Computes the volume-weighted, energy integrated fission rate in
  *        each FSR and stores them in an array indexed by FSR ID.
  * @details This is a helper method for SWIG to allow users to retrieve
  *          FSR fission rates as a NumPy array. An example of how this method
  *          can be called from Python is as follows:
  *
  * @code
  *          num_FSRs = geometry.getNumFSRs()
  *          fission_rates = solver.computeFSRFissionRates(num_FSRs)
  * @endcode
  *
  * @param fission_rates an array to store the fission rates (implicitly passed
  *                      in as a NumPy array from Python)
  * @param num_FSRs the number of FSRs passed in from Python
  */
  virtual void computeFSRFissionRates(double* fission_rates, long num_FSRs) =0;

  /**
   * @brief Returns the boundary flux array at the requested indexes
   * @param track_id The Track's Unique ID
   * @param fwd Whether the direction of the angular flux along the track is
   *        forward (True) or backward (False)
   */
  inline float* getBoundaryFlux(long track_id, long index, int tid, bool fwd) {

    /* Copy flux from boundary into buffer */ //TODO replace by a mem copy
    if (_buffer_updated.at(tid)[index] == 0){
      for (int pe=0; pe<_fluxes_per_track; ++pe){
        _track_flux_buffer.at(tid)[index*_fluxes_per_track + pe] = 
                             _start_flux(track_id, !fwd, pe);
       }
      _buffer_updated.at(tid)[index] = 1;
    }

    return &_track_flux_buffer.at(tid)[index*_fluxes_per_track];
    //return &_boundary_flux(track_id, !fwd, 0);
  }

  //FIXME
  inline float* getBoundaryFluxFromBuffer(long index, int tid, bool fwd) {

    return &_track_flux_buffer.at(tid)[index*_fluxes_per_track];
  }

  void setVerboseIterationReport();
  void printTimerReport();
  FP_PRECISION* getFluxesArray();

  //FIXME
  void limitXS();
  void setLimitingXSMaterials(std::vector<int> material_ids,
                              int reset_iteration);
  virtual void checkLimitXS(int iteration);


  //FIXME
  inline void setOTFTransport() {
    _OTF_transport = true;
  }
};


#endif /* SOLVER_H_ */
