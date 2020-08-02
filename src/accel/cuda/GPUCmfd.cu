#include "GPUCmfd.h"

collpaseXSonDevice(){

}


constructMatricesonDevice(){

}


rescaleFluxonDevice() {

}


updateMOCFluxonDevice() {

}


/**
 * @brief Constructor initializes boundaries and variables that describe
 *          the CMFD object.
 * @details The constructor initializes the many variables that describe
 *          the CMFD mesh and are used to solve the nonlinear diffusion
 *          acceleration problem.
 */
GPUCmfd::GPUCmfd()
  : Cmfd() {

  _gpu_cmfd = true;
}


/**
 * @brief Collapse cross-sections and fluxes for each CMFD cell by
 *        energy condensing and volume averaging cross sections from
 *        the MOC sweep.
 * @details This method performs a cell-wise energy condensation and flux-volume
 *          average of the cross sections of the fine, unstructured FSR mesh.
 *          The cross sections are condensed such that all reaction rates and
 *          the neutron production rate from fission are conserved. It is
 *          important to note that the volume averaging is performed before
 *          energy condensation in order to properly collapse the diffusion
 *          coefficients.
 */
void Cmfd::collapseXS() {

  log_printf(INFO, "Collapsing cross-sections onto CMFD mesh on GPU...");

}


/**
 * @brief Construct the loss + streaming matrix (A) and the fission gain
 *         matrix (M) in preparation for solving the eigenvalue problem.
 * @details This method loops over all mesh cells and energy groups and
 *          accumulates the iteraction and streaming terms into their
 *          appropriate positions in the loss + streaming matrix and
 *          fission gain matrix.
 */
void Cmfd::constructMatrices() {

  log_printf(INFO, "Constructing matrices on GPU...");

}


/**
 * @brief Rescale the initial and converged flux arrays.
 * @details The diffusion problem is a generalized eigenvalue problem and
 *          therefore the solution is independent of flux level. This method
 *          rescales the input flux and converged flux to both have an average
 *          fission source of 1.0 in each group in each cell.
 */
void Cmfd::rescaleFlux() {

}


/**
 * @brief Update the MOC flux in each FSR.
 * @details This method uses the condensed flux from the last MOC transport
 *          sweep and the converged flux from the eigenvalue problem to
 *          update the MOC flux in each FSR.
 */
void Cmfd::updateMOCFlux() {

  log_printf(INFO, "Updating MOC flux on GPU...");

}
