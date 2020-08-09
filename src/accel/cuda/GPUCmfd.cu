#include "GPUCmfd.h"
#include <cusparse.h>

/** The number of MOC energy groups */
__constant__ int num_moc_groups;

/** The number of CMFD energy groups */
__constant__ int num_cmfd_groups;

/** The map from MOC to CMFD groups */
__constant__ int cmfd_group_map[MAX_NUM_GROUPS_GPU];

/** The map from CMFD to MOC groups */
__constant__ int group_indices[MAX_NUM_GROUPS_GPU];

/** The number of azimuthal angles in the quadrature */
__constant__ int num_azim;

/** The number of polar angles in the quadrature */
__constant__ int num_polar;

/** An array for the sines of the polar angle in the Quadrature set */
__constant__ FP_PRECISION sin_thetas[MAX_POLAR_ANGLES_GPU];

/** An array of the azimuthal weights from the Quadrature set */
__constant__ FP_PRECISION azim_weights[MAX_AZIM_ANGLES_GPU];

/** An array of the polar weights from the Quadrature set */
__constant__ FP_PRECISION polar_weights[MAX_POLAR_ANGLES_GPU*MAX_AZIM_ANGLES_GPU];

/** The CMFD cell mesh dimensions */
__constant__ int local_num_x;
__constant__ int local_num_y;
__constant__ int local_num_z;

/** The boundary conditions around the domain */
__constant__ int boundaries[6];

/** Whether solver is using linear source or flat source */
__constant__ bool linear_source;

/** Whether to use flux limiting of diffusion coefficients */
__constant__ bool _flux_limiting = true;

/** Diffusion coefficient relaxation factor */
__constant__ float relaxation_factor;

/**
 * @brief Returns the width of a given surface
 * @param surface A surface index, from 0 to NUM_FACES - 1
 * @param global_ind global index of a CMFD cell
 * @param cell_widths_x widths of Mesh cells in the x-direction
 * @param cell_widths_y widths of Mesh cells in the y-direction
 * @param cell_widths_z widths of Mesh cells in the z-direction
 * @return The surface width
 */
__device__ FP_PRECISION getSurfaceWidth(int surface,
                             int global_ind,
                             double* cell_widths_x,
                             double* cell_widths_y,
                             double* cell_widths_z) {

  int _num_x = local_num_x;
  int _num_y = local_num_y;
  int ix = global_ind % _num_x;
  int iy = (global_ind % (_num_x * _num_y)) / _num_x;
  int iz = global_ind / (_num_x * _num_y);

  if (surface == SURFACE_X_MIN || surface == SURFACE_X_MAX)
    return cell_widths_y[iy] * cell_widths_z[iz];
  else if (surface == SURFACE_Y_MIN || surface == SURFACE_Y_MAX)
    return cell_widths_x[ix] * cell_widths_z[iz];
  else
    return cell_widths_x[ix] * cell_widths_y[iy];
}


/**
 * @brief Get the ID of the Mesh cell next to given Mesh cell.
 * @param cell index of the current CMFD cell
 * @param surface_id id of the surface between the current cell and the next
 * @param global work at the global (all domains together) level
 * @param neighbor give cell in neighboring domain
 * @return neighboring CMFD cell ID
 */
__device__ int getCellNext(int cell, int surface_id, bool global, bool neighbor) {

  int cell_next = -1;

  int x, y, z;
  int nx, ny, nz;
  x = (cell % (local_num_x * local_num_y)) % local_num_x;
  y = (cell % (local_num_x * local_num_y)) / local_num_x;
  z = cell / (local_num_x * local_num_y);
  nx = local_num_x;
  ny = local_num_y;
  nz = local_num_z;
  //TODO Domain decomposition
  int _num_x = local_num_x;
  int _num_y = local_num_y;
  int _num_z = local_num_z;

  /* Find the cell on the other side of the surface */
  if (surface_id == SURFACE_X_MIN) {
    if (x != 0)
      cell_next = cell - 1;
    else if (boundaries[SURFACE_X_MIN] == PERIODIC)
      cell_next = cell + (_num_x-1);
  }

  else if (surface_id == SURFACE_Y_MIN) {
    if (y != 0)
      cell_next = cell - nx;
    else if (boundaries[SURFACE_Y_MIN] == PERIODIC)
      cell_next = cell + _num_x*(_num_y-1);
  }

  else if (surface_id == SURFACE_Z_MIN) {
    if (z != 0)
      cell_next = cell - nx*ny;
    else if (boundaries[SURFACE_Z_MIN] == PERIODIC)
      cell_next = cell + _num_x*_num_y*(_num_z-1);
  }

  else if (surface_id == SURFACE_X_MAX) {
    if (x != nx - 1)
      cell_next = cell + 1;
    else if (boundaries[SURFACE_X_MAX] == PERIODIC)
      cell_next = cell - (_num_x-1);
  }

  else if (surface_id == SURFACE_Y_MAX) {
    if (y != ny - 1)
      cell_next = cell + nx;
    else if (boundaries[SURFACE_Y_MAX] == PERIODIC)
      cell_next = cell - _num_x*(_num_y-1);
  }

  else if (surface_id == SURFACE_Z_MAX) {
    if (z != nz - 1)
      cell_next = cell + nx*ny;
    else if (boundaries[SURFACE_Z_MAX] == PERIODIC)
      cell_next = cell - _num_x*_num_y*(_num_z-1);
  }

  return cell_next;
}


/**
 * @brief Returns the width of the surface perpendicular to a given surface
 * @param surface A surface index, from 0 to NUM_FACES - 1
 * @param global_ind The CMFD cell global index
 * @return The perpendicular surface width
 */
__device__ CMFD_PRECISION getPerpendicularSurfaceWidth(int surface,
                                            int global_ind,
                                            double* cell_widths_x,
                                            double* cell_widths_y,
                                            double* cell_widths_z) {

  //TODO Domain decomposition
  int _num_x = local_num_x;
  int _num_y = local_num_y;

  int ix = global_ind % _num_x;
  int iy = (global_ind % (_num_x * _num_y)) / _num_x;
  int iz = global_ind / (_num_x * _num_y);

  if (surface == SURFACE_X_MIN || surface == SURFACE_X_MAX)
    return cell_widths_x[ix];
  else if (surface == SURFACE_Y_MIN || surface == SURFACE_Y_MAX)
    return cell_widths_y[iy];
  else
    return cell_widths_z[iz];
}


/**
 * @brief Compute Larsen's effective diffusion coefficient correction factor.
 * @details By conserving reaction and leakage rates within cells, CMFD
 *          guarantees preservation of area-averaged scalar fluxes and net
 *          surface currents from the MOC fixed source iteration if the CMFD
 *          equations can be converged. However, when the MOC mesh cell size
 *          becomes significantly larger than the neutron mean free path in that
 *          cell, the step characteristics no longer preserve the linear
 *          infinite medium solution to the transport equation. While the
 *          surface diffusion coefficient correction term in CMFD is guaranteed
 *          to preserve reaction rates and surface net currents for any choice
 *          of diffusion coefficient, convergence (and convergence rate) of the
 *          nonlinear iteration acceleration of CMFD is affected by the choice
 *          of diffusion coefficient. All flat source methods, when applied for
 *          thick optical meshes, artificially distribute neutrons in space.
 *          This is the reason that Larsen's effective diffusion coefficient is
 *          useful in ensuring that the CMFD acceleration equations have a
 *          diffusion coefficient (on the flux gradient term) that is
 *          consistent, not with the physical transport problem, but with the
 *          transport problem that is being accelerated by the CMFD equations.
 *          Larsen's effective diffusion coefficient is precisely this term in
 *          the one-dimensional limit. The following publications provide
 *          further background on how this term is derived and used:
 *
 *            [1] E. Larsen, "Infinite Medium Solutions to the transport
 *                equation, Sn discretization schemes, and the diffusion
 *                approximation", M&C 2001.
 *            [2] S. Shaner, "Transient Method of Characteristics via the
 *                Adiabatic, Theta, and Multigrid Amplitude Function Methods",
 *                Masters Thesis, MIT 2014.
 * @param dif_coef Diffusion coefficient before applying correction factor
 * @param delta Width of the cell in the direction of interest
 * @return The diffusion coefficient correction factor
 */
__device__ CMFD_PRECISION computeLarsensEDCFactor(CMFD_PRECISION dif_coef,
                                       CMFD_PRECISION delta) {

  /* Initialize variables */
  CMFD_PRECISION alpha, mu, expon;
  double rho = 0.0;

  /* Loop over azimuthal angles */
  for (int a=0; a < num_azim/2; a++) {

    CMFD_PRECISION wa = azim_weights[a];

    /* Loop over polar angles */
    for (int p = 0; p < num_polar/2; p++) {
      mu = sqrt(1.0 - pow(sin_thetas(a,p), 2));
      expon = exp(-delta / (3 * dif_coef * mu));
      alpha = (1 + expon) / (1 - expon) - 2 * (3 * dif_coef * mu) / delta;
      rho += 2.0 * mu * polar_weights(a,p) * wa * alpha;
    }
  }

  /* Compute the correction factor */
  CMFD_PRECISION correction = 1.0 + delta * rho / (2 * dif_coef);

  return correction;
}


/**
 * @brief Computes the diffusion coefficient for a given CMFD cell and CMFD
 *        energy group.
 * @details This method computes the diffusion coefficient for a CMFD cell and
 *          CMFD energy group by spatially collapsing the total/transport xs
 *          in each FSR contained within the CMFD cell and then energy
 *          collapsing the diffusion coefficient (\f$1 / (3 * \Sigma_t)\f$) for
 *          all MOC groups in the given CMFD energy group.
 * @param cmfd_cell A CMFD cell
 * @param group A CMFD energy group
 * @return The diffusion coefficient
 */
__device__ CMFD_PRECISION getDiffusionCoefficient(int cmfd_cell, int group,
                                       CMFD_PRECISION* diffusion_tally,
                                       CMFD_PRECISION* reaction_tally) {
  return diffusion_tally(cmfd_cell,group) / reaction_tally(cmfd_cell,group);
}


/**
 * @brief Compute the surface diffusion coefficient for a given CMFD cell,
 *        cell surface, and group.
 * @details This method uses finite differencing to compute the surface
 *          diffusion coefficient (\f$ \hat{D} \f$) or surface diffusion
 *          coefficient correction (\f$ \tilde{D} \f$) for a given CMFD cell,
 *          cell surface, and CMFD energy group. If the MOC iteration is zero,
 *          (\f$ \tilde{D} \f$) is returned as zero. Since (\f$ \hat{D} \f$) and
 *          (\f$ \tilde{D} \f$) are dependent on each other, they must be
 *          computed together.
 * @param cmfd_cell A CMFD cell
 * @param surface A surface of the CMFD cell
 * @param group A CMFD energy group
 * @param dif_surf the surface diffusion coefficient \f$ \hat{D} \f$
 * @param dif_surf_corr the correction diffusion coefficient \f$ \tilde{D} \f$
 */
__device__ void getSurfaceDiffusionCoefficient(int cmfd_cell, int surface,
                                    int group,
                                    CMFD_PRECISION* old_flux,
                                    CMFD_PRECISION* surface_currents,
                                    CMFD_PRECISION* diffusion_tally,
                                    CMFD_PRECISION* reaction_tally,
                                    CMFD_PRECISION& dif_surf,
                                    CMFD_PRECISION& dif_surf_corr,
                                    CMFD_PRECISION* old_dif_surf_corr,
                                    double* cell_widths_x,
                                    double* cell_widths_y,
                                    double* cell_widths_z,
                                    bool old_dif_surf_valid,
                                    int moc_iteration) {

  FP_PRECISION current, current_out, current_in;
  CMFD_PRECISION flux_next;

  /* Get diffusivity and flux for Mesh cell */
  CMFD_PRECISION dif_coef = getDiffusionCoefficient(cmfd_cell, group, diffusion_tally, reaction_tally);
  int global_cmfd_cell = cmfd_cell; //getGlobalCMFDCell(cmfd_cell);
  int global_cmfd_cell_next = getCellNext(global_cmfd_cell, surface, false, false);
  CMFD_PRECISION flux = old_flux(cmfd_cell, group);
  CMFD_PRECISION delta_interface = getSurfaceWidth(surface, global_cmfd_cell,
       cell_widths_x, cell_widths_y, cell_widths_z);
  CMFD_PRECISION delta = getPerpendicularSurfaceWidth(surface, global_cmfd_cell,
       cell_widths_x, cell_widths_y, cell_widths_z);

  CMFD_PRECISION delta_next = 0.0;
  if (global_cmfd_cell_next != -1)
    delta_next = getPerpendicularSurfaceWidth(surface, global_cmfd_cell_next,
         cell_widths_x, cell_widths_y, cell_widths_z);

  int sense = 2 * (surface % 2) - 1;

  /* Correct the diffusion coefficient with Larsen's effective diffusion
   * coefficient correction factor */
  if (!linear_source)
    dif_coef *= computeLarsensEDCFactor(dif_coef, delta);

  /* If surface is on a boundary with REFLECTIVE or VACUUM BCs, choose
   * appropriate BC */
  if (global_cmfd_cell_next == -1) {

    /* REFLECTIVE BC */
    if (boundaries[surface] == REFLECTIVE) {
      dif_surf = 0.0;
      dif_surf_corr = 0.0;
    }

    /* VACUUM BC */
    else if (boundaries[surface] == VACUUM) {

      /* Compute the surface-averaged current leaving the cell */
      current_out = sense * surface_currents(cmfd_cell, surface, group) /
           delta_interface;

      /* Set the surface diffusion coefficient and MOC correction */
      dif_surf =  2 * dif_coef / delta / (1 + 4 * dif_coef / delta);
      dif_surf_corr = (sense * dif_surf * flux - current_out) / flux;
    }
  }

  /* If surface is an interface or PERIODIC BC, use finite differencing */
  else {

    /* Get the surface index for the surface in the neighboring cell */
    int surface_next = (surface + NUM_FACES / 2) % NUM_FACES;

    /* Get the outward current on surface */
    current_out = surface_currents(cmfd_cell, surface, group);

    /* Set diffusion coefficient and flux for the neighboring cell */
    int cmfd_cell_next = global_cmfd_cell_next; //getLocalCMFDCell(global_cmfd_cell_next);
    CMFD_PRECISION dif_coef_next;

    dif_coef_next = getDiffusionCoefficient(cmfd_cell_next, group, diffusion_tally, reaction_tally);
    flux_next = old_flux(cmfd_cell_next, group);

    /* Get the inward current on the surface */
    current_in = surface_currents(cmfd_cell_next, surface_next, group);

    /* Correct the diffusion coefficient with Larsen's effective diffusion
     * coefficient correction factor */
    if (!linear_source)
      dif_coef_next *= computeLarsensEDCFactor(dif_coef_next, delta_next);

    /* Compute the surface diffusion coefficient */
    dif_surf = 2.0 * dif_coef * dif_coef_next
               / (delta_next * dif_coef + delta * dif_coef_next);

    /* Compute the surface-averaged net current across the surface */
    current = sense * (current_out - current_in) / delta_interface;

    /* Compute the surface diffusion coefficient correction */
    dif_surf_corr = -(sense * dif_surf * (flux_next - flux) + current)
        / (flux_next + flux);

    /* Flux limiting condition */
    if (_flux_limiting && moc_iteration > 0) {
      double ratio = dif_surf_corr / dif_surf;
      if (std::abs(ratio) > 1.0) {

        if (sense * current > 0.0)
          dif_surf = std::abs(current / (2.0*flux));
        else
          dif_surf = std::abs(current / (2.0*flux_next));

        dif_surf_corr = -(sense * dif_surf * (flux_next - flux) + current)
                        / (flux_next + flux);

        /* Make sure diffusion coefficient is larger than the corrected one,
           to floating point precision */
        dif_surf = fmaxf(dif_surf, std::abs(dif_surf_corr));
      }
    }
  }
  /* Weight the old and new corrected diffusion coefficients by the
     relaxation factor */
  if (old_dif_surf_valid) {
    CMFD_PRECISION prev_dif_surf_corr = old_dif_surf_corr(cmfd_cell, surface, group);
    dif_surf_corr = relaxation_factor * dif_surf_corr +
        (1.0 - relaxation_factor) * prev_dif_surf_corr;
  }

  /* If it is the first MOC iteration, solve the straight diffusion problem
   * with no MOC correction */
  if (moc_iteration == 0)
    dif_surf_corr = 0.0;
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
 * @param materials a pointer to the array of CMFD materials
 * @param FSR_materials a pointer to the array of FSR materials
 * @param old_flux initial CMFD flux, set to the homogenized FSR flux as a guess
 * @param FSR_fluxes fluxes in the source regions after the transport sweep
 * @param volumes a pointer to an array of CMFD cell volumes
 * @param FSR_volumes a pointer to an array of FSR volumes
 * @param volume_tally a pointer to a 1D array of CMFD volumes
 * @param reaction_tally a pointer to a 1D array of reaction rates in CMFD cells
 * @param diffusion_tally 1D array (pointer) of transport rates in CMFD cells
 * @param cmfd_cells_fsrs_index a 1D array indexing into cmfd_cells_fsrs
 * @param cmfd_cells_fsrs a 1D array of the FSR ids in each CMFD cell
 * @param tid_offset the offset for each thread
 * @param tid_max the upper bound on the CMFD cell number
 */
__global__ void collapseXSOnDevice(dev_material* materials,
                                   dev_material* FSR_materials,
                                   FP_PRECISION* old_flux,
                                   FP_PRECISION* FSR_fluxes,
                                   FP_PRECISION* volumes,
                                   FP_PRECISION* FSR_volumes,
                                   FP_PRECISION* volume_tally,
                                   FP_PRECISION* reaction_tally,
                                   FP_PRECISION* diffusion_tally,
                                   long* cmfd_cells_fsrs_index,
                                   long* cmfd_cells_fsrs,
                                   long tid_offset,
                                   long tid_max) {

  /* Get CMFD cell index */
  int tid = tid_offset + threadIdx.x + blockIdx.x * blockDim.x;
  //NOTE The chi tally is the only part that currently precludes including
  // CMFD groups in the loops

  /* Allocate a buffer for accumulating variables */
  extern __shared__ CMFD_PRECISION buffer[];

  /* Initialize variables for FSR properties */
  FP_PRECISION volume, flux;
  FP_PRECISION tot, nu_fis, chi;
  FP_PRECISION* scat;

  /* Get buffers for prodction tallies */
  CMFD_PRECISION* scat_tally = &buffer[0];
  CMFD_PRECISION* chi_tally = &buffer[num_cmfd_groups];

  /* Pointers to material objects */
  dev_material* fsr_material;
  dev_material* cell_material;

  while (tid < tid_max) {

    int i = tid;
    cell_material = &materials[i];

    /* Zero group-wise fission buffer */
    double neutron_production_tally = 0.0;
    for (int e = 0; e < num_cmfd_groups; e++)
      chi_tally[e] = 0.0;

    /* Loop over FSRs in CMFD cell */
    //NOTE Variable length 'for loop' is not ideal for the GPU
    for (int r=cmfd_cells_fsrs_index[i]; r<cmfd_cells_fsrs_index[i+1]; r++) {

      /* Get fsr id from 1D vector of all FSRs in cmfd cells */
      int fsr = cmfd_cells_fsrs[r];

      fsr_material = &FSR_materials[fsr];
      volume = FSR_volumes[fsr];

      /* Calculate total neutron production in the FSR */
      double neutron_production = 0.0;
      for (int h = 0; h < num_moc_groups; h++)
        neutron_production += fsr_material->_nu_sigma_f[h] *
             FSR_fluxes[fsr*num_moc_groups+h] * volume;

      /* Calculate contribution to all CMFD groups */
      for (int e=0; e < num_cmfd_groups; e++) {
        chi = 0;
        for (int h = group_indices[e]; h < group_indices[e + 1]; h++)
          chi += fsr_material->_chi[h];

        chi_tally[e] += chi * neutron_production;
      }

      /* Add to total neutron production within the CMFD cell */
      neutron_production_tally += neutron_production;
    }

    /* Set chi */
    if (fabs(neutron_production_tally) > 0) {

      /* Calculate group-wise fission contributions */
      for (int e=0; e < num_cmfd_groups; e++)
        cell_material->_chi[e] = chi_tally[e] / neutron_production_tally;
    }
    else {
      /* Calculate group-wise chi to zero */
      for (int e=0; e < num_cmfd_groups; e++)
        cell_material->_chi[e] = 0.0;
    }

    /* Loop over CMFD coarse energy groups */
    for (int e = 0; e < num_cmfd_groups; e++) {

      /* Zero tallies for this group */
      double nu_fission_tally = 0.0;
      double total_tally = 0.0;

      diffusion_tally(i,e) = 0.0;
      reaction_tally(i,e) = 0.0;
      volume_tally(i,e) = 0.0;

      /* Zero each group-to-group scattering tally */
      for (int g = 0; g < num_cmfd_groups; g++)
        scat_tally[g] = 0.0;

      /* Loop over MOC energy groups within this CMFD coarse group */
      for (int h = group_indices[e]; h < group_indices[e+1]; h++) {

        /* Reset volume tally for this MOC group */
        volume_tally(i,e) = 0.0;
        double rxn_tally_group = 0.0;
        double trans_tally_group = 0.0;

        /* Loop over FSRs in CMFD cell */
        for (int r=cmfd_cells_fsrs_index[i]; r<cmfd_cells_fsrs_index[i+1]; r++) {

          /* Get fsr id from 1D vector of all FSRs in cmfd cells */
          int fsr = cmfd_cells_fsrs[r];

          /* Gets FSR volume, material, and cross sections */
          fsr_material = &FSR_materials[fsr];
          volume = FSR_volumes[fsr];
          scat = fsr_material->_sigma_s;
          flux = FSR_fluxes[fsr*num_moc_groups+h];
          tot = fsr_material->_sigma_t[h];
          nu_fis = fsr_material->_nu_sigma_f[h];

         /* Increment tallies for this group */
         total_tally += tot * flux * volume;
         nu_fission_tally += nu_fis * flux * volume;
         reaction_tally(i,e) += flux * volume;
         volume_tally(i,e) += volume;

         /* Increment diffusion MOC group-wise tallies */
         rxn_tally_group += flux * volume;
         trans_tally_group += tot * flux * volume;

         /* Scattering tallies */
         for (int g = 0; g < num_moc_groups; g++) {
           scat_tally[cmfd_group_map[g]] +=
               scat[g*num_moc_groups+h] * flux * volume;
         }
       }

       /* Condense diffusion coefficient (with homogenized transport XS) */
       if (fabs(trans_tally_group) > fabs(rxn_tally_group) * FLT_EPSILON) {
         CMFD_PRECISION flux_avg_sigma_t = trans_tally_group /
             rxn_tally_group;
         diffusion_tally(i,e) += rxn_tally_group /
             (3.0 * flux_avg_sigma_t);
       }
     }

     /* Save cross-sections to material */
     double rxn_tally = reaction_tally(i,e);

     if (rxn_tally <= 0) {
       int cell = i; //FIXME Domain decomposition: getGlobalCMFDCell(i);
       printf("WARNING: Negative or zero reaction tally calculated in CMFD cell"
              " %d in CMFD group %d : %e", cell, e + 1, rxn_tally);

       /* Set all cross sections to be 1 */
       rxn_tally = ZERO_SIGMA_T;
       reaction_tally(i,e) = ZERO_SIGMA_T;
       diffusion_tally(i,e) = ZERO_SIGMA_T;
       total_tally = ZERO_SIGMA_T;
       if (nu_fission_tally != 0)
         nu_fission_tally = ZERO_SIGMA_T;

       /* Avoid excessive downscatter */
       for (int g = 0; g < num_cmfd_groups; g++)
         scat_tally[g] = 0;
     }

     cell_material->_sigma_t[e] = total_tally / rxn_tally;
     cell_material->_nu_sigma_f[e] = nu_fission_tally / rxn_tally;

     /* Set scattering xs */
     for (int g = 0; g < num_cmfd_groups; g++) {
       cell_material->_sigma_s[g*num_cmfd_groups+e] = scat_tally[g] / rxn_tally;
     }
   }

    /* Loop over CMFD coarse energy groups */
    for (int e = 0; e < num_cmfd_groups; e++) {

      /* Load tallies at this cell and energy group */
      double vol_tally = volume_tally(i,e);
      double rxn_tally = reaction_tally(i,e);
      old_flux(i,e) = rxn_tally / vol_tally;

      /* Set the Mesh cell properties with the tallies */
      volumes[i] = vol_tally;
    }

    /* Update tid for this thread */
    tid += blockIdx.x * blockDim.x;
  }
}


/**
 * @brief Compute the length of each row of the CMFD matrices.
 * @details This method examines the CMFD materials to determine whether
 *          there is going to be a nonzero coefficient in the matrix.
 * @param materials the CMFD materials on the GPU
 * @param A_sizes the size of each row of A
 * @param M_sizes the size of each row of M
 * @param solve_3D whether this is a 3D or 2D case
 * @param tid_offset the offset for each thread
 * @param tid_max the upper bound on the CMFD cell number * group number
 */
__global__ void computeMatricesSizeOnDevice(dev_material* materials,
                                            int* A_sizes, int* M_sizes,
                                            bool solve_3D,
                                            int tid_offset,
                                            int tid_max) {

  int tid = tid_offset + threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < tid_max) {

    /* Get CMFD cell index and energy group */
    int i = tid / num_cmfd_groups;
    int e = tid % num_cmfd_groups;

    int num_scatters = 0;
    int num_fissions = 0;

    /* Look at material cross sections to determine in-group scatters */
    for (int g=0; g<num_cmfd_groups; g++) {
      num_scatters += ((&materials[i])->_sigma_s[g*num_cmfd_groups + e] != 0) *
           (g != e);
      num_fissions += ((&materials[i])->_nu_sigma_f[g] > 0);
    }
    num_fissions *= ((&materials[i])->_chi[e] > 0);

    /* Streaming, reaction and scatter terms */
    A_sizes[i*num_cmfd_groups + e] = 5 + 2 * solve_3D + num_scatters;
    M_sizes[i*num_cmfd_groups + e] = (num_fissions > 0);

    /* Update tid for this thread */
    tid += blockIdx.x * blockDim.x;
  }
}


__global__ void constructMatricesOnDevice(dev_material* materials,
                                          CMFD_PRECISION* volumes,
                                          CMFD_PRECISION* old_flux,
                                          CMFD_PRECISION* surface_currents,
                                          CMFD_PRECISION* reaction_tally,
                                          CMFD_PRECISION* diffusion_tally,
                                          int* dA_csrOffsets,
                                          int* dA_columns,
                                          CMFD_PRECISION* dA_values,
                                          int* dM_csrOffsets,
                                          int* dM_columns,
                                          CMFD_PRECISION* dM_values,
                                          CMFD_PRECISION* old_dif_surf_corr,
                                          double* cell_widths_x,
                                          double* cell_widths_y,
                                          double* cell_widths_z,
                                          int moc_iteration,
                                          bool old_dif_surf_valid,
                                          int tid_offset,
                                          int tid_max) {

  int tid = tid_offset + threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < tid_max) {

    /* Get CMFD cell index and energy group */
    int i = tid / num_cmfd_groups;
    int e = tid % num_cmfd_groups;

    /* Index in CSR vectors */
    int row_index_A = 0;
    int row_index_M = 0;
    int diagonal_index;

    FP_PRECISION removal, value, volume, delta;
    CMFD_PRECISION dif_surf, dif_surf_corr;
    int sense;
    dev_material* material;

    //TODO Domain decomposition
    int global_ind = i;

    material = &materials[i];
    volume = volumes[i];

    /* Net removal term */
    removal = material->_sigma_t[e] * volume;

    /* Scattering gain from all groups */
    for (int g = 0; g < num_cmfd_groups; g++) {
      value = - material->_sigma_s[g*num_cmfd_groups + e] * volume;
      if (std::abs(value) > FLT_EPSILON || g == e) {
        dA_columns[dA_csrOffsets(i,e) + row_index_A] = i*num_cmfd_groups + g;
        dA_values[dA_csrOffsets(i,e) + row_index_A++] += value;
      }
      if (g==e) {
        dA_columns[dA_csrOffsets(i,e) + row_index_A-1] = i*num_cmfd_groups + e;
        dA_values[dA_csrOffsets(i,e) + row_index_A-1] += removal;
        diagonal_index = row_index_A - 1;
      }
    }

    /* Streaming to neighboring cells */
    for (int s = 0; s < NUM_FACES; s++) {

      sense = 2 * (s % 2) - 1;
      delta = getSurfaceWidth(s, global_ind, cell_widths_x, cell_widths_y,
                              cell_widths_z);

      /* Set transport term on diagonal */
      getSurfaceDiffusionCoefficient(i, s, e, old_flux, surface_currents,
           diffusion_tally, reaction_tally, dif_surf, dif_surf_corr, old_dif_surf_corr,
           cell_widths_x, cell_widths_y, cell_widths_z, old_dif_surf_valid,
           moc_iteration);

      /* Record the corrected diffusion coefficient */
      old_dif_surf_corr(i,s,e) = dif_surf_corr;

      /* Set the diagonal term */
      value = (dif_surf - sense * dif_surf_corr) * delta;
      dA_columns[dA_csrOffsets(i,e) + diagonal_index] = i*num_cmfd_groups + e;
      dA_values[dA_csrOffsets(i,e) + diagonal_index] += value;

      /* Set the off diagonal term */
      int i_next = getCellNext(i, s, false, false);
      if (i_next != -1) {
        value = - (dif_surf + sense * dif_surf_corr) * delta;
        dA_columns[dA_csrOffsets(i,e) + row_index_A] = i_next*num_cmfd_groups + e;
        dA_values[dA_csrOffsets(i,e) + row_index_A++] += value;
      }
    }

    /* Fission source term */
    for (int g = 0; g < num_cmfd_groups; g++) {
      value = material->_chi[e] * material->_nu_sigma_f[g] * volume;
      if (std::abs(value) > FLT_EPSILON) {
        dM_columns[dM_csrOffsets(i,e) + row_index_M++] = i*num_cmfd_groups + g;
        dM_values[dM_csrOffsets(i,e) + row_index_M] += value;
      }
    }

    /* Update tid for this thread */
    tid += blockIdx.x * blockDim.x;
  }
}


/**
 * @brief Update the MOC flux in each FSR.
 * @details This method uses the condensed flux from the last MOC transport
 *          sweep and the converged flux from the eigenvalue problem to
 *          update the MOC flux in each FSR.
 * @param old_flux initial CMFD flux, from flux-volume weighting of FSR fluxes
 * @param new_flux computed CMFD flux
 * @param FSR_fluxes source region fluxes
 * @param cmfd_cells_fsrs_index a 1D array indexing into cmfd_cells_fsrs
 * @param cmfd_cells_fsrs a 1D array of the FSR ids in each CMFD cell
 * @param moc_iteration current iteration number of the MOC solver
 * @param num_unbounded_iterations num. of iterations with no update ratio bound
 * @param tid_offset the offset for each thread
 * @param tid_max the upper bound on the CMFD cell number * group number
 */
__global__ void updateMOCFluxOnDevice(CMFD_PRECISION* old_flux,
                                      CMFD_PRECISION* new_flux,
                                      FP_PRECISION* FSR_fluxes,
                                      long* cmfd_cells_fsrs_index,
                                      long* cmfd_cells_fsrs,
                                      int moc_iteration,
                                      int num_unbounded_iterations,
                                      int tid_offset,
                                      int tid_max) {

  /* Get CMFD cell index and energy group */
  int tid = tid_offset + threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < tid_max) {
    /* Loop over FSRs in CMFD cell */
    //NOTE Variable length for loop is not ideal for the GPU
    //Kernel could act on FSR id and MOC energy groups too
    //TODO If performance requires it, switch
    int i = tid / num_cmfd_groups;
    int e = tid % num_cmfd_groups;

    for (int r=cmfd_cells_fsrs_index[i]; r<cmfd_cells_fsrs_index[i+1]; r++) {

      /* Get fsr id from 1D vector of all FSRs in cmfd cells */
      int fsr = cmfd_cells_fsrs[r];

      /* Get the update ratio */
      CMFD_PRECISION update_ratio = 1;
      if (old_flux(i,e) > FLT_EPSILON)
        update_ratio = new_flux(i,e) / old_flux(i,e);

      /* Limit the update ratio for stability purposes. For very low flux
         regions, update ratio may be left unrestricted a few iterations*/
      if (moc_iteration > num_unbounded_iterations)
        if (update_ratio > 20.0)
          update_ratio = 20.0;
        else if (update_ratio < 0.05)
          update_ratio = 0.05;

      for (int h = group_indices[e]; h < group_indices[e + 1]; h++) {

        /* Update FSR flux using ratio of old and new CMFD flux */
        FSR_fluxes[fsr*num_moc_groups + h] *= update_ratio;

        printf("Updating flux in FSR: %d, cell: %d, MOC group: "
          "%d, CMFD group: %d, ratio: %f", fsr ,i, h, e, update_ratio);
      }
    }

    /* Update tid for this thread */
    tid += blockIdx.x * blockDim.x;
  }
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

  /* The default number of thread blocks and threads per thread block */
  _B = _NUM_GPU_THREAD_BLOCKS;
  _T = _NUM_GPU_THREADS_IN_BLOCK;

  _materials = NULL;
  _FSR_materials = NULL;
  _volumes = NULL;
  _FSR_volumes = NULL;

  _gpu_cmfd = true;
}


/**
 * @brief Destructor deletes arrays of A and M row insertion arrays.
 */
GPUCmfd::~GPUCmfd() {

  cusparseDestroySpMat(_A);
  cusparseDestroySpMat(_M);
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
void GPUCmfd::collapseXS() {

  log_printf(INFO, "Collapsing cross-sections onto CMFD mesh on GPU...");

  /* Check to see that CMFD tallies have been allocated */
  if (!_tallies_allocated)
    log_printf(ERROR, "Tallies need to be allocated before collapsing "
               "cross-sections");

  //FIXME Edge and corner current splits are not modeled

  /* Pass size of CMFD arrays to device */
  cudaMemcpyToSymbol(num_moc_groups, &_num_moc_groups, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(num_cmfd_groups, &_num_cmfd_groups, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(linear_source, &_linear_source, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  getLastCudaError();

  //TODO Move to its own initialization routine
  /* MOC to CMFD group mapping vectors */
  cudaMemcpyToSymbol(group_indices, _group_indices, _num_cmfd_groups *
                     sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cmfd_group_map, _group_indices_map, _num_moc_groups *
                     sizeof(int), 0, cudaMemcpyHostToDevice);
  getLastCudaError();

  //TODO Move to its own initialization routine
  /* Source region to CMFD cells mapping vectors */
  int num_cmfd_cells = _local_num_x * _local_num_y * _local_num_z;
  _cmfd_cells_fsrs_index.resize(num_cmfd_cells);
  _cmfd_cells_fsrs.resize(_num_FSRs);

  /* Loop over CMFD cells */
  int fsr = 0;
  std::vector<long>::iterator iter;
  for (int i = 0; i < _local_num_x * _local_num_y * _local_num_z; i++) {

    /* Number of FSRs in each CMFD cell */
    _cmfd_cells_fsrs_index[i] = _cell_fsrs.at(i).size();

    /* Form list of FSRs in each cell in a 1D vector */
    for (iter = _cell_fsrs.at(i).begin();
         iter != _cell_fsrs.at(i).end(); ++iter)
      _cmfd_cells_fsrs[fsr++] = *iter;
  }

  /* Get device pointer to the Thrust vectors */
  dev_material* materials = thrust::raw_pointer_cast(&_materials[0]);
  FP_PRECISION* old_flux = thrust::raw_pointer_cast(&_old_flux[0]);
  FP_PRECISION* volumes = thrust::raw_pointer_cast(&_volumes[0]);
  dev_material* FSR_materials = thrust::raw_pointer_cast(&_FSR_materials[0]);
  FP_PRECISION* FSR_fluxes = thrust::raw_pointer_cast(&_FSR_fluxes[0]);
  FP_PRECISION* FSR_volumes = thrust::raw_pointer_cast(&_FSR_volumes[0]);
  FP_PRECISION* volume_tally = thrust::raw_pointer_cast(&_volume_tally[0]);
  FP_PRECISION* reaction_tally = thrust::raw_pointer_cast(&_reaction_tally[0]);
  FP_PRECISION* diffusion_tally = thrust::raw_pointer_cast(&_diffusion_tally[0]);
  long* cmfd_cells_fsrs_index = thrust::raw_pointer_cast(
       &_cmfd_cells_fsrs_index[0]);
  long* cmfd_cells_fsrs = thrust::raw_pointer_cast(
       &_cmfd_cells_fsrs[0]);

  int shared_mem = _T * 2 * _num_cmfd_groups * sizeof(CMFD_PRECISION);

  /* Print advice on number of blocks/threads */
  if (get_log_level() == INFO) {
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       collapseXSOnDevice, 0, num_cmfd_cells);
    log_printf(INFO, "Kernel CollapseXS: suggested minGridSize %d blockSize %d",
               minGridSize, blockSize);
  }

  /* Loop on CMFD cells */
  //NOTE Diffusion coefficient makes looping on energy groups difficult
  collapseXSOnDevice<<<_B, _T, shared_mem>>>(materials, FSR_materials,
                                             old_flux, FSR_fluxes, volumes,
                                             FSR_volumes, volume_tally,
                                             reaction_tally, diffusion_tally,
                                             cmfd_cells_fsrs_index,
                                             cmfd_cells_fsrs, 0,
                                             num_cmfd_cells);
}


/**
 * @brief Construct the loss + streaming matrix (A) and the fission gain
 *         matrix (M) in preparation for solving the eigenvalue problem.
 * @details This method loops over all mesh cells and energy groups and
 *          accumulates the iteraction and streaming terms into their
 *          appropriate positions in the loss + streaming matrix and
 *          fission gain matrix.
 */
void GPUCmfd::constructMatrices() {

  log_printf(INFO, "Constructing matrices on GPU...");

  size_t size = getNumCells() * _num_cmfd_groups;
  thrust::device_vector<int> dev_A_sizes(size);
  thrust::device_vector<int> dev_M_sizes(size);
  int* A_sizes = thrust::raw_pointer_cast(&dev_A_sizes[0]);
  int* M_sizes = thrust::raw_pointer_cast(&dev_M_sizes[0]);
  dev_material* materials = thrust::raw_pointer_cast(&_materials[0]);

  /* Scan all CMFD cells to obtain the size of the CMFD sparse matrices */
  computeMatricesSizeOnDevice<<<_B, _T>>>(materials, A_sizes, M_sizes,
                                          _SOLVE_3D, 0, size);
  int A_num_nnz = thrust::reduce(dev_A_sizes.begin(), dev_A_sizes.end());
  int M_num_nnz = thrust::reduce(dev_M_sizes.begin(), dev_M_sizes.end());

  /* Allocate the CMFD sparse matrix vectors */
  thrust::device_vector<int> dev_dA_csrOffsets(size);
  thrust::device_vector<int> dev_dM_csrOffsets(size);
  int* dA_csrOffsets = thrust::raw_pointer_cast(&dev_dA_csrOffsets[0]);
  int* dM_csrOffsets = thrust::raw_pointer_cast(&dev_dM_csrOffsets[0]);
  int   *dA_columns, *dM_columns;
  float *dA_values, *dM_values;
  cudaMalloc((void**) &dA_columns, A_num_nnz * sizeof(int));
  cudaMalloc((void**) &dA_values,  A_num_nnz * sizeof(CMFD_PRECISION));
  cudaMalloc((void**) &dM_columns, M_num_nnz * sizeof(int));
  cudaMalloc((void**) &dM_values,  M_num_nnz * sizeof(CMFD_PRECISION));

  /* Fill the sparse matrix offsets from row sizes */
  thrust::inclusive_scan(dev_A_sizes.begin(), dev_A_sizes.end(),
                         dev_dA_csrOffsets.begin() + 1);
  thrust::inclusive_scan(dev_M_sizes.begin(), dev_M_sizes.end(),
                         dev_dM_csrOffsets.begin() + 1);

  /* Print advice on number of blocks/threads */
  if (get_log_level() == INFO) {
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       constructMatricesOnDevice, 0,
                                       getNumCells() * _num_cmfd_groups);
    log_printf(INFO, "Kernel ConstructMatrix: suggested minGridSize %d "
               "blockSize %d", minGridSize, blockSize);
  }

  /* Get device pointer to the thrust vectors */
  FP_PRECISION* volumes = thrust::raw_pointer_cast(&_volumes[0]);
  FP_PRECISION* old_flux = thrust::raw_pointer_cast(&_old_flux[0]);
  FP_PRECISION* reaction_tally = thrust::raw_pointer_cast(&_reaction_tally[0]);
  FP_PRECISION* diffusion_tally = thrust::raw_pointer_cast(&_diffusion_tally[0]);
  _old_dif_surf_corr.resize(getNumCells() * _num_cmfd_groups * NUM_FACES);
  CMFD_PRECISION* old_dif_surf_corr = thrust::raw_pointer_cast(&_old_dif_surf_corr[0]);
  CMFD_PRECISION* surface_currents = thrust::raw_pointer_cast(&_surface_currents[0]);

  /* Transfer cell widths to device */
  double* cell_widths_x = thrust::raw_pointer_cast(&_dev_cell_widths_x[0]);
  double* cell_widths_y = thrust::raw_pointer_cast(&_dev_cell_widths_y[0]);
  double* cell_widths_z = thrust::raw_pointer_cast(&_dev_cell_widths_z[0]);
  cudaMemcpyToSymbol(cell_widths_x, &_cell_widths_x[0], _cell_widths_x.size() *
                     sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cell_widths_y, &_cell_widths_y[0], _cell_widths_y.size() *
                     sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cell_widths_z, &_cell_widths_z[0], _cell_widths_z.size() *
                     sizeof(double), cudaMemcpyHostToDevice);

  /* Save relaxation factor to constant memory */
  cudaMemcpyToSymbol(relaxation_factor, &_relaxation_factor, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  /* Transfer CMFD mesh size to device */
  cudaMemcpyToSymbol(local_num_x, &_local_num_x, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(local_num_y, &_local_num_y, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(local_num_z, &_local_num_z, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(boundaries, _boundaries, 6 * sizeof(int), 0, cudaMemcpyHostToDevice);

  /* Fill the sparse matrix column and values vectors */
  constructMatricesOnDevice<<<_B, _T>>>(materials, volumes, old_flux,
                                        surface_currents,
                                        reaction_tally, diffusion_tally,
                                        dA_csrOffsets, dA_columns, dA_values,
                                        dM_csrOffsets, dM_columns, dM_values,
                                        old_dif_surf_corr, cell_widths_x,
                                        cell_widths_y, cell_widths_z,
                                        _moc_iteration, _old_dif_surf_valid,
                                        0, getNumCells() * _num_cmfd_groups);

  /* Mark correction diffusion coefficient as valid for relaxation purposes */
  _old_dif_surf_valid = true;

  /* Create the sparse matrices A and M in CSR format */
  cusparseCreateCsr(&_A, size, size, A_num_nnz,
                    dA_csrOffsets, dA_columns, dA_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&_M, size, size, M_num_nnz,
                    dM_csrOffsets, dM_columns, dM_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
}


/**
 * @brief Solve the CMFD eigenvalue problem using cuSPARSE.
 * @details GPU-equivalence of eigenvalueSolve in linalg.cpp
 * @return the K-effective of the CMFD eigenvalue problem
 */
double GPUCmfd::solveEigenvalueProblem() {

  log_printf(INFO, "Computing the Matrix-Vector eigenvalue...");
  double tol = std::max(MIN_LINALG_TOLERANCE, _source_convergence_threshold);

  /* Initialize variables */
  int num_rows = getNumCells() * _num_cmfd_groups;
  int size = num_rows;
  cusparseHandle_t handle = 0;
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  cusparseCreate(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  cusparseDnVecDescr_t d_new_flux, d_new_source, d_old_source;
  double residual, k_eff;
  int iter;

  /* Get pointers to thrust arrays */
  FP_PRECISION* new_flux = thrust::raw_pointer_cast(&_new_flux[0]);
  FP_PRECISION* new_source = thrust::raw_pointer_cast(&_new_source[0]);
  FP_PRECISION* old_source = thrust::raw_pointer_cast(&_old_source[0]);

  /* Copy to on-device dense vector */ //FIXME Needed ?
  cusparseCreateDnVec(&d_new_flux, size, new_flux, CUDA_R_32F);
  cusparseCreateDnVec(&d_new_source, size, new_source, CUDA_R_32F);
  cusparseCreateDnVec(&d_old_source, size, old_source, CUDA_R_32F);

  /* Compute fission source */
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, _M, d_new_flux, &beta, d_new_source, CUDA_R_32F,
               CUSPARSE_MV_ALG_DEFAULT, dBuffer);

  /* Solve power iteration */
  double initial_residual = -1;
  for (iter = 0; iter < MAX_LINALG_POWER_ITERATIONS; iter++) {

    /* Analyse linear system (once?) */
    cusparseScsrsm2_analysis( handle,
                             int                      algo,
                             cusparseOperation_t      transA,
                             cusparseOperation_t      transB,
                             int                      m,
                             int                      nrhs,
                             int                      nnz,
                             const float*             alpha,
                             const cusparseMatDescr_t descrA,
                             const float*             csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             const float*             B,
                             int                      ldb,
                             csrsm2Info_t             info,
                             cusparseSolvePolicy_t    policy,
                             void*                    pBuffer)

    /* Solve X = A^-1 * old_source */

    bool converged = true;

    /* Check for divergence */
    if (!converged)
      return -1.0;

    /* Compute the new source */
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, _M, d_new_flux, &beta, d_new_source, CUDA_R_32F,
                 CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    /* Compute the sum of new sources */
    double new_source_sum = thrust::reduce(_new_source.begin(),
                                           _new_source.end());

    /* Compute and set keff */
    k_eff = new_source_sum / num_rows;

    /* Scale the new source by keff */
    thrust::transform(_new_source.begin(), _new_source.end(), _new_source.begin(),
                      1.0 / k_eff * thrust::placeholders::_1);

    /* Compute the residual */
    //residual = computeRMSE(&new_source, &old_source, true, comm);
    if (iter == 0) {
      initial_residual = residual;
      if (initial_residual < 1e-14)
        initial_residual = 1e-10;
      if (_convergence_data != NULL) {
        _convergence_data->cmfd_res_1 = residual;  ////!!!!
        _convergence_data->linear_iters_1 = _convergence_data->linear_iters_end;
        _convergence_data->linear_res_1 = _convergence_data->linear_res_end;
      }
    }

    /* Copy the new source to the old source */
    //new_source.copyTo(&old_source);

    log_printf(INFO_ONCE, "Matrix-Vector eigenvalue iter: %d, keff: %f, residual: "
               "%3.2e", iter, k_eff, residual);

    /* Check for convergence */
    if ((residual / initial_residual < 0.03 || residual < MIN_LINALG_TOLERANCE)
        && iter > MIN_LINALG_POWER_ITERATIONS) {
      if (_convergence_data != NULL) {
        _convergence_data->cmfd_res_end = residual;
        _convergence_data->cmfd_iters = iter;
      }
      break;
    }
  }

  log_printf(INFO_ONCE, "Matrix-Vector eigenvalue solve iterations: %d", iter);
  if (iter == MAX_LINALG_POWER_ITERATIONS)
    log_printf(ERROR, "Eigenvalue solve failed to converge in %d iterations",
               iter);

  return k_eff;
}

/**
 * @brief Rescale the initial and converged flux arrays.
 * @details The diffusion problem is a generalized eigenvalue problem and
 *          therefore the solution is independent of flux level. This method
 *          rescales the input flux and converged flux to both have an average
 *          fission source of 1.0 in each group in each cell.
 */
void GPUCmfd::rescaleFlux() {

  /* Get pointers to thrust vectors */
  FP_PRECISION* new_flux = thrust::raw_pointer_cast(&_new_flux[0]);
  FP_PRECISION* old_flux = thrust::raw_pointer_cast(&_old_flux[0]);
  FP_PRECISION* new_source = thrust::raw_pointer_cast(&_new_source[0]);
  FP_PRECISION* old_source = thrust::raw_pointer_cast(&_old_source[0]);

  /* Rescale the new and old flux to have an avg source of 1.0 */
  cusparseHandle_t     handle = 0;
  void*  dBuffer    = NULL;
  size_t bufferSize = 0;
  cusparseCreate(&handle);
  float alpha = 1.0f;
  float beta  = 0.0f;
  size_t size = getNumCells() * _num_cmfd_groups;
  cusparseDnVecDescr_t d_new_flux, d_old_flux, d_new_source, d_old_source;

  /* Copy to on-device dense vector */ //FIXME Needed ?
  cusparseCreateDnVec(&d_new_flux, size, new_flux, CUDA_R_32F);
  cusparseCreateDnVec(&d_old_flux, size, old_flux, CUDA_R_32F);
  cusparseCreateDnVec(&d_new_source, size, new_source, CUDA_R_32F);
  cusparseCreateDnVec(&d_old_source, size, old_source, CUDA_R_32F);

  /* Allocate buffer for the matrix vector multiplication */
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, _M, d_new_flux, &beta, d_new_flux, CUDA_R_32F,
                               CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  /* Compute source from M * flux */
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, _M, d_new_flux, &beta, d_new_source, CUDA_R_32F,
               CUSPARSE_MV_ALG_DEFAULT, dBuffer);
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, _M, d_old_flux, &beta, d_old_source, CUDA_R_32F,
               CUSPARSE_MV_ALG_DEFAULT, dBuffer);

  /* Destroy matrix vector descriptors */
  cusparseDestroyDnVec(d_new_flux);
  cusparseDestroyDnVec(d_old_flux);
  cusparseDestroy(handle);

  /* Compute fission source sum */
  CMFD_PRECISION new_source_sum = thrust::reduce(_new_source.begin(),
                                                 _new_source.end());
  CMFD_PRECISION old_source_sum = thrust::reduce(_old_source.begin(),
                                                 _old_source.end());

  /* Rescale vector */
  thrust::transform(_new_flux.begin(), _new_flux.end(), _new_flux.begin(),
                    1.0 / new_source_sum * thrust::placeholders::_1);
  thrust::transform(_old_flux.begin(), _old_flux.end(), _old_flux.begin(),
                    1.0 / old_source_sum * thrust::placeholders::_1);

  // Check result
}


/**
 * @brief Update the MOC flux in each FSR.
 * @details This method uses the condensed flux from the last MOC transport
 *          sweep and the converged flux from the eigenvalue problem to
 *          update the MOC flux in each FSR.
 */
void GPUCmfd::updateMOCFlux() {

  log_printf(INFO, "Updating MOC flux on GPU...");

  FP_PRECISION* new_flux = thrust::raw_pointer_cast(&_new_flux[0]);
  FP_PRECISION* old_flux = thrust::raw_pointer_cast(&_old_flux[0]);
  FP_PRECISION* FSR_fluxes = thrust::raw_pointer_cast(&_FSR_fluxes[0]);
  long* cmfd_cells_fsrs_index = thrust::raw_pointer_cast(
       &_cmfd_cells_fsrs_index[0]);
  long* cmfd_cells_fsrs = thrust::raw_pointer_cast(
       &_cmfd_cells_fsrs[0]);

  updateMOCFluxOnDevice<<<_B, _T>>>(old_flux, new_flux, FSR_fluxes,
                                    cmfd_cells_fsrs_index, cmfd_cells_fsrs,
                                    _moc_iteration, _num_unbounded_iterations,
                                    0, getNumCells() * _num_cmfd_groups);
}


/**
 * @brief Copy surface currents to the CMFD
 * @param surface_currents the vector of surface currents
 */
void GPUCmfd::copySurfaceCurrents(thrust::device_vector<CMFD_PRECISION> surface_currents) {
  _surface_currents = surface_currents;
}


/**
 * @brief Sets the Quadrature object in use by the MOC Solver.
 * @param quadrature a Quadrature object pointer from the Solver
 */
void GPUCmfd::setQuadrature(Quadrature* quadrature) {

  Cmfd::setQuadrature(quadrature);

  /* Copy the number of angles to constant memory */
  cudaMemcpyToSymbol(num_azim, &_num_azim, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(num_polar, &_num_polar, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  /* Copy the azimuthal weights to constant memory on the GPU */
  int num_azim_2 = _quadrature->getNumAzimAngles() / 2;
  FP_PRECISION host_azim_weights[num_azim_2 * _num_polar/2];
  for (int a=0; a < num_azim_2; a++)
    for (int p=0; p < _num_polar/2; p++)
      host_azim_weights[a*_num_polar/2 + p] = _quadrature->getAzimWeight(a);
  cudaMemcpyToSymbol(azim_weights, host_azim_weights,
      num_azim_2 * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
  getLastCudaError();

  /* Copy the polar weights to constant memory on the GPU */
  FP_PRECISION host_polar_weights[num_azim_2 * _num_polar/2];
  for (int a=0; a < num_azim_2; a++)
    for (int p=0; p < _num_polar/2; p++)
      host_polar_weights[a*_num_polar/2 + p] = _quadrature->getPolarWeight(a, p);
  cudaMemcpyToSymbol(polar_weights, host_polar_weights,
      _num_polar/2 * num_azim_2 * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
  getLastCudaError();

  /* Copy the sines of the polar angles */
  auto host_sin_thetas = _quadrature->getSinThetas();
  std::vector<FP_PRECISION> fp_precision_sines(_num_polar/2);
  for (int j=0; j<_num_polar/2; ++j)
    fp_precision_sines[j] = (FP_PRECISION)host_sin_thetas[0][j];
  cudaMemcpyToSymbol(sin_thetas, &fp_precision_sines[0],
                     _num_polar/2 * sizeof(FP_PRECISION), 0,
                     cudaMemcpyHostToDevice);
  getLastCudaError();
}
