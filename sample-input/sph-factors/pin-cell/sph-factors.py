import openmoc
import openmc.openmoc_compatible
import openmc.mgxs

import pickle
import numpy as np
import matplotlib
import time
import os

# Enable Matplotib to work for headless nodes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
os.system("rm plots/*.png")

opts = openmoc.options.Options()
openmoc.log.set_log_level('INFO')
openmoc.set_line_length(120)
start = time.time()

# Plot parameters
font = {'size'   : 10}
matplotlib.rc('font', **font)
title_font = {'size': '24', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}
axis_font = {'size': '18'}

###############################################################################
#                         Simulation parameters
###############################################################################

# Geometry
n_rings_f = 1
n_rings_w = 1
n_sectors_f = 1
n_sectors_w = 2

# Discretization and SPH parameters
num_azim        = 128
azim_spacing    = 0.001 # cv near 0.001
num_polar       = 3
tolerance       = np.array([1e-10, 1e-7, 1e-10])  # push to 1e-10 to get more CV
tol_sph         = 1e-6
sph_iters       = 200
num_threads     = 32
max_iters       = [600, 600]

# Solver main specs
source          = "f"
cmfd_on         = False
if source == "l":
    print("Need to symmetrize LS source")

# Scattering
scattering = ""
#scattering = "_anisotropic/"

# Options for SPH factors
sph_mode       = "eigenvalue"
sph_regions    = "l"
symmetryze_all = True
homog_XS       = False
gen_SPH        = 4
# 0 FVW normalization, outside solver
# 1 fuel-to-water J normalization, outside solver
# 3 outer boudary J+ normalization, in solver
# 4 no normalization

n_cells   = (n_rings_f)*n_sectors_f + (n_rings_w+2)*n_sectors_w

if num_threads < 32 or num_azim < 128 or azim_spacing > 0.001 or tol_sph > 1e-5:
    print("\n\n\n RUNNING IN DEBUG MODE \n\n\n")

###############################################################################
#                         Condense MC XS
###############################################################################

suffix = "_3.1%_enriched_quarter_iso"
folder = "mgxs_libs/"
folder_mc = suffix[1:]+"/"

mgxs_file = 'mgxs_pincell_'+str(n_rings_w)+'wr'+str(n_sectors_w)\
    +'ws_'+str(n_rings_f)+'fr'+str(n_sectors_f)+'fs'+scattering[:-1]\
    +suffix
keff_file = mgxs_file.replace('mgxs', 'keff')+'.pkl'
print("Loading mgxs from "+folder+mgxs_file)

# Initialize OpenMC multi-group cross section library for a pin cell
library = openmc.mgxs.Library.load_from_file(filename=mgxs_file, directory=folder)
n_groups_fine = library.num_groups

coarse_groups = openmc.mgxs.EnergyGroups()
casmo = np.array([0., 0.005e-6, 0.01e-6, 0.015e-6, 0.02e-6, 0.025e-6, 0.03e-6, 0.035e-6,0.042e-6, 0.05e-6,0.058e-6, 0.067e-6,0.08e-6, 0.1e-6, 0.14e-6, 0.18e-6,0.22e-6, 0.25e-6, 0.28e-6, 0.3e-6,0.32e-6,0.35e-6, 0.4e-6, 0.5e-6,0.625e-6, 0.78e-6, 0.85e-6, 0.91e-6,0.95e-6, 0.972e-6, 0.996e-6,1.02e-6,1.045e-6, 1.071e-6, 1.097e-6, 1.123e-6,1.15e-6, 1.3e-6, 1.5e-6, 1.855e-6,2.1e-6, 2.6e-6,3.3e-6, 4.e-6,9.877e-6, 15.968e-6, 27.7e-6, 48.052e-6,75.501e-6, 148.73e-6,367.26001e-6,906.90002e-6,1.4251e-3, 2.2395e-3, 3.5191e-3,5.53e-3, 9.118e-3, 15.03e-3, 24.78e-3,40.85e-3,67.34e-3, 111.e-3, 183.e-3, 302.5e-3, 500.e-3,821.e-3, 1.353, 2.231, 3.679, 6.0655, 20.])*1e6

# Select energy group structure
coarse = casmo
#coarse = np.array([0., 0.03e-6, 0.058e-6, 0.14e-6, 0.28e-6, 0.35e-6, 0.625e-6, 0.972e-6, 1.02e-6, 1.097e-6, 1.15e-6, 1.855e-6, 4.e-6, 9.877e-6, 15.968e-6, 148.73e-6, 5.53e-3, 9.118e-3, 111.e-3, 500.e-3, 821.e-3, 1.353, 2.231, 3.679, 6.0655, 20.])  * 1e6
#coarse = np.array([0., 0.058e-6, 0.14e-6, 0.28e-6, 0.625e-6, 4.e-6, 5.53e-3, 821.e-3, 20.]) * 1e6# 8g
#coarse = np.array([0., 4.e-6, 20.]) * 1e6  # 2 g
##coarse = np.array([0., 20.]) * 1e6         # 1 g
n_groups = len(coarse) - 1
group_match = np.ones(n_groups + 1)

# Find group boundaries matching
k = 0
for g in range(len(coarse)):
    while coarse[g] != casmo[k]:
        k += 1
    group_match[g] = int(k)
group_match = [int(k) for k in group_match]
#print(casmo[group_match])

E_vec      = np.flipud(coarse[1:] + coarse[:-1]) / 2
E_widths   = np.flipud(coarse[1:] - coarse[:-1])
Let_widths = np.flipud(np.log(coarse[1:]) - np.log(coarse[:-1]))/np.log(10)
Let_widths[-1] = (np.log(coarse[1]) + 3)/np.log(10)
coarse_groups.group_edges = np.array(coarse)

if n_groups_fine != n_groups:
    library = library.get_condensed_library(coarse_groups)
n_groups_fine = n_groups

###############################################################################
#                 Get geometry and materials from MGXS lib
###############################################################################

# Get k_eff from pickled file
[keff_mc, statepoint] = pickle.load(open(folder+keff_file, 'rb'))
statepoint = folder_mc + statepoint
sp=openmc.StatePoint(statepoint)

# Merge the cross section and flux tallies (for linear source, the linear flux
# is from a special tally, and all XS are shared
if n_sectors_w == 2 and symmetryze_all:
    print("Merging XS in the quarter pincell to achieve symmetry")
    # Note : CHI cannot be merged, so fuel sectors have to be done someway else
    for c1 in library.domains:
        for c2 in library.domains:
            if (c1 != c2 and c1.name[-1] == "0" and c2.name[-1] == "1"
                and c1.name[1] == c2.name[1]):

                # Merge for all RR
                for XS in library.mgxs_types:
                    xs1 = library.get_mgxs(c1, XS)
                    xs2 = library.get_mgxs(c2, XS)
                    for tally_key in xs1.tallies:

                        xs1.tallies[tally_key]._mean = (xs1.tallies[tally_key]._mean + xs2.tallies[tally_key]._mean) / 2
                        xs2.tallies[tally_key]._mean = xs1.tallies[tally_key]._mean
                        xs1.tallies[tally_key]._std_dev = (xs1.tallies[tally_key]._std_dev + xs2.tallies[tally_key]._std_dev) / 2
                        xs2.tallies[tally_key]._std_dev = xs1.tallies[tally_key]._std_dev

# Create an OpenMOC Geometry from the OpenMOC Geometry
openmoc_geometry = \
    openmc.openmoc_compatible.get_openmoc_geometry(library.geometry)

# Load cross section data
openmoc_materials = \
    openmoc.materialize.load_openmc_mgxs_lib(library, openmoc_geometry)

###############################################################################
#                                    CMFD
###############################################################################

cmfd = openmoc.Cmfd()
cmfd.setLatticeStructure(1, 1) # more will create FSRs
cmfd.setCMFDRelaxationFactor(0.7)
cmfd.setCentroidUpdateOn(False)
cmfd.setGroupStructure([[i+1] for i in range(70)]) # 70 g

if cmfd_on:
    openmoc_geometry.setCmfd(cmfd)
openmoc_geometry.initializeFlatSourceRegions()

###############################################################################
#                             Generate Tracks
###############################################################################

if num_polar != 3:
    quad = openmoc.GLPolarQuad()
    quad.setNumPolarAngles(num_polar*2)
else:
    quad = openmoc.TYPolarQuad()
quad.setNumAzimAngles(num_azim)
quad.initialize()

# Initialize an OpenMOC TrackGenerator and Solver
track_generator = openmoc.TrackGenerator(
    openmoc_geometry, num_azim, azim_spacing)
track_generator.setQuadrature(quad)
track_generator.setNumThreads(num_threads)
track_generator.generateTracks()

###############################################################################
#                             Initialize Solver
###############################################################################

# Initialize an OpenMOC Solver
if source == "f":
    solver = openmoc.CPUSolver(track_generator)
elif source == "l":
    solver = openmoc.CPULSSolver(track_generator)
solver.setConvergenceThreshold(tolerance[0])
solver.setNumThreads(num_threads)

# Run an eigenvalue calulation with the MGXS from OpenMC
solver.setKeffFromNeutronBalance()
solver.computeEigenvalue(max_iters[0], res_type=openmoc.SCALAR_FLUX)
solver.printTimerReport()
keff_no_sph = solver.getKeff()
openmoc.log.py_printf('RESULT', 'OpenMOC keff w/o SPH: \t%1.5f', keff_no_sph)
openmoc.log.py_printf('RESULT', 'Bias no SPH \t%1.5f pcm', (keff_no_sph -
                      keff_mc.nominal_value)  / keff_mc.nominal_value * 1e5)

# Extract the OpenMOC scalar fluxes
fluxes_no_sph = openmoc.process.get_scalar_fluxes(solver)

###############################################################################
#                  Load linear source moments in Solver
###############################################################################

# Get cell to fsr map
cells_to_fsrs = {}
num_fsrs = openmoc_geometry.getNumFSRs()
MOC_volumes = np.zeros(num_fsrs)
for i in range(num_fsrs):       # fsr id
    cell = openmoc_geometry.findCellContainingFSR(i)   # cell pointer
    cells_to_fsrs[cell.getId()] = i
    MOC_volumes[i] = track_generator.getFSRVolume(i)
k_eff_mc = keff_mc.nominal_value
print("Cells to FSRs", cells_to_fsrs)

# Loop on cells and input source from cell to FSR
inscatter_f = np.zeros([n_groups, 2, 2])
cells = openmoc_geometry.getAllMaterialCells()
for c in cells:
    cell = cells[c]
    c_id = cell.getId()

    # Obtain source tallies
    nufission = sp.get_tally(name='Linear fission source cell '+
                             cell.getName()).get_reshaped_data()[0,:,:,:,0,0]
    inscatter = sp.get_tally(name='Linear scatt. source cell '+
                             cell.getName()).get_reshaped_data()[0,:,:,:,:,0,0]
    flux = sp.get_tally(name='Linear flux cell '+
                        cell.getName()).get_reshaped_data()[0,:,:,:,0,0]
    chi = library.get_mgxs(c_id, 'chi').get_xs(nuclides='sum')
    nufission_xs = library.get_mgxs(c_id, 'nu-fission').get_xs(nuclides='sum')
    try:
        scatter_xs = library.get_mgxs(c_id, 'consistent nu-scatter matrix').get_xs(nuclides='sum')
    except:
        scatter_xs = library.get_mgxs(c_id, 'nu-scatter matrix').get_xs(nuclides='sum')

    # Symmetrize tallies (for quarter)
    if n_sectors_w == 2 and symmetryze_all:
        if "f0s0" not in cell.getName():
            other_name = list(cell.getName())
            other_name[-1] = str(int(1.25 - int(cell.getName()[-1])))
            other_name = ''.join(other_name)
            flux_2 = sp.get_tally(name='Linear flux cell '+
                        other_name).get_reshaped_data()[0,:,:,:,0,0]
            inscatter_2 = sp.get_tally(name='Linear scatt. source cell '+
                             other_name).get_reshaped_data()[0,:,:,:,:,0,0]
            nufission_2 = sp.get_tally(name='Linear fission source cell '+
                             other_name).get_reshaped_data()[0,:,:,:,0,0]
            flux[:, 1, 0] = (flux[:, 1, 0] + flux_2[:, 0, 1]) / 2
            flux[:, 0, 1] = (flux[:, 0, 1] + flux_2[:, 1, 0]) / 2
            inscatter[:, 1, 0] = (inscatter[:, 1, 0] + inscatter[:, 1, 0]) / 2
            inscatter[:, 0, 1] = (inscatter[:, 0, 1] + inscatter[:, 0, 1]) / 2
            nufission[:, 1, 0] = (nufission[:, 1, 0] + nufission[:, 1, 0]) / 2
            nufission[:, 0, 1] = (nufission[:, 0, 1] + nufission[:, 0, 1]) / 2

    # Divide by volume
    volume = np.sum(MOC_volumes[cells_to_fsrs[c_id]])
    nufission /= volume
    inscatter /= volume
    flux /= volume

    # Sum contributions from every group
    nufission = np.sum(nufission, 0)
    inscatter = np.sum(inscatter, 0)

    # Compare to re-constructed from flux moments and cross sections
    nufission_2 = np.tensordot(np.flipud(nufission_xs), flux, axes=([0], [0]))
    inscatter_2 = np.flip(np.tensordot(np.flip(scatter_xs, 0), flux, axes=([0], [0])), 0)
    flux_2 = library.get_mgxs(c_id, 'nu-fission').tallies['flux'].mean.flatten()/volume
    print("nufission diff", np.max(np.abs(nufission - nufission_2)))
    print("inscatter diff", np.max(np.abs(inscatter - inscatter_2)))
    print("flux diff", np.max(np.abs(flux[:,0,0] - flux_2)))

    # Use XS and flux to get the transport correction
    if "ani" in scattering:
        inscatter = inscatter_2

    # Flip to make it FAST to slow
    nufission = nufission
    inscatter = np.flip(inscatter, 0)

    # Input in solver
    for g in range(n_groups):
         src = chi[g] * nufission[0, 0] / k_eff_mc
         src += inscatter[g, 0, 0]
         src_x = chi[g] * nufission[1, 0] / k_eff_mc
         src_x += inscatter[g, 1, 0]
         src_y = chi[g] * nufission[0, 1] / k_eff_mc
         src_y += inscatter[g, 0, 1]

         if source == "l" and sph_mode=="fixed source":
             solver.setFixedSourceMomentsByCell(cell, g+1, src_x, src_y, 0)
         if src < 0:
             print("Negative source set in ", cell.getName(), g, src)

    # Examine symmetry of source
    if "f" in cell.getName():
        inscatter_f += inscatter / n_sectors_f

# Print error to symmetry
print("Symmetry check of inscatter x1y0", inscatter_f[:,0,1])
print("Symmetry check of inscatter x0y1", inscatter_f[:,1,0])

###############################################################################
#                Fixed source problems to obtain SPH factors
###############################################################################

# List domains (=cells) which should have sph factors
# They have to be in the top right quarter (so have a fsr), and be fuel OR AIC
sph_domains = []
for c in cells:
    if c in cells_to_fsrs.keys():
        cell = cells[c]
        if "f" in sph_regions and ("f" in cell.getName() or "b" in cell.getName()):
            sph_domains.append(c)
        elif "w" in sph_regions and ("w" in cell.getName()):
            sph_domains.append(c)
        elif "c" in sph_regions and ("c" in cell.getName()):
            sph_domains.append(c)
print("Cell ids with SPH factors", len(sph_domains), sph_domains)

# Compute SPH factors
solver.setRestartStatus(True)
solver.setConvergenceThreshold(tolerance[1])
sph, sph_library, sph_indices = \
    openmoc.materialize.compute_sph_factors(
        library, max_sph_iters=sph_iters, sph_tol=tol_sph, azim_spacing=azim_spacing,
        num_azim=num_azim, num_threads=num_threads, geometry = openmoc_geometry,
        solver=solver, track_generator=track_generator, sph_domains=sph_domains,
        sph_mode=sph_mode)

print("SPH min max mean", np.min(sph), np.max(sph), np.mean(sph))

# Examine SPH factors
sph_fuel = np.zeros(n_groups)
for k, cell in enumerate(library.domains):
    i = cells_to_fsrs[cell.id]
    print(i, cell.name, sph[i])

    # Homogenize and compare
    if "f" in cell.name:
        sph_fuel += sph[i] / n_sectors_f

for k, cell in enumerate(library.domains):
    i = cells_to_fsrs[cell.id]

    # Compare to average
    if "f" in cell.name:
        error = (sph[i] - sph_fuel) / sph_fuel * 100
        print(cell.name, np.min(error), np.max(error), np.mean(np.sqrt(error**2)))

###############################################################################
#                Eigenvalue Calculation with SPH Factors
###############################################################################

# Run an eigenvalue calculation with the SPH-corrected modified MGXS library
solver.setRestartStatus(False)
solver.setConvergenceThreshold(tolerance[2])
solver.computeEigenvalue(max_iters[1], res_type=openmoc.SCALAR_FLUX)
solver.printTimerReport()
keff_with_sph = solver.getKeff()

# Report the OpenMC and OpenMOC eigenvalues
openmoc.log.py_printf('RESULT', 'OpenMOC keff w/o SPH: \t%1.5f', keff_no_sph)
openmoc.log.py_printf('RESULT', 'OpenMOC keff w/ SPH: \t%1.5f', keff_with_sph)
openmoc.log.py_printf('RESULT', 'OpenMC keff:         \t%1.5f +/-', \
                                              keff_mc.nominal_value)
openmoc.log.py_printf('RESULT', 'Bias no SPH \t%1.5f pcm', (keff_no_sph -
                      keff_mc.nominal_value) / keff_mc.nominal_value * 1e5)
openmoc.log.py_printf('RESULT', 'Bias SPH    \t%1.5f pcm', (keff_with_sph -
                      keff_mc.nominal_value) / keff_mc.nominal_value * 1e5)

###############################################################################
#                         Extracting Scalar Fluxes
###############################################################################

# Extract the OpenMOC scalar fluxes
num_fsrs = openmoc_geometry.getNumFSRs()
fluxes_sph = openmoc.process.get_scalar_fluxes(solver)
fluxes_sph *= sph    ##### WRONG, here fluxes are replacing reaction rates
fluxes_sph /= num_fsrs

# Extract the OpenMC scalar fluxes
num_groups = openmoc_geometry.getNumEnergyGroups()
openmc_fluxes = np.zeros((num_fsrs, num_groups), dtype=np.float64)
nu_fission_xs = np.zeros([n_cells, n_groups])
fsr_to_cells = []

# Get the OpenMC flux in each FSR
for fsr in range(num_fsrs):

    # Find the OpenMOC cell and volume for this FSR
    openmoc_cell = openmoc_geometry.findCellContainingFSR(fsr)
    cell_id = openmoc_cell.getId()
    cell_name = openmoc_cell.getName()
    fsr_volume = track_generator.getFSRVolume(fsr)

    # Store the volume-averaged flux
    mgxs = library.get_mgxs(cell_id, 'nu-fission')
    flux = mgxs.tallies['flux'].mean.flatten()
    flux = np.flipud(flux) / fsr_volume
    openmc_fluxes[fsr, :] = flux
    fsr_to_cells.append(cell_name)

# Extract energy group edges
group_edges = library.energy_groups.group_edges
group_edges += 1e-5     # Adjust lower bound to 1e-3 eV (for loglog scaling)

# Compute difference in energy bounds for each group
group_deltas = np.ediff1d(group_edges)
group_edges = np.flipud(group_edges)
group_deltas = np.flipud(group_deltas)

# Get nu-fission XS
for kk, cell1_c in enumerate(library.domains):
    mgxs = library.get_mgxs(cell1_c.id, 'nu-fission')
    nu_fission_xs[kk, :] = mgxs.get_xs(nuclides='sum')

# Normalize fluxes to the total fission source
openmc_fluxes /= np.sum(openmc_fluxes * nu_fission_xs)
fluxes_sph /= np.sum(fluxes_sph * nu_fission_xs)
fluxes_no_sph /= np.sum(fluxes_no_sph * nu_fission_xs)

###############################################################################
#                 Plot the OpenMC, OpenMOC Scalar Fluxes
###############################################################################

# Extend the mgxs values array for matplotlib's step plot of fluxes
openmc_fluxes = np.insert(openmc_fluxes, 0, openmc_fluxes[:,0], axis=1)
fluxes_no_sph = np.insert(fluxes_no_sph, 0, fluxes_no_sph[:,0], axis=1)
fluxes_sph = np.insert(fluxes_sph, 0, fluxes_sph[:,0], axis=1)

# Plot OpenMOC and OpenMC fluxes in each FSR
for fsr in range(num_fsrs):

    # Get the OpenMOC cell and material for this FSR
    cell = openmoc_geometry.findCellContainingFSR(fsr)
    material_name = cell.getFillMaterial().getName()
    cell_name = cell.getName()

    # Create a step plot for the MGXS
    fig = plt.figure()
    plt.plot(group_edges, openmc_fluxes[fsr,:],
             drawstyle='steps', color='r', linewidth=2)
    plt.plot(group_edges, fluxes_no_sph[fsr,:],
             drawstyle='steps', color='b', linewidth=2)
    plt.plot(group_edges, fluxes_sph[fsr,:],
             drawstyle='steps', color='g', linewidth=2)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Flux')
    plt.title('Normalized Flux ({0})'.format(material_name))
    plt.xlim((min(group_edges), max(group_edges)))
    plt.legend(['openmc', 'openmoc w/o sph', 'openmoc w/ sph'], loc='best')
    plt.grid()
    filename = 'plots/flux-{0}.png'.format(cell_name.replace(' ', '-'))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

###############################################################################
#                 Plot OpenMC-to-OpenMOC Scalar Flux Errors
###############################################################################

# Compute the percent relative error in the flux
rel_err_no_sph = np.zeros([n_cells, n_groups])
rel_err_sph = np.zeros([n_cells, n_groups])
rel_err_no_sph_plot = np.zeros(openmc_fluxes.shape)
rel_err_sph_plot = np.zeros(openmc_fluxes.shape)
abs_rate_no_sph = np.zeros(n_cells)
abs_rate_sph = np.zeros(n_cells)
abs_rate_mc = np.zeros(n_cells)
fuel_abs = 0
fuel_abs_error_no_sph = 0
fuel_abs_error_sph = 0

for fsr in range(num_fsrs):
    delta_flux_no_sph = fluxes_no_sph[fsr,1:] - openmc_fluxes[fsr,1:]
    delta_flux_sph = fluxes_sph[fsr,1:] - openmc_fluxes[fsr,1:]
    rel_err_no_sph[fsr,:] = delta_flux_no_sph / openmc_fluxes[fsr,1:] * 100.
    rel_err_sph[fsr,:] = delta_flux_sph / openmc_fluxes[fsr,1:] * 100.
    rel_err_no_sph_plot[fsr,:] = (fluxes_no_sph[fsr,:] - openmc_fluxes[fsr,:]) / openmc_fluxes[fsr,:] * 100.
    rel_err_sph_plot[fsr,:] = (fluxes_sph[fsr,:] - openmc_fluxes[fsr,:]) / openmc_fluxes[fsr,:] * 100.

    volume = track_generator.getFSRVolume(fsr)
    abs_xs = library.get_mgxs(openmoc_geometry.findCellContainingFSR(fsr)\
                             .getId(), 'absorption').get_xs(nuclides='sum')
    abs_rate_no_sph[fsr] += np.dot(fluxes_no_sph[fsr,1:], abs_xs) * volume
    abs_rate_sph[fsr] += np.dot(fluxes_sph[fsr,1:], abs_xs) * volume
    abs_rate_mc[fsr] += np.dot(openmc_fluxes[fsr,1:], abs_xs) * volume

for fsr in range(num_fsrs):
    if 'f' in openmoc_geometry.findCellContainingFSR(fsr).getName():
        fuel_abs += abs_rate_mc[fsr]
        fuel_abs_error_no_sph += abs_rate_no_sph[fsr] - abs_rate_mc[fsr]
        fuel_abs_error_sph += abs_rate_sph[fsr]- abs_rate_mc[fsr]

fuel_abs_error_no_sph /= fuel_abs
fuel_abs_error_sph /= fuel_abs
fuel_abs_error_no_sph *= 1e5
fuel_abs_error_sph *= 1e5

# Get cell indexes in library and fluxes from MC
cell_index = {}
MC_fluxes          = np.zeros([n_cells, n_groups_fine])
MC_fluxes_coarse   = np.zeros([n_cells, n_groups])
norm_fission_source = 0
for kk, cell1_c in enumerate(library.domains):
  cell1 = cell1_c.id
  cell_index[cell1] = int(kk)
  mgxs = library.get_mgxs(cell1, 'nu-fission')
  MC_fluxes[kk] = mgxs.tallies['flux'].mean.flatten()
  nu_fission_xs[kk, :] = mgxs.get_xs(nuclides='sum')

  # Condense MC fluxes
  for g in range(n_groups):
      for g2 in range(group_match[g], group_match[g + 1]):
          #print(g, g2, "   ", group_match[g], group_match[g + 1])
          MC_fluxes_coarse[kk, g] += MC_fluxes[kk, g2]
  MC_fluxes[kk]          = np.flipud(MC_fluxes[kk])   # to make it fast to slow
  MC_fluxes_coarse[kk]   = np.flipud(MC_fluxes_coarse[kk])
  norm_fission_source += np.dot(nu_fission_xs[kk, :], MC_fluxes_coarse[kk])

# Normalize fluxes with fission source
MC_fluxes_coarse /= np.sum(norm_fission_source)
fluxes = openmoc.process.get_scalar_fluxes(solver, fsrs='all', groups='all')
fluxes *= sph
fluxes /= num_fsrs
print(np.sum(fluxes) / np.sum(fluxes_no_sph), 1/(np.sum(fluxes) / np.sum(fluxes_no_sph)))

# Print error on fluxes and absorption rate
error_fluxes = np.zeros([n_cells, n_groups])
error_fluxes2 = np.zeros([n_cells, n_groups])
abs_rate_eq = np.zeros(n_cells)
abs_rate_mc = np.zeros(n_cells)
abs_rate_eq_p = np.zeros(num_fsrs)
abs_rate_no_sph_p = np.zeros(num_fsrs)
abs_rate_mc_p = np.zeros(num_fsrs)
abs_rate_mc_fuel = 0
fuel_abs = 0
fuel_abs_error_df = 0

for cell1_c in library.domains:
    c = cell1_c.id
    r = int(cell_index[c])
    vol_MOC = track_generator.getFSRVolume(cells_to_fsrs[c])
    for g in range(n_groups):
        error_fluxes[r, g] = (MC_fluxes_coarse[r, g] - fluxes[cells_to_fsrs[c], g]
                              * vol_MOC) / (MC_fluxes_coarse[r, g])
    print("Check flux sums  ", np.sum(MC_fluxes_coarse[r, :]),
          np.sum(fluxes[cells_to_fsrs[c], :] * vol_MOC))
    print("Check fission sum", np.sum(MC_fluxes_coarse[r, :] * nu_fission_xs[r, :]),
          np.sum(fluxes[cells_to_fsrs[c], :] * vol_MOC * nu_fission_xs[r, :]))

    abs_xs = library.get_mgxs(c, 'absorption').get_xs(nuclides='sum')
    abs_rate_eq[r] += np.dot(fluxes[cells_to_fsrs[c],:], abs_xs) * vol_MOC
    abs_rate_mc[r] += np.dot(MC_fluxes_coarse[r, :], abs_xs)
    abs_rate_eq_p[cells_to_fsrs[c]] += np.dot(fluxes[cells_to_fsrs[c],:], abs_xs) * vol_MOC
    abs_rate_no_sph_p[cells_to_fsrs[c]] += np.dot(fluxes_no_sph[cells_to_fsrs[c],1:], abs_xs) * vol_MOC
    abs_rate_mc_p[cells_to_fsrs[c]] += np.dot(MC_fluxes_coarse[r, :], abs_xs)

    if "f" in cell1_c.name:
        abs_rate_mc_fuel += np.dot(MC_fluxes_coarse[r, :], abs_xs) / n_sectors_f

# Compute integrated absorption rate error
int_abs = np.zeros(3)
int_abs_error_eq = np.zeros(3)
for cell1_c in library.domains:
    c = cell1_c.id
    r = int(cell_index[c])
    vol_MOC = track_generator.getFSRVolume(cells_to_fsrs[c])
    if 'f' in cell1_c.name:
        int_abs[0] += abs_rate_mc[r] * vol_MOC
        int_abs_error_eq[0] += (abs_rate_eq[r]- abs_rate_mc[r]) * vol_MOC
    if 'c' in cell1_c.name:
        int_abs[1] += abs_rate_mc[r] * vol_MOC
        int_abs_error_eq[1] += (abs_rate_eq[r]- abs_rate_mc[r]) * vol_MOC
    if 'w' in cell1_c.name:
        int_abs[2] += abs_rate_mc[r] * vol_MOC
        int_abs_error_eq[2] += (abs_rate_eq[r]- abs_rate_mc[r]) * vol_MOC
int_abs_error_eq = int_abs_error_eq / int_abs * 1e5

if (n_groups <= 8):
    print("Region and energy dependent flux errors", error_fluxes)
print("Eigenvalue calculation results           [fuel, gap, clad,  coolant]")
print("Average flux errors                    ", np.mean(np.abs(error_fluxes), 1)  * 1e5,  " pcm")
print("Max flux errors                        ", np.max(np.abs(error_fluxes), 1)   * 1e5,  " pcm")

print("\n Averaged over all cells results")
print("Average flux errors                    ", np.mean(np.abs(error_fluxes))  * 1e5, " pcm")
print("Max flux errors                        ", np.max(np.abs(error_fluxes))   * 1e5, " pcm")
print("Total, not absorption")
print("Absorption rate error                  ", (abs_rate_eq - abs_rate_mc) / abs_rate_mc * 1e5, "pcm")
print("Fuel/Clad/Water absorption rate error  ", int_abs_error_eq, "pcm")
print("Flux errors analyzed :", time.time()-start,"s")

###############################################################################
#                          Post-Normalize SPH factors
###############################################################################

# Normalization with total flux
if gen_SPH == 0:
    total_flux_MOC = 0
    total_flux_mc = 0
    for fsr in range(num_fsrs):
        volume = track_generator.getFSRVolume(fsr)

        total_flux_MOC += np.sum(fluxes_sph[fsr,1:]) * volume
        total_flux_mc  += np.sum(openmc_fluxes[fsr,1:]) * volume

    #sph *= total_flux_mc / total_flux_MOC
    #print("Total flux mc", total_flux_mc, "MOC", total_flux_MOC)

# Save SPH factors
sph_filename = './equivalence_factors/sph/'+keff_file.replace('keff', 'sph')[:-4]+\
               '_'+sph_regions+'.pkl'
print("SPH filename", sph_filename)
pickle.dump([sph, fsr_to_cells],open(sph_filename, 'wb'))

###############################################################################
#                             Generating Plots
###############################################################################

#openmoc.log.py_printf('NORMAL', 'Plotting geometry')
#openmoc.plotter.plot_cells(openmoc_geometry)
#openmoc.plotter.plot_flat_source_regions(openmoc_geometry)

openmoc.log.py_printf('NORMAL', 'Plotting absorption rate errors')
plot_params = openmoc.plotter.PlotParams()
plot_params.geometry = openmoc_geometry
plot_params.domain_type = 'fsr'
plot_params.filename = 'absorption_rate_frac_error'
plot_params.colorbar = True
plot_params.gridsize = 400
plot_params.cmap = matplotlib.cm.get_cmap('viridis')
#plot_params.vmin = -450
#plot_params.vmax = 250
openmoc.plotter.plot_spatial_data((abs_rate_eq_p - abs_rate_mc_p)
                                  / abs_rate_mc_p * 1e5, plot_params)

# Plotting absolute error
plot_params.filename = 'absorption_rate_error'
openmoc.plotter.plot_spatial_data((abs_rate_eq_p - abs_rate_mc_p)*1e5
                                  , plot_params)

plot_params.filename = 'absorption_rate_error_noeq'
openmoc.plotter.plot_spatial_data((abs_rate_no_sph_p - abs_rate_mc_p)*1e5
                                  , plot_params)

plot_params.filename = 'absorption_rate_MC'
openmoc.plotter.plot_spatial_data(abs_rate_mc_p, plot_params)
for cell1_c in library.domains:
    c_id = cell1_c.id
    if "f" not in cell1_c.name:
        abs_rate_mc_p[cells_to_fsrs[c_id]] = abs_rate_mc_fuel
plot_params.filename = 'absorption_rate_MC_deviation'
openmoc.plotter.plot_spatial_data(abs_rate_mc_p - abs_rate_mc_fuel, plot_params)
plot_params.filename = 'absorption_rate_MOCeq'
openmoc.plotter.plot_spatial_data(abs_rate_eq_p, plot_params)
plot_params.filename = 'absorption_rate_MOC'
openmoc.plotter.plot_spatial_data(abs_rate_no_sph_p, plot_params)

# Plot OpenMOC relative flux errors in each FSR
for fsr in range(num_fsrs):

    # Get the OpenMOC cell and material for this FSR
    cell = openmoc_geometry.findCellContainingFSR(fsr)
    material_name = cell.getFillMaterial().getName()
    cell_name = cell.getName()

    # Create a step plot for the MGXS
    fig = plt.figure()
    plt.plot(group_edges, rel_err_no_sph_plot[fsr,:],
             drawstyle='steps', color='r', linewidth=2)
    plt.plot(group_edges, rel_err_sph_plot[fsr,:],
             drawstyle='steps', color='b', linewidth=2)

    plt.xscale('log')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Relative Error [%]')
    plt.title('OpenMOC-to-OpenMC Flux Rel. Err. ({0})'.format(cell_name))
    plt.xlim((min(group_edges), max(group_edges)))
    plt.legend(['openmoc w/o sph', 'openmoc w/ sph'], loc='best')
    plt.grid()
    filename = 'plots/rel-err-{0}.png'.format(cell_name.replace(' ', '-'))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Plot SPH factors in regions of interest
plt.figure()
ax = plt.subplot(111)
for i in range(n_rings_f):
    for fsr in range(num_fsrs):
        cell = openmoc_geometry.findCellContainingFSR(fsr)
        cell_name = cell.getName()
        if "0" == cell_name[3] and (("f" in sph_regions and "f" in cell_name) or
                                    ("c" in sph_regions and "c" in cell_name) or
                                    ("w" in sph_regions and "w" in cell_name)):
            ax.plot(np.log(library.energy_groups.group_edges)/np.log(10),
                    1./np.append(sph[fsr,-1], np.flipud(sph[fsr,:])),
                    drawstyle='steps', label='1/SPH '+cell_name[:2])

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
plt.xlabel('log(E)', axis_font)
plt.ylabel('1/SPH (-)', axis_font)
plt.savefig('plots/sph_factors_'+str(n_rings_f)+'r_'+str(azim_spacing)+
            scattering[:-1]+'_'+str(n_groups)+'.png')

# Plot SPH factors everywhere
plt.figure()
ax = plt.subplot(111)
for i in range(n_rings_f):
    for fsr in range(num_fsrs):
        cell = openmoc_geometry.findCellContainingFSR(fsr)
        cell_name = cell.getName()
        ax.plot(np.log(library.energy_groups.group_edges)/np.log(10),
                1./np.append(sph[fsr,-1], np.flipud(sph[fsr,:])),
                drawstyle='steps', label='1/SPH '+cell_name[:2])

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
plt.xlabel('log(E)', axis_font)
plt.ylabel('1/SPH (-)', axis_font)
plt.savefig('plots/sph_factors_all_'+str(n_rings_f)+'r_'+str(azim_spacing)+
            scattering[:-1]+'_'+str(n_groups)+'.png')

openmoc.log.py_printf('TITLE', 'Finished')
print("End of simulation :", time.time()-start,"s")
