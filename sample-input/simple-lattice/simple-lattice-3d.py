import openmoc
import openmoc.log as log
import openmoc.plotter as plotter
from openmoc.options import Options
from geometry import geometry

###############################################################################
#######################   Main Simulation Parameters   ########################
###############################################################################

options = Options()
openmoc.set_line_length(120)

num_threads = options.num_omp_threads
azim_spacing = options.azim_spacing
num_azim = options.num_azim
polar_spacing = options.polar_spacing
num_polar = options.num_polar
tolerance = options.tolerance
max_iters = options.max_iters


###############################################################################
########################   Creating the TrackGenerator   ######################
###############################################################################

log.py_printf('NORMAL', 'Initializing the track generator...')

quad = openmoc.GLPolarQuad()
quad.setNumPolarAngles(num_polar)
quad.setNumAzimAngles(num_azim)

track_generator = openmoc.TrackGenerator3D(geometry, num_azim, num_polar,
                                           azim_spacing, polar_spacing)
track_generator.setQuadrature(quad)
track_generator.setNumThreads(num_threads)
track_generator.setTrackGenerationMethod(openmoc.MODULAR_RAY_TRACING)
track_generator.setSegmentFormation(openmoc.OTF_TRACKS)
track_generator.generateTracks()

###############################################################################
###########################   Running a Simulation   ##########################
###############################################################################

solver = openmoc.CPULSSolver(track_generator)
solver.setNumThreads(num_threads)
solver.setConvergenceThreshold(tolerance)
solver.computeEigenvalue(max_iters)
solver.printTimerReport()


###############################################################################
############################   Generating Plots   #############################
###############################################################################

log.py_printf('NORMAL', 'Plotting data...')

plotter.plot_materials(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_cells(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_flat_source_regions(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_fission_rates(solver)
plotter.plot_spatial_fluxes(solver, energy_groups=[1,2,3,4,5,6,7], \
  plane='xy', offset=0.)
plotter.plot_energy_fluxes(solver, fsrs=range(geometry.getNumFSRs()))

log.py_printf('TITLE', 'Finished')