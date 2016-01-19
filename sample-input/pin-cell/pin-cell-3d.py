from openmoc import *
import openmoc.log as log
import openmoc.plotter as plotter
from openmoc.options import Options
from geometry import *

###############################################################################
#                          Main Simulation Parameters
###############################################################################

options = Options()

num_threads = options.getNumThreads()
azim_spacing = options.getAzimSpacing()
num_azim = options.getNumAzimAngles()
polar_spacing = options.getPolarSpacing()
num_polar = options.getNumPolarAngles()
tolerance = options.getTolerance()
max_iters = options.getMaxIterations()


###############################################################################
#                          Creating the TrackGenerator
###############################################################################

log.py_printf('NORMAL', 'Initializing the track generator...')

track_generator = TrackGenerator(geometry, num_azim, num_polar, azim_spacing, \
                                 polar_spacing)
track_generator.setNumThreads(num_threads)
track_generator.setOTF()
track_generator.setGlobalZMesh()
track_generator.generateTracks()


###############################################################################
#                            Running a Simulation
###############################################################################

solver = CPUSolver(track_generator)
solver.setNumThreads(num_threads)
solver.setConvergenceThreshold(tolerance)
solver.computeEigenvalue(max_iters)
solver.printTimerReport()


###############################################################################
#                             Generating Plots
###############################################################################

log.py_printf('NORMAL', 'Plotting data...')
plotter.plot_periodic_cycles_2D(track_generator)
plotter.plot_reflective_cycles_2D(track_generator)
plotter.plot_reflective_cycles_3D(track_generator)
plotter.plot_tracks_2D(track_generator)
plotter.plot_tracks_3D(track_generator)
plotter.plot_segments_on_fsrs(geometry, track_generator)
plotter.plot_materials(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_cells(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_flat_source_regions(geometry, gridsize=500, plane='xy', offset=0.)
plotter.plot_spatial_fluxes(solver, energy_groups=[1,2,3,4,5,6,7], \
  plane='xy', offset=0.)
plotter.plot_energy_fluxes(solver, fsrs=range(geometry.getNumFSRs()))

log.py_printf('TITLE', 'Finished')