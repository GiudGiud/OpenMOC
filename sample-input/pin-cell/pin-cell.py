import openmoc

###############################################################################
#                          Main Simulation Parameters
###############################################################################

opts = openmoc.options.Options()

openmoc.log.set_log_level('DEBUG')


###############################################################################
#                            Creating Materials
###############################################################################

openmoc.log.py_printf('NORMAL', 'Importing materials data from HDF5...')

materials = openmoc.materialize.load_from_hdf5('c5g7-mgxs.h5', '../')


###############################################################################
#                            Creating Surfaces
###############################################################################

openmoc.log.py_printf('NORMAL', 'Creating surfaces...')

zcylinder = openmoc.ZCylinder(x=0.0, y=0.0, radius=1.0, name='pin')
left = openmoc.XPlane(x=-2.0, name='left')
right = openmoc.XPlane(x=2.0, name='right')
top = openmoc.YPlane(y=2.0, name='top')
bottom = openmoc.YPlane(y=-2.0, name='bottom')

left.setBoundaryType(openmoc.REFLECTIVE)
right.setBoundaryType(openmoc.REFLECTIVE)
top.setBoundaryType(openmoc.REFLECTIVE)
bottom.setBoundaryType(openmoc.REFLECTIVE)


###############################################################################
#                             Creating Cells
###############################################################################

openmoc.log.py_printf('NORMAL', 'Creating cells...')

fuel = openmoc.Cell(name='fuel')
fuel.setFill(materials['UO2'])
fuel.addSurface(halfspace=-1, surface=zcylinder)

moderator = openmoc.Cell(name='moderator')
moderator.setFill(materials['Water'])
moderator.addSurface(halfspace=+1, surface=zcylinder)
moderator.addSurface(halfspace=+1, surface=left)
moderator.addSurface(halfspace=-1, surface=right)
moderator.addSurface(halfspace=+1, surface=bottom)
moderator.addSurface(halfspace=-1, surface=top)


###############################################################################
#                            Creating Universes
###############################################################################

openmoc.log.py_printf('NORMAL', 'Creating universes...')

root_universe = openmoc.Universe(name='root universe')
root_universe.addCell(fuel)
root_universe.addCell(moderator)


###############################################################################
#                         Creating the Geometry
###############################################################################


openmoc.log.py_printf('NORMAL', 'Creating geometry...')

geometry = openmoc.Geometry()
geometry.setRootUniverse(root_universe)


###############################################################################
#                          Creating the TrackGenerator
###############################################################################

openmoc.log.py_printf('NORMAL', 'Initializing the track generator...')

track_generator = openmoc.TrackGenerator(geometry, 32, 0.01)
track_generator.setNumThreads(opts.num_omp_threads)
track_generator.generateTracks()


###############################################################################
#                            Running a Simulation
###############################################################################

solver = openmoc.CPUSolver(track_generator)

num_surfaces = 1

# DUMMY variables to show arguments
cell_from = 0 
cell_to = 1
group = 2
polar_index = 0
current = 1.1

# Initialize arrays for currents
solver.setNumSurfaces(num_surfaces)
solver.initializePartialCurrentArrays()

# Set a dummy reference partial current, to initial the current array data structures
# This needs to be done for every surface (=pair of cell_from and cell_to), otherwise
# segfault. Start by cell_from = 0, otherwise won't work. Here we have two cells 0 and 1,
# so we set the current for 0->1 and the current for 1->0
solver.setReferencePartialCurrents(cell_from, cell_to, group, polar_index, current)
solver.setReferencePartialCurrents(cell_to, cell_from, group, polar_index, current)


solver.setNumThreads(1)  # change to opts.num_omp_threads
solver.setConvergenceThreshold(opts.tolerance)
solver.computeEigenvalue(1)
solver.printTimerReport()

openmoc.process.store_simulation_state(solver, use_hdf5=True)
#simulation_state = openmoc.process.restore_simulation_state(filename='states.h5')

###############################################################################
#                     Obtain partial currents from MOC
###############################################################################

azim = 4
polar = 1
group = 2

# Access by cell from, cell_to
current_01 = solver.getAngularPartialCurrent(cell_from, cell_to, azim, polar, group)
current_10 = solver.getAngularPartialCurrent(1-cell_from, 1-cell_to, azim, polar, group)

# Access by surface index
solver.getOngoingPartialCurrent(0, azim, polar, group)

print("Current from cell 0 to 1", current_01
print("Current from cell 1 to 0", current_10)

###############################################################################
#                             Generating Plots
###############################################################################

openmoc.log.py_printf('NORMAL', 'Plotting data...')

#openmoc.plotter.plot_quadrature(solver)
#openmoc.plotter.plot_tracks(track_generator)
#openmoc.plotter.plot_segments(track_generator)
openmoc.plotter.plot_materials(geometry)
openmoc.plotter.plot_cells(geometry)
openmoc.plotter.plot_flat_source_regions(geometry)
openmoc.plotter.plot_spatial_fluxes(solver, energy_groups=[1,2,3,4,5,6,7])
openmoc.plotter.plot_energy_fluxes(solver, fsrs=range(geometry.getNumFSRs()))

openmoc.log.py_printf('TITLE', 'Finished')
