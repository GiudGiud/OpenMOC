#===============================================================================
# User Options
#===============================================================================

COMPILER       ?= mpi
OPENMP         ?= yes
OPTIMIZE       ?= yes
DEBUG          ?= no
PROFILE        ?= no
INFO           ?= no
PRECISION      ?= single
CMFD_PRECISION ?= single
CUDA           ?= no

#===============================================================================
# Source Code List
#===============================================================================

case ?= models/run_time_standard/run_time_standard.cpp
#case ?= models/assembly_1/load-assembly.cpp
#case ?= models/core/load-core.cpp

source = \
Cell.cpp \
Cmfd.cpp \
CPULSSolver.cpp \
CPUSolver.cpp \
ExpEvaluator.cpp \
Geometry.cpp \
LocalCoords.cpp \
linalg.cpp \
log.cpp \
Material.cpp \
Matrix.cpp \
Mesh.cpp \
MOCKernel.cpp \
Point.cpp \
Progress.cpp \
Quadrature.cpp \
Region.cpp \
RunTime.cpp \
Solver.cpp \
Surface.cpp \
Timer.cpp \
Track.cpp \
Track3D.cpp \
TrackGenerator.cpp \
TrackGenerator3D.cpp \
TrackTraversingAlgorithms.cpp \
TraverseSegments.cpp \
Universe.cpp \
Vector.cpp

cuda_source = \
clone.cu \
GPUQuery.cu \
GPUCmfd.cu \
GPUSolver.cu

cases = \
Takeda/Takeda-unrodded.cpp \
pin-cell/pin-cell-3d.cpp \
fixed-source/pin-cell-fixed-3d.cpp \
simple-lattice/simple-lattice-3d.cpp \
gradients/one-directional/one-directional-gradient.cpp \
gradients/two-directional/two-directional-gradient.cpp \
c5g7/c5g7-3d.cpp \
c5g7/c5g7-3d-cmfd.cpp \
c5g7/c5g7-rodded-B.cpp \
c5g7/c5g7-rodded-B-2x2.cpp \
c5g7/c5g7-ws.cpp \
homogeneous/homogeneous.cpp \
single-assembly/quarter-c5g7-assembly.cpp \
single-assembly/single-c5g7-assembly.cpp \
load-geometry/load-geometry.cpp \
load-geometry/load-single-assembly.cpp \
load-geometry/load-2D-full-core.cpp \
load-geometry/load-full-core.cpp

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS = -std=gnu++0x
CUDAFLAGS = -DNVCC -std=c++11

# Regular gcc Compiler
ifeq ($(COMPILER),gnu)
  CC = gcc
  CFLAGS += -DGNU
  LDFLAGS += -lm -lstdc++
endif

# MPI wrapped compiler
ifeq ($(COMPILER),mpi)
  CC = mpic++
  CFLAGS += -DMPIx
  LDFLAGS += -lm -lstdc++
endif

# intel Compiler
ifeq ($(COMPILER),intel)
  CC = icpc
  #source += VectorizedSolver.cpp
  CFLAGS += -DINTEL
  #CFLAGS += -I/your/path/to/mkl/include
  #LDFLAGS += -mkl
endif

# MPI wrapped intel Compiler
ifeq ($(COMPILER),impi)
  CC = mpiicpc
  CFLAGS += -DINTEL -DMPIx
endif

# Clang Compiler
ifeq ($(COMPILER),clang)
  CC = clang++
  CFLAGS += -DCLANG
endif

# BG/Q gcc Cross-Compiler
# Note: make sure to set +mpiwrapper-gcc in your .soft file,
# as we only support the gcc cross-compiler wrapper
ifeq ($(COMPILER),bluegene)
  CC = mpic++
  CFLAGS += -DGNU
endif

# nVidia compiler
ifeq ($(CUDA),yes)
  NVCC = nvcc
  CUDAFLAGS += --relocatable-device-code true
  CUDALDFLAGS = -lcudadevrt -lcudart -lcusparse
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -O0 -g -fno-omit-frame-pointer
  CUDAFLAGS += -G
else
  ifeq ($(COMPILER),gnu)
    CFLAGS += -Wno-unused-result
  endif
  ifeq ($(COMPILER),mpi)
    CFLAGS += -Wno-unused-result
  endif
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  # pprof flags
  CFLAGS += -pg -O0 -fno-omit-frame-pointer
  # gprof and vtune flags
  #CFLAGS += -g
endif

# Numerical precision Flags
ifeq ($(PRECISION),single)
  CFLAGS += -DFP_PRECISION=float -DSINGLE
  CUDAFLAGS += -DFP_PRECISION=float -DSINGLE
endif
ifeq ($(PRECISION),double)
  CFLAGS += -DFP_PRECISION=double
  CUDAFLAGS += -DFP_PRECISION=double
endif

ifeq ($(CMFD_PRECISION),single)
  CFLAGS += -DCMFD_PRECISION=float -DLINALG_TOL=1E-7
  CUDAFLAGS += -DCMFD_PRECISION=float -DLINALG_TOL=1E-7
endif
ifeq ($(CMFD_PRECISION),double)
  CFLAGS += -DCMFD_PRECISION=double -DLINALG_TOL=1E-15
  CUDAFLAGS += -DCMFD_PRECISION=double -DLINALG_TOL=1E-15
endif

# Vector Flags, for strip mining with VEC_LENGTH sized vectors
CFLAGS += -DVEC_ALIGNMENT=64
CUDAFLAGS += -DVEC_ALIGNMENT=64
ifeq ($(PRECISION),single)
  CFLAGS += -DVEC_LENGTH=8
  CUDAFLAGS += -DVEC_LENGTH=8
endif
ifeq ($(PRECISION),double)
  CFLAGS += -DVEC_LENGTH=4
  CUDAFLAGS += -DVEC_LENGTH=4
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  ifeq ($(COMPILER),gnu)
    CFLAGS += -O3 -ffast-math -fpic -march=native #-flto
  endif
  ifeq ($(COMPILER),mpi)
    CFLAGS += -O3 -ffast-math -fpic -march=native #-flto
  endif
  ifeq ($(COMPILER),intel)
    CFLAGS += -O3 -xhost -ansi-alias -no-prec-div #-ipo
  endif
  ifeq ($(COMPILER),impi)
    CFLAGS += -O3 -xhost -ansi-alias -no-prec-div #-ipo
  endif
  ifeq ($(COMPILER),bluegene)
    CFLAGS += -O3 -ffast-math -fpic
  endif
  ifeq ($(COMPILER),clang)
    CFLAGS += -O3 -ffast-math -fvectorize -fpic
  endif
  ifeq ($(CUDA),yes)
    CUDAFLAGS += --use_fast_math
  endif
  # Set the number of groups at compile time
  #CFLAGS += -DNGROUPS=70
  # Set the source to linear source at compile time
  #CFLAGS += -DLINEARSOURCE
  # Set the dimension to 3D at compile time
  #CFLAGS += -DTHREED
  # Only vacuum boundary conditions, avoid double store of track fluxes
  #CFLAGS += -DONLYVACUUMBC
endif

# Optimization report flags
ifeq ($(INFO),yes)
  ifeq ($(COMPILER),intel)
    CFLAGS += -qopt-report n=5 -qopt-report-phase=all
  endif
  ifeq ($(COMPILER),impi)
    CFLAGS += -qopt-report n=5 -qopt-report-phase=all
  endif
  ifeq ($(COMPILER),gnu)
    CFLAGS += -fopt-info-all
  endif
  ifeq ($(COMPILER),mpi)
    CFLAGS += -fopt-info-all
  endif
  ifeq ($(OPENMP),yes)
    CFLAGS += -openmp-report
  endif
endif

# OpenMP flags
ifeq ($(OPENMP),yes)
  CFLAGS += -fopenmp -DOPENMP
	LDFLAGS += -lgomp
endif

CFLAGS += -I../src/
CUDAFLAGS += -I../src/accel/cuda/

#===============================================================================
# Program Name and Object Files
#===============================================================================

obj := $(source:.cpp=.o)
source := $(addprefix ../src/, $(source))

cuda_obj := $(cuda_source:.cu=.o)
cuda_source := $(addprefix ../src/accel/cuda/, $(cuda_source))

obj := $(addprefix obj/, $(obj))
cuda_obj := $(addprefix obj/cuda/, $(cuda_obj))

cases := $(addprefix models/, $(cases))
cases += $(case)

programs = $(cases:.cpp=)
program = $(case:.cpp=)
$(info program is $(program))
#===============================================================================
# Targets to Build
#===============================================================================

default: folder $(addsuffix .o, $(program)) $(program)

# List of files to compile, then linker command
ifeq ($(CUDA),yes)
  $(programs):  $(obj) $(cuda_obj) $(addsuffix .o, $(program))
	  $(NVCC) $(obj) $(cuda_obj) $(addsuffix .o, $@) -o $@ $(LDFLAGS) $(CUDALDFLAGS)
else
  $(programs): $(obj) $(addsuffix .o, $(program))
	  $(CC) $(obj) $(addsuffix .o, $@) -o $@ $(LDFLAGS)
endif

# Recipe for object files from cpp source
obj/%.o: ../src/%.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

# Recipe for object files from cuda source
obj/cuda/%.o: ../src/accel/cuda/%.cu Makefile
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Recipe for object files from input
ifeq ($(CUDA),yes)
  %.o: %.cpp Makefile
	  $(NVCC) $(CUDAFLAGS) -c $< -o $@
else
  %.o: %.cpp Makefile
	  $(CC) $(CFLAGS) -c $< -o $@
endif

folder:
	mkdir -p obj obj/cuda

all: folder $(addsuffix .o, $(programs)) $(programs)

clean:
	rm -rf $(program) $(obj) $(cuda_obj) $(addsuffix .o, $(programs))

edit:
	vim -p $(case) $(cases)

run:
	./$(program)
	#mpirun -np 2 ./$(program)
