#include "../../../src/CPUSolver.h"
#include "../../../src/CPULSSolver.h"
#include "../../../src/log.h"
#include "../../../src/Mesh.h"
#include <array>
#include <iostream>
#include <fstream>
#include "../../models/load-geometry/helper-code/group-structures.h"
#include <fenv.h>

// IN CORE 4440 //

int main(int argc,  char* argv[]) {

#ifdef MPIx
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  log_set_ranks(MPI_COMM_WORLD);
  if (provided < MPI_THREAD_SERIALIZED) {
    log_printf(ERROR, "Not enough thread support");  
  }
#endif

  set_log_level("INFO");

  /* Define geometry to load */
  int n_rings_w = 2;
  int n_rings_f = 1;
  int n_sectors_w = 8;
  int n_sectors_f = 4;
  int i_length  = 460;
  int num_groups = 70;
  int num_cmfd_groups = 25;

  std::string scattering = "_anisotropic";
  std::string refinement = "_sym_adj";

  std::string file = "../geometries/full_core_"+std::to_string(n_rings_w)+
                     "wr"+std::to_string(n_sectors_w)+"ws_"+
                     std::to_string(n_rings_f)+"fr"+std::to_string(n_sectors_f)
                     +"fs"+scattering+refinement+"_"+
                     std::to_string(num_groups)+"g.geo_new";

  /* Define simulation parameters */
  #ifdef OPENMP
  int num_threads = omp_get_max_threads();
  #else
  int num_threads = 1;
  #endif

  /* Equivalence method */
  int use_df = 0;
  // sph activated by chosing geom file

  // Ray discretization
  double azim_spacing = 0.05;
  int num_azim = 64;
  double polar_spacing = 0.75;
  int num_polar = 10.;

  // FSR discretization though CMFD
  int n_ax_cmfd = 230;
  int n_mesh = std::max(n_ax_cmfd, 100);

  // Run parameters
  double tolerance = 1e-3;
  int max_iters = 12;
  int n_dom_x = 5;
  int n_dom_y = 5;
  int n_ax_doma = 3;
  std::vector< std::vector<double> > widths;
  std::vector<double> x_vec = {1.30302,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.30302,1.30302,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.30302,1.30302,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.30302,1.30302,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.30302,1.30302,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,
 1.30302,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.30302,1.30302,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,1.25984,
 1.25984,1.25984,1.25984,1.30302};
  std::vector<double> y_vec = x_vec;
  std::vector<double> z_vec;// = {2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.748,1.912,1.298,2.042,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.025,1.975,2.   ,1.74 ,2.26 ,2.   ,2.   ,2.   ,2.   ,1.   ,3.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.222,1.778,2.   ,1.937,2.063,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.419,1.581,2.   ,2.134,1.866,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.616,1.384,2.   ,2.331,1.669,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.813,1.187,2.   ,2.528,
// 1.472,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,3.   ,1.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,1.01 ,
// 2.99 ,2.725,1.275,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.508,1.492,2.   ,2.   ,
// 2.   ,1.806,2.194,1.164,2.   ,2.54 ,1.828,1.517,2.951,2.   ,2.   ,1.876,
// 2.124,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,2.   ,
// 2.   ,2.   };
  z_vec.resize(n_ax_cmfd, 460/n_ax_cmfd);

  widths.push_back(x_vec);
  widths.push_back(y_vec);
  widths.push_back(z_vec);

  /* Create CMFD lattice */
  Cmfd cmfd;
  cmfd.setWidths(widths);

  // Define a reduced CMFD group structure. To make this easier, I defined a
  // helper function in a seperate file (group-structures.h) which gives the
  // CASMO reduced group structures mapping (ie what MOC groups encompass each
  // CMFD group)
  std::vector<std::vector<int> > cmfd_group_structure =
      get_group_structure(num_groups, num_cmfd_groups);
  cmfd.setGroupStructure(cmfd_group_structure);

  // CMFD often needs a relaxation factor to be stable. It should be between
  // 0 and 1. The lower the relaxation factor, the more it stabilizes CMFD,
  // but the more it slows convergence as well.
  cmfd.setCMFDRelaxationFactor(0.7);
  //  cmfd.setSORRelaxationFactor(1.5);
  //cmfd.setKNearest(1);
  cmfd.setCentroidUpdateOn(false);
  cmfd.useAxialInterpolation(2);
  //cmfd.checkBalance();
  //cmfd.rebalanceSigmaT(true);
  //cmfd.useFluxLimiting(false);

  /* Load the geometry */
  log_printf(NORMAL, "Creating geometry...");
  Geometry geometry;
  geometry.loadFromFile(file);

  // After we load the geometry file, we should pass it the CMFD object
  geometry.setCmfd(&cmfd);

#ifdef MPIx
  // NOTE: this sets the domain decomposition. For this problem, we can only
  // domain decompose radially (x and y) with numbers 1 and 17 and axially (z)
  // with numbers 1, 2, 5, and 10 becuase MPI cannot split CMFD boundaries
  // (which is 17x17x10). If CMFD is turned off, you can decompose however you
  // like
  log_printf(NORMAL, "Setting domain decomposition...");  
  geometry.setDomainDecomposition(n_dom_x, n_dom_y, n_ax_doma, MPI_COMM_WORLD);
#endif

  // We always need to initialize the flat source regions when we're done
  // setting up the Geometry in ClosedMOC (not the case in OpenMOC)
  geometry.initializeFlatSourceRegions();

  /* Create the track generator */
  log_printf(NORMAL, "Initializing the track generator...");
  TrackGenerator3D track_generator(&geometry, num_azim, num_polar, azim_spacing,
                                   polar_spacing);
  track_generator.setNumThreads(num_threads);

  // VERY important to have this setting. The default is explicitly storing
  // 3D segments. It's a very bad idea. I should probably change the default...
  track_generator.setSegmentFormation(OTF_STACKS);

  // It's nice to let the TrackGenerator know where the axial material
  // boundaries are if you know them. That way it can ray trace cheaper and
  // more efficiently. It's not necessary, but can save you some time. If you
  // don't define it, OpenMOC has to try to find them itself. It's more
  // complicated than it seems.
  //
  // This case is uniform axially so it's easy. The boundaries are just the
  // geometry boundaries.
  //double z_arr[] = {-(double)i_length, 0.};
  //std::vector<double> segmentation_zones(z_arr, z_arr + sizeof(z_arr)
  //                                       / sizeof(double));
  //track_generator.setSegmentationZones(segmentation_zones);

  // Finally, generate the tracks
  track_generator.generateTracks();

  /* Run simulation */
  CPUSolver solver(&track_generator);
  solver.setNumThreads(num_threads);

  // More descriptive iteration history
  solver.setVerboseIterationReport();
  
  // Equivalence   --  MAKE SURE getDFindex has the right shape as well
  if (use_df) {
    //solver.useDiscontinuityFactors(use_df);
    //solver.setFirstDFIteration(14);
    //solver.loadDFFromFile("../../../../MGXS_runs/BEAVRS_pincells/df_new/SAVE/LS_df_"
    //     +std::to_string(n_rings_w)+"wr"+std::to_string(n_sectors_w)+"ws_"
    //     +std::to_string(n_rings_f)+"fr"+std::to_string(n_sectors_f)+"fs_"
    //     +std::to_string(num_polar/2)+"p_fuel"+scattering+"_"+std::to_string(num_groups)
    //     +".txt", 2*(n_rings_f)*n_sectors_w);

    /* Set indexes and keys in surface map to find right DF in solver */
    //for(int r = 0; r < n_rings_f-1; r++){
    //    geometry.fillSurfaceMap("f"+std::to_string(r)+"->"+"f"+std::to_string(r+1), 2 * r);
    //    geometry.fillSurfaceMap("f"+std::to_string(r+1)+"->"+"f"+std::to_string(r), 2 * r+1);
    //}

    //int r = n_rings_f - 1;
    //for (int i = 0; i<n_sectors_w; i++)
    //  geometry.fillSurfaceMap("f"+std::to_string(r)+"s0->g0s0", r+i);
    //for (int i = 0; i<n_sectors_w; i++)
    //  geometry.fillSurfaceMap("g0s0->f"+std::to_string(r)+"s0", r+n_sectors_w+i);
    //r++;
    //geometry.fillSurfaceMap("c->wr0", 2 * r);
    //geometry.fillSurfaceMap("wr0->c", 2 * r+1);
  }

  // This is really important for transport corrected cross-sections.
  //
  // Here damping is defined between 0 and infiinity. The higher the number,
  // the more the damping (opposite of CMFD damping).
  solver.stabilizeTransport(0.4);  // not 1!

  // Finally tell OpenMOC to solve the eigenvalue problem with a certain
  // RMS fission rate threshold and maximum number of iterations, print the
  // timing report at the end
  solver.setConvergenceThreshold(tolerance);
  solver.computeEigenvalue(max_iters);
  solver.printTimerReport();

  Lattice mesh_lattice;
  Mesh mesh(&solver);
  mesh.createLattice(289, 289, n_mesh);
  Vector3D rx_rates = mesh.getFormattedReactionRates(FISSION_RX, true);
  Vector3D rx_rates2 = mesh.getFormattedReactionRates(ABSORPTION_RX, true);
  Vector3D rx_rates3 = mesh.getFormattedReactionRates(NUFISSION_RX, true);

  /* Append _df if a df calculation */
  std::string eq_suff = "";
  if (use_df == 1)
      eq_suff += "_df";
  
  /* Print reaction rates to file */
  int my_rank = 0;
#ifdef MPIx
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
  if (my_rank == 0){

    /* Fission */
    std::ofstream myfile;
    myfile.open("MOC_fission_rates_"+std::to_string(n_rings_w)+"wr_"+
                std::to_string(n_rings_f)+"fr"+eq_suff+".txt", 
                std::ios::out);
    for (int k=0; k < rx_rates.at(0).at(0).size(); k++) {
      for (int j=0; j < rx_rates.at(0).size(); j++) {
        for (int i=0; i < rx_rates.size(); i++) {
          myfile << rx_rates.at(i).at(j).at(k) << " ";
        }
        myfile << std::endl;
      }
    }
    myfile.close();

    /* U238 absorption */
    myfile.open("MOC_U238abs_rates_"+std::to_string(n_rings_w)+"wr_"+
                        std::to_string(n_rings_f)+"fr"+eq_suff+".txt"
                        , std::ios::out);
    for (int k=0; k < rx_rates2.at(0).at(0).size(); k++) {
      for (int j=0; j < rx_rates2.at(0).size(); j++) {
        for (int i=0; i < rx_rates2.size(); i++) {
          myfile << rx_rates2.at(i).at(j).at(k) << " ";
        }
        myfile << std::endl;
      }
    }
    myfile.close();

    /* Nu-Fission source */
    myfile.open("MOC_nu-fission_rates_"+std::to_string(n_rings_w)+"wr_"+
                std::to_string(n_rings_f)+"fr"+eq_suff+".txt", 
                std::ios::out);
    double fission_norm = 0.;
    for (int k=0; k < rx_rates3.at(0).at(0).size(); k++) {
      for (int j=0; j < rx_rates3.at(0).size(); j++) {
        for (int i=0; i < rx_rates3.size(); i++) {
          myfile << rx_rates3.at(i).at(j).at(k) << " ";
          fission_norm += rx_rates3.at(i).at(j).at(k);
        }
        myfile << std::endl;
      }
    }
    myfile.close();
    std::cout << "nu-fission norm " << fission_norm << std::endl;

    /* Keff + others */
    myfile.open("MOC_Keff"+std::to_string(n_rings_w)+"wr_"+
                std::to_string(n_rings_f)+"fr"+eq_suff+".txt", 
                std::ios::out);
    myfile << solver.getKeff();
    myfile.close();
    
  }
  
  log_printf(TITLE, "Finished");

  //MPI needs to finalize at the end. If not, MPI throws an error.
#ifdef MPIx
  MPI_Finalize();
#endif
  return 0;
}
