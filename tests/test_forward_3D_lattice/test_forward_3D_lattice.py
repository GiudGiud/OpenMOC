#!/usr/bin/env python

import os
import sys
sys.path.insert(0, os.pardir)
sys.path.insert(0, os.path.join(os.pardir, 'openmoc'))
import openmoc
from testing_harness import TestHarness
from input_set import SimpleLatticeInput


class SimpleLatticeTestHarness(TestHarness):
    """An eigenvalue calculation in a 3D lattice with 7-group C5G7 data."""

    def __init__(self):
        super(SimpleLatticeTestHarness, self).__init__()
        self.input_set = SimpleLatticeInput(num_dimensions=3)
        self.num_polar = 4
        self.azim_spacing = self.spacing
        self.z_spacing = 0.5


    def _create_trackgenerator(self):
        """Instantiate a TrackGenerator."""
        geometry = self.input_set.geometry
        geometry.initializeFlatSourceRegions()
        self.track_generator = \
            openmoc.TrackGenerator3D(geometry, self.num_azim, self.num_polar,
                                     self.azim_spacing, self.z_spacing)
        self.track_generator.setSegmentFormation(openmoc.OTF_STACKS)


    def _get_results(self, num_iters=True, keff=True, fluxes=True,
                     num_fsrs=False, num_tracks=False, num_segments=False,
                     hash_output=True):
        """Digest info in the solver and return hash as a string."""
        return super(SimpleLatticeTestHarness, self)._get_results(
                num_iters=num_iters, keff=keff, fluxes=fluxes,
                num_fsrs=num_fsrs, num_tracks=num_tracks,
                num_segments=num_segments, hash_output=hash_output)

if __name__ == '__main__':
    harness = SimpleLatticeTestHarness()
    harness.main()