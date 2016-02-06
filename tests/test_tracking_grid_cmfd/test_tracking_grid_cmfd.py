#!/usr/bin/env python

import os
import sys
import math
sys.path.insert(0, os.pardir)
sys.path.insert(0, os.path.join(os.pardir, 'openmoc'))
from testing_harness import TestHarness
from input_set import GridInput

import openmoc

class TrackingGridCMFDTestHarness(TestHarness):
    """Tests tracking over a grid geometry with an overlaid CMFD mesh."""

    def __init__(self):
        super(TrackingGridCMFDTestHarness, self).__init__()
        self.input_set = GridInput()
        self._result = ''

    def _setup(self):
        """Initialize the materials and geometry in the InputSet."""
        super(TrackingGridCMFDTestHarness, self)._create_geometry()

    def _segment_track(self, track, geometry):
        """Segments a given track over a given geometry and records the
           resulting segment information to a string"""

        # Segmentize a track in a geometry, recording the segments in a string
        geometry.segmentize(track)
        num_segments = track.getNumSegments()
        info = ' ' + str(num_segments) + '\n'
        for i in range(num_segments):
            info += str(i) + ': '
            segment = track.getSegment(i)
            info += str(round(segment._length, 8)) + ', '
            info += str(segment._region_id) + ', '
            info += str(segment._cmfd_surface_fwd) + ', '
            info += str(segment._cmfd_surface_bwd) + ', '
            info += str(segment._material.getName()) + ', '
            info += str(segment._material.getId()) + '\n'
        track.clearSegments()
        return info

    def _run_openmoc(self):
        """Creates tracks over the geometry and segments them, saving the
           results in the _result string"""

        # Initialize track objects
        diag_track = openmoc.Track()
        nudge_diag_track = openmoc.Track()
        hor_track = openmoc.Track()
        ver_track = openmoc.Track()
        rev_diag_track = openmoc.Track()

        # Set track trajectories and locations
        diag_track.setValues(-3, -3, 0, 3, 3, 0, math.atan(1))
        nudge = 1e-5
        nudge_diag_track.setValues(-3+nudge, -3, 0, 3, 3-nudge, 0, math.atan(1))
        hor_track.setValues(-3, 0, 0, 3, 0, 0, 0)
        ver_track.setValues(0, -3, 0, 0, 3, 0, math.pi/2)
        rev_diag_track.setValues(3, 3, 0, -3, -3, 0, math.pi + math.atan(1))

        # Segmentize over the geometry with a fine and coarse cmfd mesh
        for m in [3, 51]:

            # Overlay simple CMFD mesh
            self._result += '{0} x {0} CMFD mesh\n'.format(m)
            geometry = self.input_set.geometry
            cmfd = openmoc.Cmfd()
            cmfd.setLatticeStructure(m, m)
            geometry.setCmfd(cmfd)
            geometry.initializeCmfd()

            # Segmentize tracks over the geometry
            self._result += 'Diagonal track'
            self._result += self._segment_track(diag_track, geometry)
            self._result += 'Nudged Diagonal track'
            self._result += self._segment_track(nudge_diag_track, geometry)
            self._result += 'Horizontal track'
            self._result += self._segment_track(hor_track, geometry)
            self._result += 'Vertical track'
            self._result += self._segment_track(ver_track, geometry)
            self._result += 'Reverse Diagonal track'
            self._result += self._segment_track(rev_diag_track, geometry)

    def _get_results(self, num_iters=False, keff=False, fluxes=False,
                     num_fsrs=True, num_segments=True, num_tracks=True,
                     hash_output=False):
        """Return the result string"""
        return self._result

if __name__ == '__main__':
    harness = TrackingGridCMFDTestHarness()
    harness.main()
