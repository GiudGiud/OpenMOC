import openmoc
from cells import cells, surfaces

###############################################################################
##########################   Creating Universes   #############################
###############################################################################

universes = {}

# Instantiate Cells
universes['Root']                        = openmoc.Universe()
universes['UO2']                         = openmoc.Universe()
universes['MOX 4.3%']                    = openmoc.Universe()
universes['MOX 7.0%']                    = openmoc.Universe()
universes['MOX 8.7%']                    = openmoc.Universe()
universes['Guide Tube']                  = openmoc.Universe()
universes['Fission Chamber']             = openmoc.Universe()
universes['Control Rod']                 = openmoc.Universe()
universes['Moderator Pin']               = openmoc.Universe()
universes['Reflector']                   = openmoc.Universe()
universes['Refined Reflector Mesh']      = openmoc.Universe()
universes['UO2 Unrodded Assembly']       = openmoc.Universe()
universes['UO2 Rodded Assembly']         = openmoc.Universe()
universes['MOX Unrodded Assembly']       = openmoc.Universe()
universes['MOX Rodded Assembly']         = openmoc.Universe()
universes['Reflector Unrodded Assembly'] = openmoc.Universe()
universes['Reflector Rodded Assembly']   = openmoc.Universe()
universes['Reflector Right Assembly']    = openmoc.Universe()
universes['Reflector Bottom Assembly']   = openmoc.Universe()
universes['Reflector Corner Assembly']   = openmoc.Universe()

# Add cells to universes
universes['Root']                       .addCell(cells['Root'])
universes['UO2']                        .addCell(cells['UO2'])
universes['UO2']                        .addCell(cells['Moderator'])
universes['MOX 4.3%']                   .addCell(cells['MOX 4.3%'])
universes['MOX 4.3%']                   .addCell(cells['Moderator'])
universes['MOX 7.0%']                   .addCell(cells['MOX 7.0%'])
universes['MOX 7.0%']                   .addCell(cells['Moderator'])
universes['MOX 8.7%']                   .addCell(cells['MOX 8.7%'])
universes['MOX 8.7%']                   .addCell(cells['Moderator'])
universes['Guide Tube']                 .addCell(cells['Guide Tube'])
universes['Guide Tube']                 .addCell(cells['Moderator'])
universes['Fission Chamber']            .addCell(cells['Fission Chamber'])
universes['Fission Chamber']            .addCell(cells['Moderator'])
universes['Control Rod']                .addCell(cells['Control Rod'])
universes['Control Rod']                .addCell(cells['Moderator'])
universes['Moderator Pin']              .addCell(cells['Moderator in Pin'])
universes['Moderator Pin']              .addCell(cells['Moderator'])
universes['Reflector']                  .addCell(cells['Reflector'])
universes['Refined Reflector Mesh']     .addCell(cells['Refined Reflector Mesh'])
universes['UO2 Unrodded Assembly']      .addCell(cells['UO2 Unrodded Assembly'])
universes['UO2 Rodded Assembly']        .addCell(cells['UO2 Rodded Assembly'])
universes['MOX Unrodded Assembly']      .addCell(cells['MOX Unrodded Assembly'])
universes['MOX Rodded Assembly']        .addCell(cells['MOX Rodded Assembly'])
universes['Reflector Unrodded Assembly'].addCell(cells['Reflector Unrodded Assembly'])
universes['Reflector Rodded Assembly']  .addCell(cells['Reflector Rodded Assembly'])
universes['Reflector Right Assembly']   .addCell(cells['Reflector Right Assembly'])
universes['Reflector Bottom Assembly']  .addCell(cells['Reflector Bottom Assembly'])
universes['Reflector Corner Assembly']  .addCell(cells['Reflector Corner Assembly'])