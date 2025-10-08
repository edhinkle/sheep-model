#!/usr/bin/env bash

GEOMETRY="simple_LAr_geo.gdml"
OUTPUT=$1
NEVENTS=$2
EDEP_MACRO="electron_sim"
OUTDIR="/pscratch/sd/e/ehinkle/nd_ana/sheep_single_shower/ELECTRON_SAMPLES"
INDIR="/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/sample_gen"
FILE_INDEX=$3

echo "Setting to 2x2 sim container for running edep-sim."
shifter --image=mjkramer/sim2x2:ndlar011 --module=cvmfs -- /bin/bash << EOF1
set +o posix
source /opt/environment
chmod +x make_edep_sim_electron_sample.sh
./make_edep_sim_electron_sample.sh ${GEOMETRY} ${OUTPUT} ${NEVENTS} ${EDEP_MACRO} ${OUTDIR} ${INDIR} ${FILE_INDEX}
EOF1