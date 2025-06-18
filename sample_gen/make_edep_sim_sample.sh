#!/usr/bin/env bash

GEOMETRY=$1
EDEP_FILE=$2
NEVENTS=$3
EDEP_MACRO=$4

export CPATH=$EDEPSIM/include/EDepSim:$CPATH

edep-sim \
    -g ${GEOMETRY} \
    -o ${EDEP_FILE} \
    -e ${NEVENTS} \
    ${EDEP_MACRO}


H5_FILE=${EDEP_FILE%.root}.hdf5

python3 single_shower_convert_edepsim_roottoh5.py ${EDEP_FILE} ${H5_FILE} 

#mv ${EDEP_FILE} ${OUTDIR}/edep/${EDEP_FILE}
#mv ${H5_FILE} ${OUTDIR}/h5df/${H5_FILE}