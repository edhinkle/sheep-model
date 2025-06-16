#!/usr/bin/env bash

GEOMETRY=$1
OUTPUT=$2
NEVENTS=$3
EDEP_MACRO=$4

export CPATH=$EDEPSIM/include/EDepSim:$CPATH

edep-sim \
    -g ${GEOMETRY} \
    -o ${OUTPUT} \
    -e ${NEVENTS} \
    ${EDEP_MACRO}