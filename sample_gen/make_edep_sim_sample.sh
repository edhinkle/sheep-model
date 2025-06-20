#!/usr/bin/env bash

GEOMETRY=$1
EDEP_FILENAME=$2
NEVENTS=$3
EDEP_MACRO=$4
OUTDIR=$5
FILE_INDEX=$6

baseFileName=${EDEP_FILENAME}.${FILE_INDEX}
EDEP_FILE=${baseFileName}.EDEPSIM.root 
rndSEED=$(( 1000 + ${FILE_INDEX} ))
EDEP_SEEDED_MACRO=${EDEP_MACRO}_${rndSEED}.mac

# Add random seed to macro file
cp ${EDEP_MACRO}.mac ${EDEP_SEEDED_MACRO}
sed "1i /edep/random/randomSeed ${rndSEED}" ${EDEP_MACRO}.mac > ${EDEP_SEEDED_MACRO}

# Also randomize e+/e-
PARTICLE=$( ((RANDOM % 2)) && echo "e+" || echo "e-" )
sed "s|/gps/particle e-.*|/gps/particle $PARTICLE|" ${EDEP_SEEDED_MACRO} > ${EDEP_SEEDED_MACRO}.tmp
mv ${EDEP_SEEDED_MACRO}.tmp ${EDEP_SEEDED_MACRO}

echo "Generating EDepSim sample with ${NEVENTS} events, output file: ${EDEP_FILE}, random seed: ${rndSEED}"

export CPATH=$EDEPSIM/include/EDepSim:$CPATH

edep-sim \
    -g ${GEOMETRY} \
    -o ${EDEP_FILE} \
    -e ${NEVENTS} \
    ${EDEP_SEEDED_MACRO}


H5_FILE=${baseFileName}.CONVERT2H5.hdf5

python3 single_shower_convert_edepsim_roottoh5.py ${EDEP_FILE} ${H5_FILE} 


if [ -f "${OUTDIR}/SAMPLES/EDEPSIM/${EDEP_FILE}" ]; then
    rm -f "${OUTDIR}/SAMPLES/EDEPSIM/${EDEP_FILE}"
fi

if [ -f "${OUTDIR}/SAMPLES/CONVERT2H5/${H5_FILE}" ]; then
    rm -f "${OUTDIR}/SAMPLES/CONVERT2H5/${H5_FILE}"
fi

if [ -f "${OUTDIR}/SAMPLES/MACROS/${EDEP_SEEDED_MACRO}" ]; then
    rm -f "${OUTDIR}/SAMPLES/MACROS/${EDEP_SEEDED_MACRO}"
fi


mv ${EDEP_FILE} ${OUTDIR}/SAMPLES/EDEPSIM/${EDEP_FILE}
mv ${H5_FILE} ${OUTDIR}/SAMPLES/CONVERT2H5/${H5_FILE}
mv ${EDEP_SEEDED_MACRO} ${OUTDIR}/SAMPLES/MACROS/${EDEP_SEEDED_MACRO}
echo "Sample generation complete. Files moved to ${OUTDIR}/SAMPLES/"