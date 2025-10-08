#!/usr/bin/env bash

GEOMETRY=$1
EDEP_FILENAME=$2
NEVENTS=$3
EDEP_MACRO=$4
OUTDIR=$5
INDIR=$6
FILE_INDEX=$7


rndSEED=$(( 1000 + ${FILE_INDEX} ))
baseFileName=${EDEP_FILENAME}.${rndSEED}
EDEP_FILE=${baseFileName}.EDEPSIM.root 
EDEP_SEEDED_MACRO=${EDEP_MACRO}_${rndSEED}.mac

# Go to output directory
cd ${OUTDIR}

# Add random seed to macro file
cp ${INDIR}/${EDEP_MACRO}.mac ${EDEP_SEEDED_MACRO}
sed "1i /edep/random/randomSeed ${rndSEED}" ${INDIR}/${EDEP_MACRO}.mac > ${EDEP_SEEDED_MACRO}

echo "Generating EDepSim sample with ${NEVENTS} events, output file: ${EDEP_FILE}, random seed: ${rndSEED}"

export CPATH=$EDEPSIM/include/EDepSim:$CPATH

edep-sim \
    -g ${INDIR}/${GEOMETRY} \
    -o ${EDEP_FILE} \
    -e ${NEVENTS} \
    ${EDEP_SEEDED_MACRO}


H5_FILE=${baseFileName}.CONVERT2H5.hdf5

python3 ${INDIR}/single_shower_convert_edepsim_roottoh5.py ${EDEP_FILE} ${H5_FILE} 


if [ -f "${OUTDIR}/EDEPSIM/${EDEP_FILE}" ]; then
    rm -f "${OUTDIR}/EDEPSIM/${EDEP_FILE}"
fi

if [ -f "${OUTDIR}/CONVERT2H5/${H5_FILE}" ]; then
    rm -f "${OUTDIR}/CONVERT2H5/${H5_FILE}"
fi

if [ -f "${OUTDIR}/MACROS/${EDEP_SEEDED_MACRO}" ]; then
    rm -f "${OUTDIR}/MACROS/${EDEP_SEEDED_MACRO}"
fi


mv ${EDEP_FILE} ${OUTDIR}/EDEPSIM/${EDEP_FILE}
mv ${H5_FILE} ${OUTDIR}/CONVERT2H5/${H5_FILE}
mv ${EDEP_SEEDED_MACRO} ${OUTDIR}/MACROS/${EDEP_SEEDED_MACRO}
echo "Sample generation complete. Files in ${OUTDIR}/"
cd ${INDIR}
echo "Returned to input directory: ${INDIR}/"