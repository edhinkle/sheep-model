#!/usr/bin/env bash

GEOMETRY=$1
EDEP_FILENAME=$2
NEVENTS=$3
EDEP_MACRO=$4
EDEP2SUPERA_YAML=$5
OUTDIR=$6
INDIR=$7
FILE_INDEX=$8


rndSEED=$(( 1000 + ${FILE_INDEX} ))
baseFileName=${EDEP_FILENAME}.${rndSEED}
EDEP_FILE=${baseFileName}.EDEPSIM.root 
EDEP_SEEDED_MACRO=${EDEP_MACRO}_${rndSEED}.mac
LARCV_FILE=${baseFileName}.LARCV.root 

# Go to output directory
cd ${OUTDIR}

if [[ ! -d "${OUTDIR}/EDEPSIM" ]]; then
    mkdir -p "${OUTDIR}/EDEPSIM"
fi

if [[ ! -d "${OUTDIR}/LARCV" ]]; then
    mkdir -p "${OUTDIR}/LARCV"
fi                                             

if [[ ! -d "${OUTDIR}/MACROS" ]]; then
    mkdir -p "${OUTDIR}/MACROS"
fi     


# Add random seed to macro file
cp ${INDIR}/${EDEP_MACRO}.mac ${EDEP_SEEDED_MACRO}
sed "1i /edep/random/randomSeed ${rndSEED}" ${INDIR}/${EDEP_MACRO}.mac > ${EDEP_SEEDED_MACRO}

# Also randomize e+/e- 
PARTICLE=$( ((RANDOM % 2)) && echo "e+" || echo "e-" )
sed "s|/gps/particle e-.*|/gps/particle $PARTICLE|" ${EDEP_SEEDED_MACRO} > ${EDEP_SEEDED_MACRO}.tmp
mv ${EDEP_SEEDED_MACRO}.tmp ${EDEP_SEEDED_MACRO}

echo "Generating edep-sim sample with ${NEVENTS} events, output file: ${EDEP_FILE}, random seed: ${rndSEED}"

shifter --image=mjkramer/sim2x2:ndlar011 --module=cvmfs -- /bin/bash << EOF1
set +o posix
source /opt/environment
export CPATH=$EDEPSIM/include/EDepSim:$CPATH

edep-sim \
    -g ${INDIR}/${GEOMETRY} \
    -o ${EDEP_FILE} \
    -e ${NEVENTS} \
    ${EDEP_SEEDED_MACRO}
EOF1

echo "Converting edep-sim files to LARCV files using edep2supera."

module load python
shifter --image=deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim bash << EOF2
python3 ${INDIR}/edep2supera/bin/run_edep2supera.py -c ${INDIR}/${EDEP2SUPERA_YAML} -o ${LARCV_FILE} ${EDEP_FILE}
EOF2

echo "Removing any existing files with the same name in output directories and moving new outputs to output directories."

if [ -f "${OUTDIR}/EDEPSIM/${EDEP_FILE}" ]; then
    rm -f "${OUTDIR}/EDEPSIM/${EDEP_FILE}"
fi

#if [ -f "${OUTDIR}/CONVERT2H5/${H5_FILE}" ]; then # REPLACED BY edep2supera
#    rm -f "${OUTDIR}/CONVERT2H5/${H5_FILE}"       # REPLACED BY edep2supera
#fi                                                # REPLACED BY edep2supera

if [ -f "${OUTDIR}/LARCV/${LARCV_FILE}" ]; then
    rm -f "${OUTDIR}/LARCV/${LARCV_FILE}"      
fi                                               

if [ -f "${OUTDIR}/MACROS/${EDEP_SEEDED_MACRO}" ]; then
    rm -f "${OUTDIR}/MACROS/${EDEP_SEEDED_MACRO}"
fi


mv ${EDEP_FILE} ${OUTDIR}/EDEPSIM/${EDEP_FILE}
#mv ${H5_FILE} ${OUTDIR}/CONVERT2H5/${H5_FILE} # REPLACED BY edep2supera
mv ${LARCV_FILE} ${OUTDIR}/LARCV/${LARCV_FILE}
mv ${EDEP_SEEDED_MACRO} ${OUTDIR}/MACROS/${EDEP_SEEDED_MACRO}
echo "Sample generation complete. Files in ${OUTDIR}/"
cd ${INDIR}
echo "Returned to input directory: ${INDIR}/"