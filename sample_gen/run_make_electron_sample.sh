#!/usr/bin/env bash

GEOMETRY="simple_LAr_geo.gdml"
OUTPUT=$1
NEVENTS=$2
EDEP_MACRO="electron_sim_NDLAr"
EDEP2SUPERA_YAML="sheep_ndlar_edep2supera.yaml"
OUTDIR="/pscratch/sd/e/ehinkle/nd_ana/sheep_single_shower/NDLAR_ELECTRON_SAMPLES"
INDIR="/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/sample_gen"
FILE_INDEX=$3
STEPS=$4


echo "Running script to generate electron samples."
chmod +x make_edep_sim_electron_sample.sh
./make_edep_sim_electron_sample.sh -g ${GEOMETRY} -f ${OUTPUT} -n ${NEVENTS} -m ${EDEP_MACRO} -y ${EDEP2SUPERA_YAML} -o ${OUTDIR} -i ${INDIR} -x ${FILE_INDEX} -s ${STEPS}

