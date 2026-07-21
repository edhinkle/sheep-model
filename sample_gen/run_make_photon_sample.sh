#!/usr/bin/env bash

GEOMETRY="simple_LAr_geo.gdml"
OUTPUT=$1
OUTDIR=$2
NEVENTS=$3
EDEP_MACRO="photon_sim_2x2"
EDEP2SUPERA_YAML="sheep_ndlar_edep2supera.yaml"
INDIR="/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/sample_gen"
FILE_INDEX=$4
STEPS=$5

echo "Running script to generate photon samples."
chmod +x make_edep_sim_photon_sample.sh
./make_edep_sim_photon_sample.sh -g ${GEOMETRY} -f ${OUTPUT} -n ${NEVENTS} -m ${EDEP_MACRO} -y ${EDEP2SUPERA_YAML} -o ${OUTDIR} -i ${INDIR} -x ${FILE_INDEX} -s ${STEPS}
