#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2
STEPS=$3
OUTDIR="/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/sample_gen/PHOTON_SAMPLES"

for i in $(seq 0 ${NFILES}); do
    ./run_make_photon_sample.sh photon_10MeVto2GeV_TEST ${OUTDIR} ${NEVENTS} $i ${STEPS} &
done

wait
