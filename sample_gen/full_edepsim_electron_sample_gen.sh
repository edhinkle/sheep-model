#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2
STEPS=$3
OUTDIR="/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/sample_gen/2x2_ELECTRON_TEST"

for i in $(seq 1 ${NFILES}); do
    ./run_make_electron_sample.sh electron_2x2_10MeVto2GeV_TEST ${OUTDIR} ${NEVENTS} $i ${STEPS} &
done

wait
