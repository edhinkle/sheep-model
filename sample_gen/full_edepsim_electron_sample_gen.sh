#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2
STEPS=$3

for i in $(seq 1 ${NFILES}); do
    ./run_make_electron_sample.sh electron_NDLAr_10MeVto15GeV_TEST_EDEP2SUPERA_5EVENTS ${NEVENTS} $i ${STEPS} &
done

wait
