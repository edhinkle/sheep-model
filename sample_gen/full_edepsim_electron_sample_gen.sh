#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2

for i in $(seq 1 ${NFILES}); do
    ./run_make_electron_sample.sh electron_NDLAr_10MeVto15GeV_TEST_EDEP2SUPERA_5EVENTS ${NEVENTS} $i &
done

wait
