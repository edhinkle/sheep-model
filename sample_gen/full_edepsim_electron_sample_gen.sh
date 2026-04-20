#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2

for i in $(seq 0 ${NFILES}); do
    ./run_make_electron_sample.sh electron_NDLAr_10MeVto15GeV_TEST6 ${NEVENTS} $i &
done

wait
