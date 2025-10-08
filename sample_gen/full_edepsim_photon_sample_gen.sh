#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2

for i in $(seq 0 ${NFILES}); do
    ./run_make_photon_sample.sh photon_10MeVto2GeV_TEST ${NEVENTS} $i &
done

wait
