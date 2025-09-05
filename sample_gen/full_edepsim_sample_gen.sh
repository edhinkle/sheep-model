#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2

for i in $(seq 0 ${NFILES}); do
    ./run_make_sample.sh electron_positron_10MeVto2GeV_TEST ${NEVENTS} $i &
done

wait
