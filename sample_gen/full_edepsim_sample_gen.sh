#!/usr/bin/env bash

NEVENTS=$1
NFILES=$2

for i in $(seq 0 ${NFILES}); do
    ./run_make_sample.sh test_electron ${NEVENTS} $i &
done

wait
