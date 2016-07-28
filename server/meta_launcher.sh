#!/usr/bin/env bash

let "begin=0"
let "end=9"

for i in {${begin}..${end}; do

    qsub basile-simulation_${i}.sh

    done
