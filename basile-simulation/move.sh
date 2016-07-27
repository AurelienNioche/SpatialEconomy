#!/usr/bin/env bash

let "begin=0"
let "end=9"

for i in {${begin}..${end}; do
            mv basile-simulation_${i}.sh basile-simulation
        done
