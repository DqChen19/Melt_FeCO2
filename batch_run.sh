#!/bin/bash

for i in $(seq 0.05 0.1 0.90)
do
    echo "Running for CO2 = $i"
    python3 micrometeorite_entry_model.py $i
done
