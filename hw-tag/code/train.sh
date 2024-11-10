#!/bin/bash

# Define the set of values to replace 0.05
values=(5 0.5 0.1 0.01 0.001)

# Loop over each value and run the command with that value
for lambda in "${values[@]}"; do
    # First command with the current lambda value
    python3 tag.py ../data/endev --model en_hmm_${lambda}.pkl --train ../data/ensup --lambda $lambda
    
    # Second command with the current lambda value
    python3 tag.py ../data/endev --model en_hmm_raw_${lambda}.pkl --train ../data/ensup ../data/enraw --lambda $lambda
done