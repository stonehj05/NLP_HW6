#!/bin/bash

values=(5 1 0.5 0.1 0.05 0.01 0.001)

for lambda in "${values[@]}"; do
    echo $lambda
    python3 tag.py ../data/endev --model en_hmm_${lambda}.pkl --loss viterbi_error
done