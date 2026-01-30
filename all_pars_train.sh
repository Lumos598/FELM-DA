#!/bin/bash

pars=("bristle" "fltrust" "tofi" "scclip" "krum" "median" "qc" "avg" "balance" "felm_da")
attacks=("hidden" "random" "gaussian" "little" "converse")


for p in "${pars[@]}"
do
    for a in "${attacks[@]}"
    do
        nohup python ./train.py --par $p --attack $a --dataset "FEMNIST" --logdir "$p/femnist/$a-attack" > "femnist-log/$p-femnist-$a.log" 2>&1 &
        wait
    done
done