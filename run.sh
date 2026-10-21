#!/bin/bash

base_cmd="uv run lp.py"

### GERMAN
lst_k=$(seq 10 10 100)
lst_eps=(0 0.5 1 10 100)
sens_attrs=("age" "sex")
# id=1
# for k in $lst_k; do
#   for eps in ${lst_eps[@]}; do
#     for sens_attr in ${sens_attrs[@]}; do
#       fname="logs/german_${k}_${sens_attr}_eps${eps}.csv"
#       echo "$base_cmd --k $k --sens-attr $sens_attr --eps $eps --save-csv $fname"
#       bash -c "$base_cmd --k $k --sens-attr $sens_attr --eps $eps --save-csv $fname"
#       id=$((id + 1))
#     done
#   done
# done

### COMPAS
lst_k=$(seq 10 10 100)
lst_eps=(0 0.5 1 10 100)
sens_attrs=("race" "sex")
N=1000
id=1
for k in $lst_k; do
  for eps in ${lst_eps[@]}; do
    for sens_attr in ${sens_attrs[@]}; do
      fname="logs/compas_${k}_${sens_attr}_eps${eps}.csv"
      echo "$base_cmd --k $k --n $N --sens-attr $sens_attr --score-attr score --eps $eps --dataset data/compas-processed.csv --save-csv $fname"
      bash -c "$base_cmd --k $k --n $N --sens-attr $sens_attr --score-attr score --eps $eps --dataset data/compas-processed.csv --save-csv $fname"
      id=$((id + 1))
    done
  done
done

echo "number of jobs: $id"
