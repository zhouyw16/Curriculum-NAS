# bash run-ws.sh

num="0"
seeds="777"
algos="gdas setn random enas"

for seed in ${seeds}
  do
  for alg in ${algos}
  do
    CUDA_VISIBLE_DEVICES=6 python search_ws.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo ${alg} --rand_seed ${seed} --sub_candidate_num ${num}
  done
done


