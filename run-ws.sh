# bash run-ws.sh

num="5"
seeds="999"
algos="setn"

for seed in ${seeds}
  do
  for alg in ${algos}
  do
    CUDA_VISIBLE_DEVICES=4 python search_ws.py --dataset cifar10  --data_path $TORCH_HOME --algo ${alg} --rand_seed ${seed} --subnet_candidate_num ${num}
  done
done


