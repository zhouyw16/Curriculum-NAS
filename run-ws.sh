# bash run-ws.sh

num="5"
seeds="777 888 999"
algos="darts-v1"

for seed in ${seeds}
  do
  for alg in ${algos}
  do
    CUDA_VISIBLE_DEVICES=0 python search_ws.py --dataset cifar10  --data_path $TORCH_HOME --algo ${alg} --rand_seed ${seed} --subnet_candidate_num ${num}
  done
done


