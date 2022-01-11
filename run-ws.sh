# bash run-ws.sh

seeds="666 888 999"
algos="darts-v1 darts-v2 gdas setn random enas"

for seed in ${seeds}
  do
  for alg in ${algos}
  do
    CUDA_VISIBLE_DEVICES=2 python search_ws.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo ${alg} --rand_seed ${seed}
  done
done


