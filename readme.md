# Readme


## To Run the code


1. Clone the repository

```bash
git clone git@github.com:zhouyw16/data-nas.git
```

2. Setup the environment

```bash
cd data-nas
source ~/pyenv/bin/activate
export TORCH_HOME=/DATA/DATANAS1/zyw16/TorchData/
```

3. Modify the XAutoDL module
```bash
pip install xautodl
```

```python
# optimizers.py
def get_optim_scheduler(parameters, config, two_criterion=False):
    ......
    if config.criterion == "Softmax":
        criterion = torch.nn.CrossEntropyLoss()
        w_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    elif config.criterion == "SmoothSoftmax":
        criterion = CrossEntropyLabelSmooth(config.class_num, config.label_smooth)
        w_criterion = CrossEntropyLabelSmooth(config.class_num, config.label_smooth, reduction='none')
    else:
        raise ValueError("invalid criterion : {:}".format(config.criterion))
    if two_criterion:
        return optim, scheduler, criterion, w_criterion
    ......

# generic_model.py
    def return_rank(self, arch):
        archs = Structure.gen_all(self._op_names, self._max_nodes, False)
        pairs = [(self.get_log_prob(a), a) for a in archs]
        sorted_pairs = sorted(pairs, key=lambda x: -x[0])
        n = len(sorted_pairs)
        for i, pair in enumerate(sorted_pairs):
            p, a = pair
            if arch == a.tostr():
                return i, n
        return -1, n

```


4. Run
```bash
CUDA_VISIBLE_DEVICES=n python search_ws.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1 --rand_seed 777 --subnet_candidate_num 5
```

5. Batch Run
```bash
bash run-ws.sh
```