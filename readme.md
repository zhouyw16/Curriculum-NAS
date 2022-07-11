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
export TORCH_HOME=</data and XAutoDL files>
```

3. Modify the XAutoDL module
```bash
pip install xautodl
```

```python
# evaluation_utils.py
def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# XAutoDL/optimizers.py
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

# XAutoDL/generic_model.py
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
