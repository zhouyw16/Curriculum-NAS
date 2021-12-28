import os
import sys
import json
import numpy as np

ops = ["maxpool", "avgpool", "skipconnect", 
    "sepconv3x3", "sepconv5x5", "dilconv3x3", "dilconv5x5"]

checkpoint = {
    "normal_n2_p0": "none", 
    "normal_n2_p1": "none", 
    "normal_n3_p0": "none", 
    "normal_n3_p1": "none", 
    "normal_n3_p2": "none", 
    "normal_n4_p0": "none", 
    "normal_n4_p1": "none", 
    "normal_n4_p2": "none", 
    "normal_n4_p3": "none", 
    "normal_n5_p0": "none", 
    "normal_n5_p1": "none", 
    "normal_n5_p2": "none", 
    "normal_n5_p3": "none", 
    "normal_n5_p4": "none", 
    "reduce_n2_p0": "none", 
    "reduce_n2_p1": "none", 
    "reduce_n3_p0": "none", 
    "reduce_n3_p1": "none", 
    "reduce_n3_p2": "none", 
    "reduce_n4_p0": "none", 
    "reduce_n4_p1": "none", 
    "reduce_n4_p2": "none", 
    "reduce_n4_p3": "none", 
    "reduce_n5_p0": "none", 
    "reduce_n5_p1": "none", 
    "reduce_n5_p2": "none", 
    "reduce_n5_p3": "none", 
    "reduce_n5_p4": "none", 
    "normal_n2_switch": [], 
    "normal_n3_switch": [], 
    "normal_n4_switch": [], 
    "normal_n5_switch": [], 
    "reduce_n2_switch": [], 
    "reduce_n3_switch": [], 
    "reduce_n4_switch": [], 
    "reduce_n5_switch": []}


K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
np.random.seed(666)
for k in range(K):
    for key in checkpoint:
        if 'switch' in key:
            i = int(key[8])
            checkpoint[key] = np.random.choice(a=range(i), size=2, replace=False).tolist()
        else:
            checkpoint[key] = ops[np.random.randint(len(ops))]
    with open("checkpoints/checkpoint-%d.json" % k,"w") as f:
        json.dump(checkpoint, f)