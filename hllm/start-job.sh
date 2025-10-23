#!/bin/bash

export LANG="UTF-8"

#python3 /njfs/train-comment/example/gaolin3/hllm/hllm.py
#python /jfs/train-comment/example/gaolin3/demo.py 
#torchrun --nproc_per_node=4 /njfs/train-comment/example/gaolin3/hllm/hllm.py
torchrun --nproc_per_node=1 /njfs/train-comment/example/gaolin3/hllm/item_embedding_infer.py
#torchrun --nproc_per_node=1 /njfs/train-comment/example/gaolin3/hllm/user_embedding_infer.py
