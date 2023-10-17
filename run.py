import os
import numpy as np
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

rand_num = np.random.randint(10000000000)
os.system('accelerate launch \
        --multi_gpu \
        --num_processes 1 \
        --mixed_precision fp16 \
        --main_process_port '+str(rand_num%10000+10000)+' \
            train.py \
        --config=cifar10_config.py \
        --config.nnet.scalelong=0 \
        --config.nnet.kappa=0.5 \
        --config.train.batch_size=64 \
')

# scalelong:

# 0:    'orgin'
# 1:    '1/sqrt(2)-CS'
# 2:    'CS'
# 3:    'LS'
# 4:    'LS (non-share)'