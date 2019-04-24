#!/bin/bash

python run.py --config-file configs/R50v2_GNv2_BZ1_noPretrain.yaml SYSTEM.DEVICES [6]
python run.py --config-file configs/R50v2_GNv2_BZ1_Pretrain.yaml SYSTEM.DEVICES [6]


