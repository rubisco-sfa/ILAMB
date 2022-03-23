#!/bin/bash

source $CONDA_SOURCE
conda activate $CONDA_ENV_ROOT/ilamb

python $CMEC_CODE_DIR/parse_config.py "ILAMB/ILAMB"
