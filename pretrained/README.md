# Optional pretrained initialization

Place a compatible MSRL4SSC Stage-1 checkpoint in this directory and pass its
path through `train.py --pretrained`. The checkpoint must contain a
`CrossModalSSL` state dictionary compatible with `model.py`.
