


## `base_cnn`


### training


```text
is cuda available?: True
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type       | Params
---------------------------------------
0 | layer_1 | Sequential | 336   
1 | fc_1    | Linear     | 27.0 K
---------------------------------------
27.3 K    Trainable params
0         Non-trainable params
27.3 K    Total params
0.109     Total estimated model params size (MB)
Epoch 0:  99%|██████████████████████████████████████▌| 1543/1563 [00:09<00:00, 162.32it/s, loss=1.36, v_num=1]Metric val_loss improved. New best score: 1.349█████████████████████▋      | 279/313 [00:00<00:00, 431.53it/s]
Epoch 3: 100%|████████████████████████████████████████▉| 1560/1563 [00:22<00:00, 68.26it/s, loss=1.1, v_num=1]Monitored metric val_loss did not improve in the last 3 records. Best score: 1.349. Signaling Trainer to stop.
Epoch 3: 100%|█████████████████████████████████████████| 1563/1563 [00:22<00:00, 68.18it/s, loss=1.1, v_num=1]

```
### evaluating



##  `two_cnn`

### training



### evaluating



## `three_cnn`

### training


### evaluating




