# CDFSL SetFSL Test

## Arguments
- `num_object`: Number of objects.
- `layer`: Layer used for individual cross-attention.
- `withcls`: Whether to classify using the CLS token.
- `continual_layers`: Layers used for continual cross-attention.
- `train_w_qkv`, `train_w_o`: Whether to train weights in the cross-attention module.

## Example Execution Scripts

### Test 1
Uses the 11th layer, trains only `w_o`, and classifies using only the object.
```bash
python main.py -m 'setfsl' -tr -tc 'fewshot' -d 'miniimagenet' -e 100 -lr 0.001 -bs 256 -opt 'adamW' -log 'set_the_name_of_logfile' -img_size 224 -patch_size 16 -layer 11 -train_w_o
```

### Test 2
Uses layers 2, 5, 8, and 11, trains only `w_o`, and classifies using the CLS token.
```bash
python main.py -m 'setfsl' -tr -tc 'fewshot' -d 'miniimagenet' -e 100 -lr 0.001 -bs 256 -opt 'adamW' -log 'test_layer11_mean' -img_size 224 -patch_size 16 -layer 11 -train_w_o -withcls -continual_layers 2 5 8 11
```

