# DDAS_code
Direct Differentiable Augmentation Search for ImageNet
1. Make the dataset for augmentation search on ImageNet

```bash
├── imagenet
│   ├── train/
│   ├── val/
│   ├── proxy_train_s/
│   ├── proxy_val_s/
│   ├── ptv_process.py
```
```
python ptv_process.py
```
2. Run the search code on ImageNet
```
python search_proxy_td.py -c confs/resnet18_b64_reduce_search.yaml --dataroot /path/to/your/imagenet/ --save resnet18_b64_imagenet_new_search.pth --loader_num 4 --param_lr 0.003 --tp_lr 0.001 --explore_ratio 0.9999 --init_tp 0.35 --tag 1
```
3. Process the searched augmentation policy
```
python process_npy.py --file_name resnet18_b64_imagenet_new_search_try_save_dict.npy --out_name resnet18_imagenet_smoothed_save_dict.npy
```
4. Run the training code
```
python -m torch.distributed.launch --nproc_per_node=8 train_apex_pos_o2_step.py -c confs/resnet50_b512_rwt2_270e_step.yaml --dataroot /path/to/your/imagenet/ --tag 1 --save resnet50_step_ddas_bs1.pth --load_tp resnet18_imagenet_smoothed_save_dict.npy
```
