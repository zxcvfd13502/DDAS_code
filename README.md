# DDAS_code
Direct Differentiable Augmentation Search

**Commnad for search on CIFAR-10:**
```
python search.py -c confs/wresnet40x2_cifar10_b32_fast_search.yaml --dataroot ../data/ --tag 1 --save wresnet_cifar10_new_search_bs_final.pth --explore_ratio 0.99999 --cv-ratio 0.96 --param_lr 0.005 --tp_lr 0.001 --init_tp 0.35
```
**Commnad for training on CIFAR-10:**
```
python process_npy.py --file_name wresnet_cifar10_new_search_bs_final_save_dict.npy --out_name wresnet_cifar10_new_search_smoothed.npy
python train.py -c confs/wresnet28x10_cifar10_b128_rwaug.yaml --dataroot ../data/ --save wresnet_cifar10_new_search2_bs1.pth --load_tp wresnet_cifar10_new_search_smoothed.npy --tag 1
```
**Commnad for search on CIFAR-100:**
```
python search.py -c confs/wresnet40x2_cifar100_b32_fast_search.yaml --dataroot ../data/ --save wresnet_cifar_new_search_bs_final.pth --tag 1 --explore_ratio 0.9999 --cv-ratio 0.96 --param_lr 0.005 --tp_lr 0.001 --init_tp 0.35
```
**Commnad for training on CIFAR-100:**
```
python process_npy.py --file_name wresnet_cifar_new_search_bs_final_save_dict.npy --out_name wresnet_cifar100_smoothed.npy
python train.py -c confs/wresnet28x10_cifar100_b128_rwaug_train.yaml --dataroot ../data/ --save wresnet_cifar100_new_bs1.pth --load_tp wresnet_cifar100_smoothed.npy --tag 1
```
