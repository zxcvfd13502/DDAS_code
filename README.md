# DDAS_code
Direct Differentiable Augmentation Search

**Commnad for search on CIFAR-10:**
```
python search.py -c confs/wresnet40x2_cifar10_b32_fast_search.yaml --dataroot ../data/ --tag 1 --save wresnet_cifar10_new_search_bs_final.pth --explore_ratio 0.99999 --cv-ratio 0.96 --param_lr 0.005 --tp_lr 0.001 --init_tp 0.35
```
