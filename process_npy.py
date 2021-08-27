import numpy as np
import matplotlib.pyplot as plt 
import argparse 
l = 1

def avg_filter(y, al):
    return [sum(y[i : i + al])/al for i in range(0,len(y)-al+1)]
 
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--file_name', type=str, default='wresnet_cifar_new_ab_02_r_save_dict.npy')
parser.add_argument('--out_name', type=str, default='wresnet_cifar_new_ab_02_r_save_dict_smooth_1.npy')
args = parser.parse_args()
#wresnet40x2_cifar10_fast_itp035_slr_sa_b1_save_dict.npy
name = args.file_name
pdict = np.load(name,allow_pickle=True).item()

prob_dis = pdict['dis_ps']

pdict_save = pdict.copy()
y1 = pdict['tps']
y_save_smooth = avg_filter(y1, l) + [avg_filter(y1, l)[-1] for _ in range(l)]
pdict_save['tps'] = y_save_smooth
pdict_save['w0s_mt'] = y_save_smooth[:len(pdict_save['dis_ps'])]

print(pdict['tps'])
print(pdict_save['tps'])
print(len(pdict_save['tps']))
print(len(pdict_save['dis_ps']))
out_name = args.out_name
np.save(out_name, pdict_save)
