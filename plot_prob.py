import numpy as np
import matplotlib.pyplot as plt  
l = 6
def prob_sum(name_ls, prob_ls):
    dict_op = {}
    dict_class = {}
    dict_mag = {}
    color_ops = ['AutoContrast', 'Equalize', 'Invert', 'Blur', 'Smooth','Posterize', 'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness']
    geo_ops = ['FlipLR', 'FlipUD', 'Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY']
    dict_cg = {'color':[],'geo':[]}
    for name in name_ls:
        dict_op[name] = []
        class_name = name.split('.')[0]
        mag = name.split('.')[1]
        if class_name not in dict_class:
            dict_class[class_name] = []
        if mag not in dict_mag:
            dict_mag[mag] = []
    for i in range(len(prob_ls)):
        probs = prob_ls[i]
        assert len(probs) == len(name_ls)
        for j in range(len(name_ls)):
            op_name = name_ls[j]
            class_name = op_name.split('.')[0]
            mag = op_name.split('.')[1]
            dict_op[op_name].append(probs[j])
            if len(dict_class[class_name]) == i:
                dict_class[class_name].append(probs[j])
            else:
                dict_class[class_name][-1] += probs[j]
            if len(dict_mag[mag]) == i:
                dict_mag[mag].append(probs[j])
            else:
                dict_mag[mag][-1] += probs[j]
            if class_name in color_ops:
                if len(dict_cg['color']) == i:
                    dict_cg['color'].append(probs[j])
                else:
                    dict_cg['color'][-1] += probs[j]
            if class_name in geo_ops:
                if len(dict_cg['geo']) == i:
                    dict_cg['geo'].append(probs[j])
                else:
                    dict_cg['geo'][-1] += probs[j]

    return dict_op, dict_class, dict_mag, dict_cg
def plot_dict(pdict):
    move_prob = []
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    probs = []
    for key in pdict.keys():
        if len(move_prob) == 0:
            move_prob = np.array(pdict[key])
            x = [i for i in range(len(move_prob))]
            probs.append(0)
        else:
            last_prob = move_prob.copy()
            probs.append(last_prob)
            move_prob += pdict[key]
        ax1.plot(x,move_prob,label=key)
    probs.append(move_prob)
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines)+1)]
    
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])
        ax1.fill_between(x, y1=probs[i+1], y2=probs[i],facecolor = colors[i])
    ax1.legend(loc=2)
    plt.show()
    
def cut_p(dis_ps):
    cuted_dis_ps = []
    for p in dis_ps:
        p_mean = np.mean(p)
        dis_tmp = []
        for pi in p:
            dis_tmp.append((pi>=p_mean)*pi)
        cuted_dis_ps.append(np.array(dis_tmp)/sum(dis_tmp))
        print(len(cuted_dis_ps[-1]))
    return cuted_dis_ps

def avg_filter(y, al):
    return [sum(y[i : i + al])/al for i in range(0,len(y)-al+1)]
name = 'resnet18_b64_imagenet_reduced_td_new_search_tpi10_fix_b1_save_dict.npy'
pdict = np.load(name,allow_pickle=True).item()
aug_name_ls =['identity.', 'FlipLR.', 'FlipUD.', 'AutoContrast.', 'Equalize.', 'Invert.', 'Blur.', 'Smooth.', 'Rotate.7', 'Rotate.14', 'Posterize.7', 'Posterize.14', 'ShearX.7', 'ShearX.14', 'ShearY.7', 'ShearY.14', 'TranslateX.7', 'TranslateX.14', 'TranslateY.7', 'TranslateY.14', 'Solarize.7', 'Solarize.14', 'Cutout.7', 'Cutout.14', 'Color.7', 'Color.14', 'Contrast.7', 'Contrast.14', 'Brightness.7', 'Brightness.14', 'Sharpness.7', 'Sharpness.14']


prob_dis = pdict['dis_ps']
print(prob_dis)
print(len(prob_dis[-1]))
dict_op, dict_class, dict_mag, dict_cg = prob_sum(aug_name_ls, prob_dis)
print(dict_class.keys())
print(dict_op)
print(dict_class)
print(dict_mag)
# plot_dict(dict_op)
# plot_dict(dict_class)
# plot_dict(dict_mag)
# plot_dict(dict_cg)
tp_ls = []
for tpa in pdict['tps']:
    tp_ls.append(tpa)
pdict['tps'] = np.array(tp_ls)
y1 = pdict['tps']
print(pdict)
print(y1)
print(len(y1))
y_ab1 = y1
y1_avg = avg_filter(y_ab1, l)
x1 = [i for i in range(len(y1_avg))]
y2 = y1
y_ab2 = y1
y2_avg = avg_filter(y_ab2, l)
x2 = [i for i in range(len(y2_avg))]
#l1 = plt.plot(x1,y1_avg,'r--',label='sqrt1')
#l1 = plt.plot(x2,y2_avg,'g--',label='sqrt3')
epochs_0 = 10
f1 = np.polyfit(x1, y1_avg, 1)
f2 = np.polyfit(x2, y2_avg, 1)
print("f1")
print(f1)
f1[0] /= (epochs_0/10)
print(f1)
print("f2")
print(f2)
f2[0] /= (epochs_0/10)
print(f2)
x0 = [i for i in range(epochs_0)]
p1 = np.poly1d(f1)
p2 = np.poly1d(f2)
y1p = p1(x0)
y2p = p2(x0)
print("y1p")
print(y1p)
print("y2p")
print(y2p)
np.save('tp1.npy', y1p)
np.save('tp2.npy', y2p)
pdict_save = pdict.copy()
y_save_smooth = avg_filter(y1, l) + [avg_filter(y1, l)[-1] for _ in range(l)]
pdict_save['tps'] = y_save_smooth
pdict_save['w0s_mt'] = y_save_smooth[:len(pdict_save['dis_ps'])]

print(pdict['tps'])
print(pdict_save['tps'])
print(len(pdict_save['tps']))
print(len(pdict_save['dis_ps']))
np.save(name[:-4]+'_smooth_'+str(l)+'.npy', pdict_save)
#pdict_save['dis_ps'] = [pdict['dis_ps'][-1]]*len(pdict['dis_ps'])
#pdict_save['tps'] = 1-y1p
#print(pdict_save)
#np.save(name[:-4]+'_cut_fit_'+str(l)+'.npy', pdict_save)
#pdict_save['dis_ps']  =  [pdict['dis_ps'][10*(i//10)] for i in range(len(pdict['dis_ps']))]
#np.save(name[:-4]+'_slice_10_fit_'+'.npy', pdict_save)
#pdict_save['dis_ps'] = cut_p(pdict_save['dis_ps'])
# np.save(name[:-4]+'_cuted'+'_smooth_'+str(l)+'.npy', pdict_save)
#prob_dis = pdict_save['dis_ps']
#print(prob_dis)
#print(len(prob_dis[-1]))
#dict_op, dict_class, dict_mag, dict_cg = prob_sum(aug_name_ls, prob_dis)
#print(dict_class.keys())
#print(dict_op)
#print(dict_class)
#print(dict_mag)
#plot_dict(dict_op)
#plot_dict(dict_class)
#plot_dict(dict_mag)
#plot_dict(dict_cg)

#plt.plot(x1,y1_avg,'ro',x2,y2_avg,'g+',x0, y1p, 'r-',x0, y2p, 'g-')
# plt.plot(x2,y2_avg,'ro-')
# plt.xlabel('Epoch', fontdict={'size':17})
# plt.ylabel('Total Probability', fontdict={'size':17})
# plt.yticks(size = 15)
# plt.xticks(size = 15)
# plt.legend()
# plt.show()
