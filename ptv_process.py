import os
import shutil
import random

dir_path = './train'
proxy_train = './proxy_train_s'
proxy_val = './proxy_val_s'
dir_ls_all=os.listdir(dir_path)
dir_ls = random.sample(dir_ls_all,120)
print(len(dir_ls))
print(dir_ls)
class_set=set()
for dir in dir_ls:
    print(dir)
    pt_class_path = os.path.join(proxy_train, dir)
    if os.path.exists(pt_class_path):
        pass
    else:
        os.mkdir(pt_class_path)
    pv_class_path = os.path.join(proxy_val, dir)
    if os.path.exists(pv_class_path):
        pass
    else:
        os.mkdir(pv_class_path)
    
    class_path=os.path.join(dir_path, dir)
    image_ls = os.listdir(class_path)

    pt_images = random.sample(image_ls[0:500],50)
    pv_images = random.sample(image_ls[500:],20)

    for pt_names in pt_images:
        image_path = os.path.join(class_path, pt_names)
        shutil.copy(image_path, pt_class_path)
    
    for pv_names in pv_images:
        image_path = os.path.join(class_path, pv_names)
        shutil.copy(image_path, pv_class_path)
    class_set.add(dir)

print(class_set)
print(len(class_set))
