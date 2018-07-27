
# coding: utf-8

# In[1]:

import numpy as np
import scipy.ndimage
import scipy.misc
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


# In[2]:

def read_images_and_labels_from_text_file(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
    Returns:
       list of filenames, list of labels
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    n_classes = len(list(set(labels)));
    one_hot_labels = []
    for l in labels:
        temp = n_classes*[0]
        temp[l] = 1;
        one_hot_labels.append(temp)
    
    return filenames, labels, one_hot_labels
    


# In[3]:

thelist = read_images_and_labels_from_text_file('Downloads/imdb_faces.txt')

'''
data = []
with open('Downloads/imdb.json') as f:
    data = json.load(f)
'''


# In[4]:

filelist = thelist[0]
#if statement for the firs two, make them 90, 91, 92, etc.

labellist = thelist[1]
onehotlist=thelist[2]
newlist=[]


# In[5]:

for i in range(10):
    print(filelist[i]+": "+str(labellist[i]))


# In[6]:

list_of_90s=[]
labels_of_90s=[] 
for i in range(len(filelist)):
    if int(filelist[i][0:2])>=80:
        list_of_90s.append(filelist[i])
        labels_of_90s.append(labellist[i])


# In[7]:

for i in range(10):
    print(list_of_90s[i]+": "+str(labels_of_90s[i]))


# In[8]:

'''
remove_idx=[9534, 15783, 25660, 26430, 30575, 30576, 31237, 36485, 37394, 37397, 37398, 37402, 37403, 37404]
new_idx=[i for i in range(len(list_of_90s)) if i not in remove_idx]
new_image_list=[]

new_idx2=[i for i in range(len(labels_of_90s)) if i not in remove_idx]
new_label_list=[]
'''


# In[9]:

'''
for i in new_idx:
    new_image_list.append(list_of_90s[i])
    new_label_list.append(labels_of_90s[i])
    
new_image_list=new_image_list[:37386]
new_label_list=new_label_list[:37386]
'''


# In[10]:

new_idx3=range(len(list_of_90s))
idx_list=[]
idx_list=[i for i in new_idx3]


# In[11]:

random.shuffle(idx_list)
actual_image_list=[]
actual_label_list=[]

for i in idx_list:
    actual_image_list.append(list_of_90s[i])
    actual_label_list.append(labels_of_90s[i])


# In[12]:

'''
remove_idx_2=[37386]
new_idx_2=[i for i in range(len(new_image_list)) if i not in remove_idx_2]
new_image_list_2=[]

for i in new_idx_2:
    new_image_list_2.append(new_image_list[i])
'''


# In[13]:

string0='/home/data/vision7/azlu/faces/imdb_crop/'


# In[14]:

for i in range(len(actual_image_list)):
    actual_image_list[i]=string0+actual_image_list[i]


# In[15]:

train_img_set=actual_image_list[7388:36939]
test_img_set=actual_image_list[36939:]
val_img_set=actual_image_list[:7388]

train_lbl_set=actual_label_list[7388:36939]
test_lbl_set=actual_label_list[36939:]
val_lbl_set=actual_label_list[:7388]


# In[16]:

for i in range(10):
    print(train_img_set[i]+": "+str(train_lbl_set[i]))


# In[31]:

count_of_0s=0
count_of_1s=0
for i in actual_label_list:
    if i==0:
        count_of_0s+=1
    else:
        count_of_1s+=1

print('percent of males: ', count_of_1s/(count_of_0s+count_of_1s))

print('There are ', count_of_0s, ' 0s and ', count_of_1s, ' 1s')


# In[18]:

len(actual_image_list)


# In[19]:

def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]


# In[22]:

batches_labels = group_list(train_lbl_set, 64)
lbls = np.zeros((64, 2))
num = 0
for batch in tqdm(batches_labels):
    num += 1
    for label, ind in zip(batch, range(64)):
        label_vector=np.zeros(2)
        label_vector[label]=1
        lbls[ind] = label_vector
    np.savez('/home/data/vision7/azlu/imdb_real_80+/train_set/labels/label_tr_batch_{0}.npz'.format(num), labels=lbls)


# In[25]:

batches = group_list(train_img_set, 64)
ims = np.zeros((64, 224, 224, 3))
num = 0
for batch in tqdm(batches):
    num += 1
    for path, ind in zip(batch, range(64)):
        im = scipy.ndimage.imread(path)
        print(im.shape)
        if len(im.shape) < 3:
            im = np.stack((im,im,im), axis=2)
            
        
        ims[ind] = scipy.misc.imresize(im, (224, 224))
    np.savez('/home/data/vision7/azlu/imdb_real_80+/train_set/images/img_tr_batch_{0}.npz'.format(num), images=ims)


# ![title](/home/data/vision7/azlu/imbd_wiki_9/imdb/90/nm0000090_rm1298714112_1930-12-17_1992.jpg)

# !
