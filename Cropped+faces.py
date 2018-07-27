
# coding: utf-8

# In[9]:

orglist=[]
orglist.append('/home/data/vision7/azlu/imbd_wiki_0/imdb/00/nm0029400_rm371169792_1987-12-3_2005.jpg')
orglist.append('/home/data/vision7/azlu/imbd_wiki_0/imdb/00/nm0029400_rm3971455232_1987-12-3_2005.jpg')


# In[13]:

from PIL import Image

for pic in orglist:
    image = Image.open(pic)
    image.show()


# In[11]:

croplist=[]
croplist.append('/home/data/vision7/azlu/faces/imdb_crop/00/nm0029400_rm371169792_1987-12-3_2005.jpg')
croplist.append('/home/data/vision7/azlu/faces/imdb_crop/00/nm0029400_rm3971455232_1987-12-3_2005.jpg')


# In[12]:

from PIL import Image

for pic in croplist:
    image = Image.open(pic)
    image.show()


# In[ ]:

#to save an image into Downloads folder, use the command line and type:
#scp username@tynamo.soic.indiana.edu:/full_path_to_image ~/Downloads/whatever.jpg

