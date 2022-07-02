#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import cv2
import numpy as np
import matplotlib.pyplot  as plt 
import seaborn as sns
from scipy.signal import convolve2d
import scipy.misc
import scipy.ndimage
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import precision_score, confusion_matrix,recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from scipy.signal import convolve2d
import scipy.misc
import scipy.ndimage
from scipy import stats
from skimage import img_as_ubyte
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import pickle
from skimage.io import imread, imshow
from PIL import Image
import time


# In[ ]:





# In[2]:


datadir = 'D:/test project/Data/train' #root directory


categories = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib','large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
             'normal','squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']


# In[3]:


start = time.time()


# In[ ]:





# In[4]:


def preprocess(img):
    img_resized=resize(img_array,(50,50,3)) 
    gray = cv2.cvtColor(np.float32(img_resized), cv2.COLOR_RGB2GRAY)
    return gray

flat_data_arr=[]
target_arr=[]
for i in categories:
    
   print(f'loading... category : {i}')   
   path=os.path.join(datadir,i) 
   for img in os.listdir(path):  
      img_array=imread(os.path.join(path,img))
      flatData=preprocess(img_array)
      flat_data_arr.append(flatData)   
      target_arr.append(categories.index(i))   
      print(f'loaded category:{i} successfully')
      target=np.array(target_arr)


# In[5]:


plt.imshow(flat_data_arr[0],cmap='gray')
plt.show()


# In[ ]:





# In[6]:


def normal(img1):
    img1 = img1 / img1.max() #normalizes data in range 0 - 255
    img1 = 255 * img1
    img1 = img1.astype(np.uint8)
    return img1


# In[7]:


def edge_dete(i):
    smoothed_image = img_as_ubyte(gaussian(i, sigma=3, mode='constant', cval=0.0))
    #img_canny = cv2.Canny(smoothed_image,10,20)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(smoothed_image, -1, kernelx)
    img_prewitty = cv2.filter2D(smoothed_image, -1, kernely)
    prewitt=img_prewittx + img_prewitty
    return prewitt   


# In[8]:


def extract_sift_features(image):

    image_descriptors = []
    sift = cv2.SIFT_create()
    #for image in list_image:
        
    kp,des=sift.detectAndCompute(image,None)
    x=cv2.drawKeypoints(image, kp, image.copy()) 
    image_descriptors.append(x)

    return image_descriptors    


# In[9]:


images=[]
for data in flat_data_arr:
    image=normal(data)
    edge_image=edge_dete(image)
    flat_data=extract_sift_features(edge_image)
    for i in flat_data:
        
        images.append(i.flatten())


# In[ ]:





# In[10]:


plt.imshow(flat_data[0],cmap='gray')
plt.show()


# In[ ]:





# In[11]:


xtrain,xtest,ytrain,ytest=train_test_split(images,target, test_size=0.25, random_state=42,
                                           stratify=target)
model = SVC(C=1,kernel='poly',gamma='auto') 
model.fit(xtrain, ytrain)  
y_pred = model.predict(xtest)


# In[12]:


#print('Accuracy: {:.2f}'.format(accuracy_score(ytest, y_pred)))
accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred, average='micro')
recall = recall_score(ytest, y_pred , average='micro')
score = f1_score(ytest, y_pred , average='micro') 
print('accuracy:',accuracy)
print('Precision: ',precision)
print('Recall: ',recall)
print('f1_score',score)


# In[13]:


plt.figure(figsize=(23,10))
sns.set_context('notebook',font_scale = 1.5)
sns.barplot(x=['accuracy','precision','recall','score'],y=[accuracy,precision,recall,score])
plt.tight_layout()


# In[14]:


end = time.time()
print(f"Runtime of the program is{end - start}")


# In[ ]:




