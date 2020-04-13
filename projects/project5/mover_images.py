#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import shutil
import os
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions

from keras.models import load_model
m = load_model('room_classifier')

to_class = {0: 'bathroom',
 1: 'bedroom',
 2: 'exterior',
 3: 'kitchen',
 4: 'living',
 5: 'plan'}

IMG_WIDTH, IMG_HEIGHT = 299, 299 

# makes the prediction of the file path image passed as parameter 
def predict(file, model, to_class):
    im = load_img(file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    index = model.predict(x).argmax()
    return to_class[index]


# execute this when you want to save the model
#m.save('room_classifier')

# execute this when you want to load the model


#process_images('images')
def process_images(fld):
    path = 'images'
    dest_path = 'images_classed'
    for img in os.listdir(path+"/"+fld):
        try:
            file = img.split('_')[0]
            source_dir = path+"/" +str(file)+"/"+str(img)
            room_class = predict(source_dir, m, to_class)
            dest_source = dest_path+"/"+room_class
            shutil.copy(src=source_dir,dst=dest_source)
        except:
            print('Failed: ' + source_dir)

            
path = 'images'
source_dir_fld = os.listdir(path)
dest_path = 'images_classed'
dest_living_path = 'images_classed/living'
dest_bathroom_path = 'images_classed/bathroom'
dest_bedroom_path = 'images_classed/bedroom'
dest_exterior_path = 'images_classed/exterior'
dest_plan_path = 'images_classed/plan'
dest_kitchen_path = 'images_classed/kitchen'
incomp_fld = []

for fld in source_dir_fld:
    fld_dir = path+"/" +str(fld)
    for img in os.listdir(fld_dir):
        count=0
        if os.path.exists(dest_living_path+"/"+img):
            count+=1
        if os.path.exists(dest_bathroom_path+"/"+img):
            count+=1
        if os.path.exists(dest_bedroom_path+"/"+img):
            count+=1
        if os.path.exists(dest_exterior_path+"/"+img):
            count+=1
        if os.path.exists(dest_plan_path+"/"+img):
            count+=1
        if os.path.exists(dest_kitchen_path+"/"+img):
            count+=1
        if count == 0:
            incomp_fld.append(fld)            

incomp_fld = set(incomp_fld)    

print(len(incomp_fld))

#image_dir = os.listdir(path)
num_cores = multiprocessing.cpu_count()

if __name__ == '__main__':
    processed_list = Parallel(n_jobs=num_cores-1)(delayed(process_images)(i) for i in tqdm(incomp_fld))
