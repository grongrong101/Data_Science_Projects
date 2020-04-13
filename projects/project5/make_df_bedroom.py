#!/usr/bin/env python
# coding: utf-8

import json
from pymongo import MongoClient
from pprint import pprint
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
import random
import time, os
from tqdm import tqdm
import io
#import multiprocessing
#from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import pickle

from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.models import load_model
import numpy


base_dir = 'input/example' #for tests
#base_dir = 'images_classed' #for tests
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 100
IMG_WIDTH = 299
IMG_HEIGHT = 299

base_model = Xception(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), weights='imagenet', include_top=False)

bedroom_input_cnt = len(os.listdir(base_dir+"/bedroom"))

features = np.zeros(shape=(bedroom_input_cnt,10,10, 2048))
generator = datagen.flow_from_directory(
    base_dir,
    target_size=(299, 299),
    shuffle=False,
    batch_size=batch_size,
    class_mode=None,
    classes=['bedroom'])
list_img = [img.split('/') for img in generator.filenames]
img_df = pd.DataFrame(list_img, columns = ['class','img_name'])
img_df['listingid'] = img_df['img_name'].apply(lambda x: x.split('_')[0])
pickle_out = open("images_classed_df/df_features_bedroom.pickle","wb")
pickle.dump(img_df, pickle_out)
pickle_out.close()
i = 0
#pdb.set_trace()
for inputs_batch in tqdm(generator):
    try:
        features_batch = base_model.predict(inputs_batch)
        #pdb.set_trace()
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        i += 1
        if i * batch_size >= bedroom_input_cnt:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
        print('processing: '+ str(inputs_batch))
    except:
        print('failed: '+ inputs_batch)
#features = numpy.array(features)
print('done processing and now reshaping')
features = np.reshape(features, (bedroom_input_cnt, 10 * 10 * 2048))
print('done reshaping and now saving')
np.save('images_classed_arry/arry_features_bedroom.npy', features)
print('done')