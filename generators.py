
from sklearn.utils import shuffle
from PIL import Image
from keras.utils import to_categorical
from random import *
import numpy as np


def generate_data(samples, target_files,  batch_size=16, factor=0.1):
    # number of categories
    num_classes = 50
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_targets = target_files[offset:offset+batch_size]
            images = []
            targets = []
            for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                batch_target = batch_targets[i]
                im = Image.open(batch_sample)
                cur_width = im.size[0]  # current width of image
                cur_height = im.size[1]  # current height of image
                
                # computing aspect ratio of height
                height_fac = 113 / cur_height  
                # keeping the ration for width
                new_width = int(cur_width * height_fac)
                size = new_width, 113
                imresize = im.resize((size), Image.ANTIALIAS)
                now_width = imresize.size[0]
                now_height = imresize.size[1]
                # finding the vertical "X" point of image that has to be
                # less than width - 113 pixels
                avail_x_points = list(range(0, now_width - 113))
                pick_num = int(len(avail_x_points)*factor)
                # Finding random x point
                random_startx = sample(avail_x_points,  pick_num)
                for start in random_startx:
                    imcrop = imresize.crop((start, 0, start+113, 113))
                    # add to list of images and at same time writers
                    images.append(np.asarray(imcrop))
                    targets.append(batch_target)

            X_train = np.array(images)
            y_train = np.array(targets)
            X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
            X_train = X_train.astype('float32')
            X_train /= 255

            # One hot encoding
            y_train = to_categorical(y_train, num_classes)
            yield shuffle(X_train, y_train)
