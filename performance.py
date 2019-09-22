
from sklearn.utils import shuffle
from PIL import Image
from keras.utils import to_categorical
from random import *
import numpy as np

model.load_weights('check-08-0.3533.hdf5')
scores = model.evaluate_generator(test_generator, 842)
print("Accuracy = ", scores[1])

images = []
for filename in test_files[:50]:
    im = Image.open(filename)
    cur_width = im.size[0]
    cur_height = im.size[1]
    height_fac = 113 / cur_height

    new_width = int(cur_width * height_fac)
    size = new_width, 113

    imresize = im.resize((size), Image.ANTIALIAS)  
    now_width = imresize.size[0]
    now_height = imresize.size[1]

    avail_x_points = list(range(0, now_width - 113 ))
    factor = 0.1
    pick_num = int(len(avail_x_points)*factor)

    random_startx = sample(avail_x_points,  pick_num)

    for start in random_startx:
        imcrop = imresize.crop((start, 0, start+113, 113))
        images.append(np.asarray(imcrop))

    X_test = np.array(images)

    X_test = X_test.reshape(X_test.shape[0], 113, 113, 1)
    #convert to float and normalize
    X_test = X_test.astype('float32')
    X_test /= 255
    shuffle(X_test)

predictions = model.predict(X_test, verbose =1)

predicted_writer = []
for pred in predictions:
    predicted_writer.append(np.argmax(pred))
print(len(predicted_writer))