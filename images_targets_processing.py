import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def get_files():
    d = {}
    # ucitavanje autora i povezivanje sa tekstom
    with open('./input/forms_for_parsing.txt') as f:
        for line in f:
            key = line.split(' ')[0]
            writer = line.split(' ')[1]
            d[key] = writer

    tmp = []
    target_list = []

    path_to_files = os.path.join('./input/data_subset', '*')

    for filename in sorted(glob.glob(path_to_files)):
        tmp.append(filename)
        image_name = filename.split('/')[-1]
        file, ext = os.path.splitext(image_name)
        only_name = file.split('\\')[1]
        parts = only_name.split('-')
        form = parts[0] + '-' + parts[1]
        for key in d:
            if key == form:
                target_list.append(str(d[form]))

    # nazivi fajlova
    img_files = np.asarray(tmp)
    # autori
    img_targets = np.asarray(target_list)

    # priprema podataka za CNN
    # podaci nisu kategoricni, pa iz tog razloga koristimo LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(img_targets)
    encoded_Y = encoder.transform(img_targets)

    # podatke delimo na trening, validacione i test podatke. Trening podaci ce zauzimati 66%, dok ce validacioni
    # i test podaci ostalih biti po 17%
    train_files, rem_files, train_targets, rem_targets = train_test_split(
        img_files, encoded_Y, train_size=0.66, random_state=52, shuffle=True)

    validation_files, test_files, validation_targets, test_targets = train_test_split(
        rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)
    return train_files, train_targets, validation_files, validation_targets, test_files, test_targets