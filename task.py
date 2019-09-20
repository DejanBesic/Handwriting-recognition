from __future__ import division
from subprocess import check_output
import numpy as np
import os
from generators import generate_data
from cnn_model import build_model, resize_image, train_cnn
from images_targets_processing import get_files


def main():
    train_files, train_targets, validation_files, validation_targets, test_files, test_targets = get_files()
    # kreiranje generatora za treniranje, validaciju i testiranje
    batch_size = 16
    train_generator = generate_data(
        train_files, train_targets, batch_size=batch_size, factor=0.3)
    validation_generator = generate_data(
        validation_files, validation_targets, batch_size=batch_size, factor=0.3)
    test_generator = generate_data(
        test_files, test_targets, batch_size=batch_size, factor=0.1)

if __name__ == '__main__':
    main()
