from __future__ import print_function

import os
import gzip
import numpy as np

import cv2

from constants import *
data_path = '/home/daryl/unsupervised/CEAL-Medical-Image-Segmentation/train'
masks_path = '/home/daryl/unsupervised/CEAL-Medical-Image-Segmentation/gt'


def preprocessor(input_img):
    """
    Resize input images to constants sizes
    :param input_img: numpy array of images
    :return: numpy array of preprocessed images
    """
    output_img = np.ndarray((input_img.shape[0], input_img.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(input_img.shape[0]):
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return output_img


def create_train_data():
    """
    Generate training data numpy arrays and save them into the project path
    """

    image_rows = 420
    image_cols = 580

    images = os.listdir(data_path)
    masks = os.listdir(masks_path)
    images.sort()
    masks.sort()
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    print('images[]', images[0:10])
    print('masks[]', masks[0:10])

    for i in range(total):
        image_name = images[i]
        img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (image_rows, image_cols), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        # print('img shape', img.shape)

        # cv2.imshow('img', img)
        # cv2.waitKey()
        img = np.array([img])
        # print('img shape', img.shape)
        # print('imgs shape', imgs.shape)
        imgs[i] = img

    # for image_mask_name in masks:
    for i in range(total):
        image_mask_name = masks[i]
        img_mask = cv2.imread(os.path.join(masks_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        # img_mask = cv2.resize(img_mask, (image_rows, image_cols), interpolation=cv2.INTER_CUBIC)
        img_mask = cv2.resize(img_mask, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)

        img_mask = np.array([img_mask])
        imgs_mask[i] = img_mask

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)


def load_train_data():
    """
    Load training data from project path
    :return: [X_train, y_train] numpy arrays containing the training data and their respective masks.
    """
    print("\nLoading train data...\n")
    # X_train = np.load(gzip.open('skin_database/imgs_train.npy.gz'))
    # y_train = np.load(gzip.open('skin_database/imgs_mask_train.npy.gz'))

    X_train = np.load('imgs_train.npy')
    y_train = np.load('imgs_mask_train.npy')

    X_train = preprocessor(X_train)
    y_train = preprocessor(y_train)

    X_train = X_train.astype('float32')

    mean = np.mean(X_train)  # mean for data centering
    std = np.std(X_train)  # std for data normalization

    X_train -= mean
    X_train /= std

    y_train = y_train.astype('float32')
    y_train /= 255.  # scale masks to [0, 1]
    return X_train, y_train

def get_data_mean():
    X_train = np.load('imgs_train.npy')
    y_train = np.load('imgs_mask_train.npy')

    X_train = X_train.astype('float32')

    mean = np.mean(X_train)  # mean for data centering
    std = np.std(X_train)  # std for data normalization

    return mean, std

if __name__ == '__main__':
    create_train_data()
