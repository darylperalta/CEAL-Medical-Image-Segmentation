from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.morphology import distance_transform_edt as edt

from data import load_train_data
from constants import *
from unet import get_unet
from utils import *

def evaluate(test_image, weights):
    """
    Performs the Cost-Effective Active Learning labeling step, giving the available training data for each iteration.
    :param test_image: Test image to segment.
    :param weights: pre-trained unet weights.

    :return: predictions: Predicted output.

    """
    # load models
    modelUncertain = get_unet(dropout=True)
    modelUncertain.load_weights(weights)
    modelPredictions = get_unet(dropout=False)
    modelPredictions.load_weights(weights)

    # predictions
    print("Computing log predictions ...\n")
    predictions = predict(test_image, modelPredictions)
    print("PREDICTION SHAPE: {}, TYPE: {}".format(predictions.shape, type(predictions)))

    return predictions

def evaluate_sets(X_train, y_train, labeled_index, unlabeled_index, weights):
    """
    Performs the Cost-Effective Active Learning labeling step, giving the available training data for each iteration.
    :param X_train: Overall training data.
    :param y_train: Overall training labels. Including the unlabeled samples to simulate the oracle annotations.
    :param labeled_index: Index of labeled samples.
    :param unlabeled_index: Index of unlabeled samples.
    :param weights: pre-trained unet weights.

    :return: predictions: Predicted output.

    """
    # load models
    modelUncertain = get_unet(dropout=True)
    modelUncertain.load_weights(weights)
    modelPredictions = get_unet(dropout=False)
    modelPredictions.load_weights(weights)

    # predictions
    print("Computing log predictions ...\n")
    predictions = predict(X_train[unlabeled_index], modelPredictions)
    print("PREDICTION SHAPE: {}, TYPE: {}".format(predictions.shape, type(predictions)))
    # print("SAMPLE SHAPE: {}, TYPE: {}".format(y_train[unlabeled_index][0].shape, type(y_train[unlabeled_index])))

    # from matplotlib import pyplot as plt
    # plt.imshow(predictions[0][0])
    # plt.show()

    mean = np.mean(X_train)  # mean for data centering
    std = np.std(X_train)

    for i in range(5):
        sample = (X_train[unlabeled_index[i]].reshape([1, 1, img_rows, img_cols])*255).astype('uint8')
        sample_prediction = (cv2.threshold(predictions[i], 0.5, 1, cv2.THRESH_BINARY)[1]*255).astype('uint8')
        cv2.imwrite('outputs/code_pred_{:02d}.png'.format(i), sample_prediction[0])
        cv2.imwrite('outputs/code_sample_{:02d}.png'.format(i), sample[0][0])
        cv2.imwrite('outputs/code_gt_{:02d}.png'.format(i), (y_train[unlabeled_index[i]][0]*255).astype('uint8'))

        sample = (X_train[unlabeled_index][i][0]*std + mean) * 255
        cv2.imwrite('outputs/pred{:02d}.png'.format(i), (predictions[i][0]*255).astype('uint8'))
        cv2.imwrite('outputs/sample{:02d}.png'.format(i), (sample).astype('uint8'))
        cv2.imwrite('outputs/gt{:02d}.png'.format(i), (y_train[unlabeled_index][i][0]*255).astype('uint8'))

    return predictions


if __name__ == '__main__':
    X_train, y_train = load_train_data()
    labeled_index = np.arange(0, nb_labeled)
    unlabeled_index = np.arange(nb_labeled, len(X_train))

    # image_rows = 224#420
    # image_cols = 224#580
    #
    # test_image = '/media/izza/IzzaClaire/Dataset/COCO/train2017/000000000009.jpg'
    # test_image = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    # test_image = cv2.resize(test_image, (image_rows, image_cols), interpolation=cv2.INTER_CUBIC)
    # test_image = np.array([test_image])

    active_weights_path ='./models/active_model6.h5'

    model = get_unet(dropout=True)
    weights = active_weights_path

    predictions = evaluate_sets(X_train, y_train, labeled_index, unlabeled_index, weights)
    # predictions = evaluate(test_image, weights)
