from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.morphology import distance_transform_edt as edt

from constants import *
from unet import get_unet, get_unet_multi
from data_split import  get_colored_segmentation_image

def range_transform(sample):
    """
    Range normalization for 255 range of values
    :param sample: numpy array for normalize
    :return: normalize numpy array
    """
    if (np.max(sample) == 1):
        sample = sample * 255

    m = 255 / (np.max(sample) - np.min(sample))
    n = 255 - m * np.max(sample)
    return (m * sample + n) / 255


def predict(data, model):
    """
    Data prediction for a given model
    :param data: input data to predict.
    :param model: unet model.
    :return: predictions.
    """
    return model.predict(data, verbose=0)


def compute_uncertain(sample, prediction, model):
    """
    Computes uncertainty map for a given sample and its prediction for a given model, based on the
    number of step predictions defined in constants file.
    :param sample: input sample.
    :param prediction: input sample prediction.
    :param model: unet model with Dropout layers.
    :return: uncertainty map.
    """
    X = np.zeros([1, img_rows, img_cols])

    for t in range(nb_step_predictions):
        prediction = model.predict(sample, verbose=0).reshape([1, img_rows, img_cols])
        X = np.concatenate((X, prediction))

    X = np.delete(X, [0], 0)

    if (apply_edt):
        # apply distance transform normalization.
        var = np.var(X, axis=0)
        transform = range_transform(edt(prediction))
        return np.sum(var * transform)

    else:
        return np.sum(np.var(X, axis=0))

def compute_uncertain_multi(sample, model, n_classes =14):
    """
    Computes uncertainty map for a given sample and its prediction for a given model, based on the
    number of step predictions defined in constants file.
    :param sample: input sample.
    :param prediction: input sample prediction.
    :param model: unet model with Dropout layers.
    :return: uncertainty map.
    """
    X = np.zeros([1, img_rows, img_cols, n_classes])
    cv2.imshow('sample', sample[0])
    cv2.waitKey(0)
    for t in range(nb_step_predictions):

        prediction = model.predict(sample, verbose=0).reshape([1, img_rows, img_cols, n_classes])

        pred_show = ((prediction[0]*255)).astype(np.uint8).argmax(axis=2)
        pred_show_colored = get_colored_segmentation_image(pred_show, 14).astype(np.uint8)
        cv2.imshow('pred', pred_show_colored)
        cv2.waitKey()

        X = np.concatenate((X, prediction))

    X = np.delete(X, [0], 0)
    X_var = np.var(X, axis=0)
    print('xvar shape',X_var.shape)
    X_var_sum = np.sum(X_var, axis = 2)
    print('shape sum var',X_var_sum.shape)
    print('X var max', np.max(X_var))
    print('X var min', np.min(X_var))
    print('X varsum max', np.max(X_var_sum))
    print('X varsum min', np.min(X_var_sum))
    cv2.imshow('var', (X_var_sum*255*100).astype(np.uint8))
    cv2.waitKey()


def detect_blobs(X_var_sum, bg_mask):

    var_max = np.max(X_var_sum)
    var_min = np.min(X_var_sum)
    X_var_norm= (X_var_sum-var_min)/(var_max-var_min) *255

    # erode bg mask
    kernel = np.ones((5,5), np.uint8)
    bg_erosion = cv2.erode(bg_mask, kernel, iterations=1)
    bg_erosion = cv2.erode(bg_mask, kernel, iterations=2)

    # cv2.imshow('Erosion', img_erosion)

    # X_masked = X_var_norm * bg_mask

    # cv2.imshow('bg_mask', (bg_mask*255).astype(np.uint8))
    # cv2.waitKey()
    # cv2.imshow('bg_eroded', (bg_erosion*255).astype(np.uint8))
    # cv2.waitKey()

    X_masked = X_var_norm * bg_erosion

    var_max = np.max(X_masked)
    var_min = np.min(X_masked)

    print('max mask', np.max(var_max))
    X_masked_norm= (((X_masked-var_min)/(var_max-var_min)) *255).astype(np.uint8)
    print('max mask2', np.max(X_masked_norm))


    return X_masked_norm

def compute_uncertain_multi_img(sample, model, bg_gt, n_classes =14, is_blur = True, is_bgmask = True):
    """
    Computes uncertainty map for a given sample and its prediction for a given model, based on the
    number of step predictions defined in constants file.
    :param sample: input sample.
    :param prediction: input sample prediction.
    :param model: unet model with Dropout layers.
    :return: uncertainty map.
    """
    X = np.zeros([1, img_rows, img_cols, n_classes])
    bg_mask =bg_gt.astype(np.uint8)
    # print('bg shape', bg_mask.shape)
    # cv2.imshow('bg_mask', (bg_mask*255).astype(np.uint8))
    # cv2.waitKey()
    for t in range(nb_step_predictions):

        prediction = model.predict(sample, verbose=0).reshape([1, img_rows, img_cols, n_classes])

        # pred_show = ((prediction[0]*255)).astype(np.uint8).argmax(axis=2)
        # pred_show_colored = get_colored_segmentation_image(pred_show, 14).astype(np.uint8)
        # cv2.imshow('pred', pred_show_colored)
        # cv2.waitKey()

        X = np.concatenate((X, prediction))

    X = np.delete(X, [0], 0)
    X_var = np.var(X, axis=0)
    # print('xvar shape',X_var.shape)
    X_var_sum = np.sum(X_var, axis = 2)/n_classes
    if is_blur:
        X_var_sum = cv2.GaussianBlur(X_var_sum,(5,5),0)

    var_max = np.max(X_var_sum)
    var_min = np.min(X_var_sum)
    X_var_norm = ((X_var_sum-var_min)/(var_max-var_min)) *255



    print('max dtect blob', var_max)

    print('shape sum var',X_var_sum.shape)
    print('X var max', np.max(X_var))
    print('X var min', np.min(X_var))
    print('X varsum max', np.max(X_var_sum))
    print('X varsum min', np.min(X_var_sum))
    transform_input = 1 - prediction[:,:,:,0] # foreground in bg
    transform_input[transform_input < 0.2] = 0
    print('transform shape',transform_input.shape)
    # transform = range_transform(edt(prediction[:,:,:,0]))
    transform = range_transform(edt(transform_input))

    x_transformed = (X_var_sum * transform)[0]
    x_transformed_norm = ((x_transformed-np.min(x_transformed))/(-np.max(x_transformed)-np.min(x_transformed))) *255
    if is_bgmask:
        X_post = detect_blobs(X_var_sum, bg_mask)


    else:
        X_post = X_var_norm

        # x_transformed_masked = x_transformed_norm
    x_transformed_masked = detect_blobs(x_transformed, bg_mask)
    # x_try = range_transform(X_var_sum)


    # threshold
    retval, thresholded = cv2.threshold(X_post.astype(np.uint8),127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('orig Image',X_var_norm.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.imshow('Thresholded Image',thresholded.astype(np.uint8))
    # cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100 :
            contour_list.append(contour)

    # input_sample = sample[0].astype(np.uint8)
    # cv2.drawContours(input_sample, contour_list,  -1, (255,0,0), 2)
    # cv2.imshow('Objects Detected',input_sample)
    # cv2.waitKey(0)

    return np.sum(np.var(X, axis=0)), X_var_norm.astype(np.uint8), x_transformed_masked.astype(np.uint8), X_post.astype(np.uint8), X_var_sum, contour_list


def interval(data, start, end):
    """
    Returns the index of data within range values from start to end.
    :param data: numpy array of data.
    :param start: starting value.
    :param end: ending value.
    :return: numpy array of data index.
    """
    p = np.where(data >= start)[0]
    return p[np.where(data[p] < end)[0]]


def get_pseudo_index(uncertain, nb_pseudo):
    """
    Gives the index of the most certain data, to make the pseudo annotations.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_pseudo: Total of pseudo samples.
    :return: Numpy array of index.
    """
    h = np.histogram(uncertain, 80)

    pseudo = interval(uncertain, h[1][np.argmax(h[0])], h[1][np.argmax(h[0]) + 1])
    np.random.shuffle(pseudo)
    return pseudo[0:nb_pseudo]


def random_index(uncertain, nb_random):
    """
    Gives the index of the random selection to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_random: Total of random samples.
    :return: Numpy array of index.
    """
    histo = np.histogram(uncertain, 80)
    # TODO: automatic selection of random range
    index = interval(uncertain, histo[1][np.argmax(histo[0]) + 6], histo[1][len(histo[0]) - 33])
    np.random.shuffle(index)
    return index[0:nb_random]


def no_detections_index(uncertain, nb_no_detections):
    """
    Gives the index of the no detected samples to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_no_detections: Total of no detected samples.
    :return: Numpy array of index.
    """
    return np.argsort(uncertain)[0:nb_no_detections]


def most_uncertain_index(uncertain, nb_most_uncertain, rate):
    """
     Gives the index of the most uncertain samples to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_most_uncertain: Total of most uncertain samples.
    :param rate: Hash threshold to define the most uncertain area. Bin of uncertainty histogram.
    TODO: automatic selection of rate.
    :return: Numpy array of index.
    """
    data = np.array([]).astype('int')

    histo = np.histogram(uncertain, 80)

    p = np.arange(len(histo[0]) - rate, len(histo[0]))  # index of last bins above the rate
    pr = np.argsort(histo[0][p])  # p index accendent sorted
    cnt = 0
    pos = 0
    index = np.array([]).astype('int')

    while (cnt < nb_most_uncertain and pos < len(pr)):
        sbin = histo[0][p[pr[pos]]]

        index = np.append(index, p[pr[pos]])
        cnt = cnt + sbin
        pos = pos + 1

    for i in range(0, pos):
        data = np.concatenate((data, interval(uncertain, histo[1][index[i]], histo[1][index[i] + 1])))

    np.random.shuffle(data)
    print('most uncertain',nb_most_uncertain)
    return data[0:nb_most_uncertain]


def get_oracle_index(uncertain, nb_no_detections, nb_random, nb_most_uncertain, rate):
    """
    Gives the index of the unlabeled data to annotated at specific CEAL iteration, based on their uncertainty.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_no_detections: Total of no detected samples.
    :param nb_random: Total of random samples.
    :param nb_most_uncertain: Total of most uncertain samples.
    :param rate: Hash threshold to define the most uncertain area. Bin of uncertainty histogram.
    :return: Numpy array of index.
    """
    return np.concatenate((no_detections_index(uncertain, nb_no_detections), random_index(uncertain, nb_random),
                           most_uncertain_index(uncertain, nb_most_uncertain, rate)))
    # return most_uncertain_index(uncertain, nb_most_uncertain, rate)
    # return no_detections_index(uncertain, nb_no_detections)

def get_oracle_index_multi(uncertain, nb_no_detections, nb_random, nb_most_uncertain, rate):
    """
    Gives the index of the unlabeled data to annotated at specific CEAL iteration, based on their uncertainty.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_no_detections: Total of no detected samples.
    :param nb_random: Total of random samples.
    :param nb_most_uncertain: Total of most uncertain samples.
    :param rate: Hash threshold to define the most uncertain area. Bin of uncertainty histogram.
    :return: Numpy array of index.
    """
    # return np.concatenate((no_detections_index(uncertain, nb_no_detections), random_index(uncertain, nb_random),
                           # most_uncertain_index(uncertain, nb_most_uncertain, rate)))

    return no_detections_index(uncertain, nb_no_detections)


def compute_dice_coef(y_true, y_pred):
    """
    Computes the Dice-Coefficient of a prediction given its ground truth.
    :param y_true: Ground truth.
    :param y_pred: Prediction.
    :return: Dice-Coefficient value.
    """
    smooth = 1.  # smoothing value to deal zero denominators.
    y_true_f = y_true.reshape([1, img_rows * img_cols])
    y_pred_f = y_pred.reshape([1, img_rows * img_cols])
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def compute_train_sets(X_train, y_train, labeled_index, unlabeled_index, weights, iteration):
    """
    Performs the Cost-Effective Active Learning labeling step, giving the available training data for each iteration.
    :param X_train: Overall training data.
    :param y_train: Overall training labels. Including the unlabeled samples to simulate the oracle annotations.
    :param labeled_index: Index of labeled samples.
    :param unlabeled_index: Index of unlabeled samples.
    :param weights: pre-trained unet weights.
    :param iteration: Currently CEAL iteration.

    :return: X_labeled_train: Update of labeled training data, adding the manual and pseudo annotations.
    :return: y_labeled_train: Update of labeled training labels, adding the manual and pseudo annotations.
    :return: labeled_index: Update of labeled index, adding the manual annotations.
    :return: unlabeled_index: Update of labeled index, removing the manual annotations.

    """
    print("\nActive iteration " + str(iteration))
    print("-" * 50 + "\n")

    # load models
    modelUncertain = get_unet(dropout=True)
    modelUncertain.load_weights(weights)
    modelPredictions = get_unet(dropout=False)
    modelPredictions.load_weights(weights)

    # predictions
    print("Computing log predictions ...\n")
    predictions = predict(X_train[unlabeled_index], modelPredictions)

    for index in range(0, 10):
        sample = X_train[unlabeled_index[index]].reshape([1, 1, img_rows, img_cols])
        sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        #print(sample.shape, sample_prediction.shape, y_train[unlabeled_index[index]][0].shape)

        # cv2.imwrite("./outputs/pred_{:03d}_{:02d}.png".format(iteration, index), sample_prediction[0])
        # cv2.imwrite("./outputs/sample_{:03d}_{:02d}.png".format(iteration, index), sample[0][0])
        # cv2.imwrite("./outputs/gt_{:03d}_{:02d}.png".format(iteration, index), y_train[unlabeled_index[index]][0])
        #accuracy[index] = compute_dice_coef(y_train[unlabeled_index[index]][0], sample_prediction)

    uncertain = np.zeros(len(unlabeled_index))
    accuracy = np.zeros(len(unlabeled_index))

    print("Computing train sets ...")
    for index in range(0, len(unlabeled_index)):

        if index % 100 == 0:
            print("completed: " + str(index) + "/" + str(len(unlabeled_index)))

        sample = X_train[unlabeled_index[index]].reshape([1, 1, img_rows, img_cols])

        sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')

        accuracy[index] = compute_dice_coef(y_train[unlabeled_index[index]][0], sample_prediction)
        uncertain[index] = compute_uncertain(sample, sample_prediction, modelUncertain)

    np.save(global_path + "logs/uncertain" + str(iteration), uncertain)
    np.save(global_path + "logs/accuracy" + str(iteration), accuracy)

    oracle_index = get_oracle_index(uncertain, nb_no_detections, nb_random, nb_most_uncertain,
                                    most_uncertain_rate)

    oracle_rank = unlabeled_index[oracle_index]
    print('oracle index', oracle_index)
    print('oracle rank', oracle_rank)
    np.save(global_path + "ranks/oracle" + str(iteration), oracle_rank)
    np.save(global_path + "ranks/oraclelogs" + str(iteration), oracle_index)

    labeled_index = np.concatenate((labeled_index, oracle_rank))

    if (iteration >= pseudo_epoch):

        pseudo_index = get_pseudo_index(uncertain, nb_pseudo_initial + (pseudo_rate * (iteration - pseudo_epoch)))
        pseudo_rank = unlabeled_index[pseudo_index]

        np.save(global_path + "ranks/pseudo" + str(iteration), pseudo_rank)
        np.save(global_path + "ranks/pseudologs" + str(iteration), pseudo_index)

        X_labeled_train = np.concatenate((X_train[labeled_index], X_train[pseudo_index]))
        y_labeled_train = np.concatenate((y_train[labeled_index], predictions[pseudo_index]))

    else:
        X_labeled_train = np.concatenate((X_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])
        y_labeled_train = np.concatenate((y_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])

    unlabeled_index = np.delete(unlabeled_index, oracle_index, 0)

    return X_labeled_train, y_labeled_train, labeled_index, unlabeled_index


def compute_train_sets_multi(X_train, y_train, labeled_index, unlabeled_index, weights, iteration):
    """
    Performs the Cost-Effective Active Learning labeling step, giving the available training data for each iteration.
    :param X_train: Overall training data.
    :param y_train: Overall training labels. Including the unlabeled samples to simulate the oracle annotations.
    :param labeled_index: Index of labeled samples.
    :param unlabeled_index: Index of unlabeled samples.
    :param weights: pre-trained unet weights.
    :param iteration: Currently CEAL iteration.

    :return: X_labeled_train: Update of labeled training data, adding the manual and pseudo annotations.
    :return: y_labeled_train: Update of labeled training labels, adding the manual and pseudo annotations.
    :return: labeled_index: Update of labeled index, adding the manual annotations.
    :return: unlabeled_index: Update of labeled index, removing the manual annotations.

    """
    print("\nActive iteration " + str(iteration))
    print("-" * 50 + "\n")

    # load models
    modelUncertain = get_unet_multi(dropout=True,channels=3,n_class=14)
    modelUncertain.load_weights(weights)
    modelPredictions = get_unet_multi(dropout=False,channels=3,n_class=14)
    modelPredictions.load_weights(weights)

    # predictions
    print("Computing log predictions ...\n")
    predictions = predict(X_train[unlabeled_index], modelPredictions)

    # for index in range(0, 10):
    #     sample = X_train[unlabeled_index[index]].reshape([1, 1, img_rows, img_cols])
    #     sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        #print(sample.shape, sample_prediction.shape, y_train[unlabeled_index[index]][0].shape)

        # cv2.imwrite("./outputs/pred_{:03d}_{:02d}.png".format(iteration, index), sample_prediction[0])
        # cv2.imwrite("./outputs/sample_{:03d}_{:02d}.png".format(iteration, index), sample[0][0])
        # cv2.imwrite("./outputs/gt_{:03d}_{:02d}.png".format(iteration, index), y_train[unlabeled_index[index]][0])
        #accuracy[index] = compute_dice_coef(y_train[unlabeled_index[index]][0], sample_prediction)

    uncertain = np.zeros(len(unlabeled_index))
    accuracy = np.zeros(len(unlabeled_index))

    print("Computing train sets ...")
    for index in range(0, len(unlabeled_index)):

        if index % 100 == 0:
            print("completed: " + str(index) + "/" + str(len(unlabeled_index)))

        sample = X_train[unlabeled_index[index]].reshape([1, img_rows, img_cols, 3])
        print('sample', sample.shape)
        # sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        # print('sample pred', sample_prediction.shape)
        # accuracy[index] = compute_dice_coef(y_train[unlabeled_index[index]][0], sample_prediction)
        uncertain[index] = compute_uncertain_multi(sample, modelUncertain)

    np.save(global_path + "logs/uncertain" + str(iteration), uncertain)
    # np.save(global_path + "logs/accuracy" + str(iteration), accuracy)

    oracle_index = get_oracle_index_multi(uncertain, nb_no_detections, nb_random, nb_most_uncertain,
                                    most_uncertain_rate)

    oracle_rank = unlabeled_index[oracle_index]
    print('oracle index', oracle_index)
    print('oracle rank', oracle_rank)
    np.save(global_path + "ranks/oracle" + str(iteration), oracle_rank)
    np.save(global_path + "ranks/oraclelogs" + str(iteration), oracle_index)

    labeled_index = np.concatenate((labeled_index, oracle_rank))

    if (iteration >= pseudo_epoch):

        pseudo_index = get_pseudo_index(uncertain, nb_pseudo_initial + (pseudo_rate * (iteration - pseudo_epoch)))
        pseudo_rank = unlabeled_index[pseudo_index]

        # np.save(global_path + "ranks/pseudo" + str(iteration), pseudo_rank)
        # np.save(global_path + "ranks/pseudologs" + str(iteration), pseudo_index)

        X_labeled_train = np.concatenate((X_train[labeled_index], X_train[pseudo_index]))
        y_labeled_train = np.concatenate((y_train[labeled_index], predictions[pseudo_index]))

    else:
        X_labeled_train = np.concatenate((X_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])
        y_labeled_train = np.concatenate((y_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])

    unlabeled_index = np.delete(unlabeled_index, oracle_index, 0)

    return X_labeled_train, y_labeled_train, labeled_index, unlabeled_index


def data_generator():
    """
    :return: Keras data generator. Data augmentation parameters.
    """
    return ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.2,
        rotation_range=40,
        horizontal_flip=True)


def log(history, step, log_file):
    """
    Writes the training history to the log file.
    :param history: Training history. Dictionary with training and validation scores.
    :param step: Training step
    :param log_file: Log file.
    """
    for i in range(0, len(history.history["loss"])):
        if len(history.history.keys()) == 4:
            log_file.write('{0} {1} {2} {3} \n'.format(str(step), str(i), str(history.history["loss"][i]),
                                                       str(history.history["val_dice_coef"][i])))


def create_paths():
    """
    Creates all the output paths.
    """
    path_ranks = global_path + "ranks/"
    path_logs = global_path + "logs/"
    path_plots = global_path + "plots/"
    path_models = global_path + "models/"

    if not os.path.exists(path_ranks):
        os.makedirs(path_ranks)
        print("Path created: ", path_ranks)

    if not os.path.exists(path_logs):
        os.makedirs(path_logs)
        print("Path created: ", path_logs)

    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
        print("Path created: ", path_plots)

    if not os.path.exists(path_models):
        os.makedirs(path_models)
        print("Path created: ", path_models)
