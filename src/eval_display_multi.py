from __future__ import print_function

from keras.callbacks import ModelCheckpoint

from data_split import load_train_data_withPlanes, get_data_mean_multi, load_train_data_multiclass, get_colored_segmentation_image
from utils import *

create_paths()
log_file = open(global_path + "logs/log_file.txt", 'a')

# CEAL data definition
X_train, y_train = load_train_data_multiclass()
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, len(X_train))

# (1) Initialize model
model = get_unet_multi(dropout=False,channels=3,n_class=14)

mean_data, std_data = get_data_mean_multi()
model.load_weights(initial_weights_path)
# model.load_weights(global_path + "models/active_model10.h5")
print('input shape', X_train.shape)

modelUncertain = get_unet_multi(dropout=True,channels=3,n_class=14)
modelUncertain.load_weights(initial_weights_path)

test_num = 10

out = model.predict(X_train[nb_labeled:nb_labeled+test_num])
data_path = '/home/daryl/unsupervised/datasets/coco/plane_split/non_plane/'
masks_path = '/home/daryl/unsupervised/datasets/coco/plane_split/non_plane_mask/'
images = sorted(os.listdir(data_path))
masks = sorted(os.listdir(masks_path))

import numpy as np
#print(np.unique(out))
#print(np.unique(y_train))
# ids = [2770, 2832 ,2610, 2851]
# ids = [2770, 2832, 3129, 2610 ]
# ids = [2622, 2278, 3017] # uncertain
# ids = [2770 2832 3129 2610 3075 2851 3135 2836 2676 2633] # complete no detected
# ids = [2770, 2832, 3129, 2610, 3075] # complete no detected
# ids = [2001,2003,2200, 2703,2832, 2900,2905,3130,3149,3000] # test
# ids = [3000,3001,3002,3120,3121,3122,3123,3125,3130,3149]
ids = [0,1,2,3,4,6,7,8,9,10]
for id in range(15):


    # print('filename', images[id])
    print(np.max(X_train))
    print(X_train.dtype)
    out = model.predict(X_train[id:id+1])

    x_show = ((X_train[id]*std_data) + mean_data).astype(np.uint8)
    print(x_show.shape)
    print(np.max(x_show))
    print('mx0', np.max(y_train[id]))
    print('ytrain', y_train.shape)
    gt_show = (((y_train[id])*255)).astype(np.uint8).argmax(axis=2)
    print(gt_show.shape)
    print('mx gt', np.max(gt_show))
    print('min gt', np.min(gt_show))

    gt_show_colored = get_colored_segmentation_image(gt_show, 14).astype(np.uint8)
    print(gt_show_colored.shape)
    print('mx', np.max(gt_show_colored))
    print('nin', np.min(gt_show_colored))
    pred_show = ((out[0]*255)).astype(np.uint8).argmax(axis=2)
    pred_show_colored = get_colored_segmentation_image(pred_show, 14).astype(np.uint8)
    # pred_show_colored = get_colored_segmentation_image(pred_show, 14)
    print('pred shape',pred_show.shape)
    # print(np.unique(pred_show))

    sample = X_train[id].reshape([1, img_rows, img_cols, 3])
    _ , img_var = compute_uncertain_multi_img(sample, modelUncertain)

    cv2.imshow('input', x_show)
    cv2.waitKey()
    cv2.imshow('gt', gt_show_colored)
    cv2.waitKey()
    cv2.imshow('pred', pred_show_colored)
    cv2.waitKey()
    cv2.imshow('VAR', img_var.astype(np.uint8))
    cv2.waitKey()

    cv2.imwrite('outputs/image_{}.png'.format(id), x_show)
    cv2.imwrite('outputs/pred_{}.png'.format(id), pred_show_colored)
    cv2.imwrite('outputs/gt_{}.png'.format(id), gt_show_colored)
    cv2.imwrite('outputs/var_{}.png'.format(id), img_var)



    # cv2.imshow('input', x_show[0])
    # cv2.waitKey()
    # cv2.imshow('gt', gt_show[0])
    # cv2.waitKey()
    # cv2.imshow('pred', pred_show[0])
    # cv2.waitKey()
