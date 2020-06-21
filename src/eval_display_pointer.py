from __future__ import print_function

from keras.callbacks import ModelCheckpoint

from data_split import load_train_data_withPlanes, get_data_mean_multi, load_train_data_multiclass, get_colored_segmentation_image
from utils import *

create_paths()
log_file = open(global_path + "logs/log_file.txt", 'a')

# CEAL data definition
X_train, y_train = load_train_data_multiclass(nonplanes = False)
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, len(X_train))

# (1) Initialize model
model = get_unet_multi(dropout=False,channels=3,n_class=14)

mean_data, std_data = get_data_mean_multi()
model.load_weights(initial_weights_path)
# model.load_weights(global_path + "models/active_model10.h5")
print('x shape', X_train.shape)
print('y shape', y_train.shape)

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
# ids = [0,1,2,3,4,6,7,8,9,10]
# ids = list(range(nb_labeled, nb_labeled+30))
# ids = list(range(3100, 3130))
# ids = list(range(2026, 2027))
# ids = list(range(20, 40))
ids = list(range(0, 50)) + list(range(1000, 1100))
# ids = list(range(2500, 2530))
# ids = list(range(50))
show = False
invert_color = True
for id in ids:

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
    _ , img_var, img_dist, img_post, img_raw, contour_list = compute_uncertain_multi_img(sample, modelUncertain, y_train[id][:,:,0])
    print(img_dist.shape)
    # print('max dist', np.max(img_dist))
    # print('min dist', np.min(img_dist))
    print('max var', np.max(img_var))
    print('min var', np.min(img_var))
    print('max post', np.max(img_post))
    print('min post', np.min(img_post))

    (minVal, maxVal, minLoc, maxLoc_var) = cv2.minMaxLoc(img_var)
    # cv2.circle(img_var, maxLoc_var, 8, (255, 0, 0), 2)
    print('max loc normalized for variance', maxLoc_var)
    print('variance at var',img_var[maxLoc_var[1],maxLoc_var[0]])
    print('variance at norm',img_post[maxLoc_var[1],maxLoc_var[0]])
    print('variance at raw',img_raw[maxLoc_var[1],maxLoc_var[0]])

    (minVal, maxVal, minLoc, maxLoc_dist) = cv2.minMaxLoc(img_dist)
    cv2.circle(img_dist, maxLoc_dist, 8, (255, 0, 0), 2)
    print('max loc edt variance', maxLoc_dist)

    (minVal, maxVal, minLoc, maxLoc_post) = cv2.minMaxLoc(img_post)
    cv2.circle(img_post, maxLoc_post, 8, (255, 0, 0), 2)
    print('max loc normalized for post processed', maxLoc_post)
    print('variance at var',img_var[maxLoc_post[1],maxLoc_post[0]])
    print('variance at norm',img_post[maxLoc_post[1],maxLoc_post[0]])
    print('variance at raw',img_raw[maxLoc_post[1],maxLoc_post[0]])
    color_with_max = np.copy(x_show)

    cv2.circle(color_with_max, maxLoc_post, 8, (255, 0, 0), 2)
    # cv2.circle(color_with_max, maxLoc_var, 8, (0, 255, 0), 2)
    cv2.circle(color_with_max, maxLoc_dist, 8, (0, 0, 255), 2)

    cv2.drawContours(color_with_max, contour_list,  -1, (255,0,255), 2)
    #invert_color
    if invert_color:
        img_var = (255-img_var)
        img_dist = (255-img_dist)
        img_post = (255-img_post)

    if show:
        cv2.imshow('input', x_show)
        cv2.waitKey()
        cv2.imshow('gt', gt_show_colored)
        cv2.waitKey()
        cv2.imshow('pred', pred_show_colored)
        cv2.waitKey()
        cv2.imshow('VAR', img_var.astype(np.uint8))
        cv2.waitKey()
        cv2.imshow('Dist', img_dist.astype(np.uint8))
        cv2.waitKey()
        cv2.imshow('post', img_post.astype(np.uint8))
        cv2.waitKey()
        cv2.imshow('colors with circle', color_with_max)
        cv2.waitKey()
    # print(img_dist.shape)
    # im_h = cv2.hconcat([x_show, pred_show_colored, gt_show_colored, img_var, img_dist.astype(np.uint8), img_post,color_with_max])
    im_h = cv2.hconcat([x_show, pred_show_colored, gt_show_colored, cv2.cvtColor(img_var.astype(np.uint8), cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(img_post.astype(np.uint8), cv2.COLOR_GRAY2BGR), cv2.cvtColor(img_dist.astype(np.uint8), cv2.COLOR_GRAY2BGR), color_with_max])


    cv2.imwrite('outputs/image_{}.png'.format(id), x_show)
    cv2.imwrite('outputs/pred_{}.png'.format(id), pred_show_colored)
    cv2.imwrite('outputs/gt_{}.png'.format(id), gt_show_colored)
    cv2.imwrite('outputs/var_{}.png'.format(id), img_var)
    cv2.imwrite('outputs/dist_{}.png'.format(id), img_dist.astype(np.uint8))
    cv2.imwrite('outputs/var_masked_{}.png'.format(id), img_post)
    cv2.imwrite('outputs/color_with_max_{}.png'.format(id), color_with_max)
    cv2.imwrite('outputs/concat_{}.png'.format(id), im_h)

    bg_img = y_train[id][:,:,0].astype(np.uint8)
    # print('max bg', np.max(bg_img))
    bg_img = (bg_img * 255).astype(np.uint8)
    print(bg_img.shape)
    # print('max bg after', bg_img)
    cv2.imwrite('out_pointer/bg_{}.png'.format(id), bg_img)
    cv2.imwrite('out_pointer/image_{}.png'.format(id), x_show)
    cv2.imwrite('out_pointer/gt_{}.png'.format(id), gt_show_colored)
    cv2.imwrite('out_pointer/pred_{}.png'.format(id), pred_show_colored)
    cv2.imwrite('out_pointer/color_with_max_{}.png'.format(id), color_with_max)
    cv2.imwrite('out_pointer/var_post_{}.png'.format(id), img_post)
    cv2.imwrite('out_pointer/var_dist_{}.png'.format(id), img_dist)


    print('max loc normalized for variance', maxLoc_var)
    print('max loc edt variance', maxLoc_dist)
    print('max loc normalized for post processed', maxLoc_post)
    np.savetxt('out_pointer/pointer_dist_{}.txt'.format(id),maxLoc_dist)

    np.savetxt('out_pointer/pointer_post_{}.txt'.format(id),maxLoc_post)
    # cv2.imshow('input', x_show[0])
    # cv2.waitKey()
    # cv2.imshow('gt', gt_show[0])
    # cv2.waitKey()
    # cv2.imshow('pred', pred_show[0])
    # cv2.waitKey()
