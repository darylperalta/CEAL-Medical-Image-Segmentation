from __future__ import print_function

from keras.callbacks import ModelCheckpoint

from data import load_train_data, get_data_mean
from utils import *

create_paths()
log_file = open(global_path + "logs/log_file.txt", 'a')

# CEAL data definition
X_train, y_train = load_train_data()
labeled_index = np.arange(0, nb_labeled)
unlabeled_index = np.arange(nb_labeled, len(X_train))

# (1) Initialize model
model = get_unet(dropout=True)
mean_data, std_data = get_data_mean()
# model.load_weights(initial_weights_path)
model.load_weights(global_path + "models/active_model10.h5")
print('input shape', X_train.shape)


test_num = 4

out = model.predict(X_train[nb_labeled:nb_labeled+test_num])

for i in range(test_num):
    print(np.max(X_train))
    print(X_train.dtype)

    x_show = ((X_train[nb_labeled+i]*std_data) + mean_data).astype(np.uint8)
    print(x_show.shape)
    gt_show = ((y_train[nb_labeled+i]*255)).astype(np.uint8)
    print(x_show.shape)
    pred_show = ((out[i]*255)).astype(np.uint8)


    cv2.imshow('input', x_show[0])
    cv2.waitKey()
    cv2.imshow('gt', gt_show[0])
    cv2.waitKey()
    cv2.imshow('pred', pred_show[0])
    cv2.waitKey()
# if initial_train:
#     model_checkpoint = ModelCheckpoint(initial_weights_path, monitor='loss', save_best_only=True)
#
#     if apply_augmentation:
#         for initial_epoch in range(0, nb_initial_epochs):
#             history = model.fit_generator(
#                 data_generator().flow(X_train[labeled_index], y_train[labeled_index], batch_size=32, shuffle=True),
#                 steps_per_epoch=len(labeled_index), nb_epoch=1, verbose=1, callbacks=[model_checkpoint])
#
#             model.save(initial_weights_path)
#             log(history, initial_epoch, log_file)
#     else:
#         history = model.fit(X_train[labeled_index], y_train[labeled_index], batch_size=32, nb_epoch=nb_initial_epochs,
#                             verbose=1, shuffle=True, callbacks=[model_checkpoint])
#
#         log(history, 0, log_file)
# else:
#     model.load_weights(initial_weights_path)
#
# # Active loop
# model_checkpoint = ModelCheckpoint(final_weights_path, monitor='loss', save_best_only=True)
#
# for iteration in range(1, nb_iterations + 1):
#     if iteration == 1:
#         weights = initial_weights_path
#
#     else:
#         weights = final_weights_path
#
#     # (2) Labeling
#     X_labeled_train, y_labeled_train, labeled_index, unlabeled_index = compute_train_sets(X_train, y_train,
#                                                                                           labeled_index,
#                                                                                           unlabeled_index, weights,
#                                                                                           iteration)
#     # (3) Training
#     history = model.fit(X_labeled_train, y_labeled_train, batch_size=32, nb_epoch=nb_active_epochs, verbose=1,
#                         shuffle=True, callbacks=[model_checkpoint])
#
#     log(history, iteration, log_file)
#     model.save(global_path + "models/active_model" + str(iteration) + ".h5")
#
# log_file.close()
