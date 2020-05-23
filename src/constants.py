# PATH definition
global_path = "./"
initial_weights_path = "models/[initial_weights_name].hdf5"
final_weights_path = "models/[output_weights_name].hdf5"

data_path = '/media/airscan/Backup/izza/Dataset/coco/val2017'
masks_path= '/media/airscan/Backup/izza/Dataset/coco/img_annotations/val2017'

# Data definition
img_rows = 224#64 * 3
img_cols = 224#80 * 3

# nb_total = 3053#2000 (118287)
# nb_train = 2800#1600 (118287)
# nb_labeled = 2000#600 (39429)
# nb_unlabeled = nb_train - nb_labeled

#with planes
nb_total = 3150#2000 (118287)
nb_train = 3150#1600 (118287)
nb_labeled = 2000#600 (39429)
nb_unlabeled = nb_train - nb_labeled

# CEAL parameters
apply_edt = True
# nb_iterations = 10#50
nb_iterations = 20#50
# nb_iterations = 1#50


nb_step_predictions = 20

nb_no_detections = 10
nb_random = 15
nb_most_uncertain = 10
most_uncertain_rate = 5

pseudo_epoch = 5
nb_pseudo_initial = 20
pseudo_rate = 20

initial_train = False
# initial_train = False

apply_augmentation = False
# nb_initial_epochs = 20#50
nb_initial_epochs = 150#50
# nb_active_epochs = 2#10
nb_active_epochs = 6#10
batch_size = 32
