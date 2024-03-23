import numpy as np

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

file = "C:/Users/snack/Desktop/SEAM/voc12/cls_labels.npy"

file_data = np.load(file, allow_pickle=True).item()

def get_class_labels_and_indexes_for_image(image_id):
    if image_id in file_data:
        class_array = file_data[image_id]
        class_indices = np.nonzero(class_array)[0]
        class_labels = [CAT_LIST[i] for i in class_indices]
        joined_class_labels = ' and '.join(class_labels)
        return joined_class_labels, class_indices
    else:
        return None, None
