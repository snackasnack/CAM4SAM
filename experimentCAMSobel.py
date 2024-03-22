'''
Using out_crf and sobel filter npy values
Find out how to derive the best set of points for Segment Anything Model (SAM)
'''

# Standard library imports
from collections import Counter
from io import BytesIO
import itertools
import os
import sys
sys.path.append("..")
import warnings

warnings.filterwarnings("ignore")

# Third-party library imports
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import cv2
from scipy import ndimage
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from torchvision.ops import box_convert
import torch

# Local application/library specific imports
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor
from peakActivation import texture_analysis, predictThresholds


def get_file_data(file_name):

    test_image = os.path.join('C:/Users/snack/Desktop/SEAM/VOC2012_data/JPEGImages/', file_name + '.jpg')
    sample_zero_image = cv2.imread(test_image)

    #image_crf_npy = os.path.join('C:/Users/snack/Desktop/SEAM/voc12/out_crf_4.0', file_name + '.npy')
    #sample_npy_data = np.load(image_crf_npy, allow_pickle=True).item()

    image_cam_npy = os.path.join('C:/Users/snack/Desktop/SEAM/voc12/out_cam/'+ file_name + '.npy')
    sample_npy_data = np.load(image_cam_npy, allow_pickle=True).item()
    
    return sample_zero_image, sample_npy_data


def sobel_filter(image_npy):
    dx = ndimage.sobel(image_npy[0], axis=0)
    dy = ndimage.sobel(image_npy[0], axis=1)
    sobel_filtered_image = np.hypot(dx, dy)  # Equivalent to sqrt(dx^2 + dy^2)

    sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)
    return sobel_filtered_image


def remove_outliers(points_array):
    data_array = np.array(points_array)
    mean = np.mean(data_array, axis=0)
    std_dev = np.std(data_array, axis=0)
    
    std_dev = np.where(std_dev == 0, 1e-6, std_dev)
    std_dev = np.where(np.isnan(std_dev), 1e-6, std_dev)
    
    threshold = 1.8
    z_scores = np.abs((data_array - mean) / std_dev)
    filtered_data = data_array[(z_scores < threshold).all(axis=1)]
    
    return filtered_data


def baselinePointSelection(sample_npy_data):
    points_of_interest = np.where((sample_npy_data[0] <= 0.2))
    filtered_points_of_interest = remove_outliers(points_of_interest)   
    centroid_y = np.mean(filtered_points_of_interest[0])
    centroid_x = np.mean(filtered_points_of_interest[1])
    centroid = (centroid_x, centroid_y)
    return centroid


def pointSelection(cam_threshold, sobel_threshold, sample_npy_data, sobel_filtered_image):

    boundary_points = []

    points_of_interest = np.where((sample_npy_data[0] >= cam_threshold))
    filtered_points_of_interest = remove_outliers(points_of_interest)   

    for point in zip(*filtered_points_of_interest):
        x, y = point

        sobel_value = sobel_filtered_image[x, y]
        
        if (sobel_value > sobel_threshold):
            
            boundary_points.append((y, x))

    boundary_points = remove_outliers(boundary_points)
    boundary_points_array = np.array(boundary_points)

    # Apply DBSCAN clustering
    eps = 10  # Adjust as needed, represents the maximum distance between two samples for them to be considered as in the same neighborhood
    min_samples = 5  # Adjust as needed, represents the number of samples in a neighborhood for a point to be considered as a core point
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(boundary_points_array)

    # Select one point from each cluster
    unique_labels = np.unique(cluster_labels)
    selected_points = []
    for label in unique_labels:
        cluster_points = boundary_points_array[cluster_labels == label]
        if len(cluster_points) > 0:
            # Select the point closest to the centroid of the cluster
            centroid = np.mean(cluster_points, axis=0)
            closest_point_idx = np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))
            selected_points.append(cluster_points[closest_point_idx])

    # Convert the selected points to array
    selected_points_array = np.array(selected_points)

    # Save the selected points to a new file
    output_file_path = os.path.join('C:/Users/snack/Desktop/CAM4SAM/temp_files/points_of_interest.txt')
    np.savetxt(output_file_path, selected_points_array, fmt='%d', delimiter=',')

    return output_file_path


# Grounding DINO

def groundingdino(file_name, TEXT_PROMPT = 'plane', BOX_TRESHOLD = 0.35, TEXT_TRESHOLD = 0.25):
    model = load_model("C:/Users/snack/Desktop/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "C:/Users/snack/Desktop/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    IMAGE_PATH = os.path.join('C:/Users/snack/Desktop/SEAM/VOC2012_data/JPEGImages/', file_name + '.jpg')
    TEXT_PROMPT = "plane"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    boxes_array = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    height, width, _ = image_source.shape

    # Convert bounding box coordinates from normalized values to pixel values
    bounding_boxes_pixel = np.zeros_like(boxes_array)
    for i, box in enumerate(boxes_array):
        x_min, y_min, x_max, y_max = box
        bounding_boxes_pixel[i, 0] = x_min * width 
        bounding_boxes_pixel[i, 1] = y_min * height  
        bounding_boxes_pixel[i, 2] = x_max * width 
        bounding_boxes_pixel[i, 3] = y_max * height 

    output_file_path = os.path.join("C:/Users/snack/Desktop/CAM4SAM/temp_files/dino_boxes/"+ file_name + "_boxes.npy")
    np.save(output_file_path, bounding_boxes_pixel)

    return output_file_path


# SAM 

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


def set_up_SAM():
    sam_checkpoint = "C:/Users/snack/Desktop/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def sam_process_image(sam_predictor, test_image):
    image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image)


def points_in_box(box, points):
    x_min, y_min, x_max, y_max = box
    in_box = []

    for point in points:
        if isinstance(point, (int, float)):
            x = point
            y = None 
        else:
            x, y = point
        if (x is not None) and (y is not None) and (x_min <= x <= x_max) and (y_min <= y <= y_max):
            in_box.append(point)
    return np.array(in_box)


def get_input(pointAnnoFile, boxAnnoFile):
    input_points = np.loadtxt(pointAnnoFile, delimiter=',')
    if input_points.ndim == 0:
        print("There are no point annotations")

    input_boxes = np.load(boxAnnoFile)
    no_instance_detected = input_boxes.ndim

    boxes_to_points = {}

    for idx, box in enumerate(input_boxes):
        box_key = idx
        in_box_points = points_in_box(box, input_points)        
        if len(in_box_points) > 0:
            boxes_to_points[box_key] = np.array(in_box_points)

    return boxes_to_points


def run_sam(sam_predictor, input_points=None, input_labels=None, input_boxes=None):
    if input_boxes is None:
        masks, scores , logits = sam_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True, 
        )
    elif input_points is None:
        masks, scores , logits = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box = input_boxes,
        multimask_output=True, 
        )
    else: 
        masks, scores , logits = sam_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box = input_boxes,
        multimask_output=True, 
        )
        

    return masks, scores , logits


def getIou(sam_masks, file_name):

    gt_mask = cv2.imread(os.path.join('C:/Users/snack/Desktop/SEAM/VOC2012_data/SegmentationClass/', file_name + '.png'), cv2.IMREAD_GRAYSCALE)
    predicted_mask = sam_masks

    # Convert masks to boolean arrays
    gt_mask_bool = gt_mask > 0
    predicted_mask_bool = predicted_mask > 0

    # Calculate intersection and union
    intersection = np.logical_and(gt_mask_bool, predicted_mask_bool)
    union = np.logical_or(gt_mask_bool, predicted_mask_bool)

    # Compute IoU
    iou = np.sum(intersection) / np.sum(union)
    #print(iou*100)
    
    return iou*100


def experiment(file_name, thresholds, sample_image, sample_image_npy, sobel_filtered_image):

    mask_results = []
    #threshold_combi = []
    

    for cam_threshold, sobel_threshold in itertools.product(thresholds['cam'], thresholds['sobel']):
        #print(cam_threshold, sobel_threshold)
        #print(mask_results)
        try:
            pointAnno = pointSelection(cam_threshold, sobel_threshold, sample_image_npy, sobel_filtered_image)
            boxAnno = groundingdino(file_name)
            boxes_to_points = get_input(pointAnno, boxAnno)
            input_boxes = np.load(boxAnno)
            sam_predictor = set_up_SAM()
            sam_process_image(sam_predictor, sample_image)
            best_masks = []
            for input_boxes_idx, input_points in boxes_to_points.items():
                input_labels = np.ones(len(input_points), dtype=int)
                for i, point in enumerate(input_points):
                    input_labels[i] = 1
                input_box = input_boxes[input_boxes_idx]
                masks, scores, logits = run_sam(sam_predictor, input_points, input_labels, input_box)
                best_mask_index = np.argmax(scores)
                best_masks.append(masks[best_mask_index])

            # Combine the best masks
            combined_mask = np.zeros_like(best_masks[0], dtype=bool)
            for mask in best_masks:
                combined_mask |= mask
            iou_result = getIou(combined_mask, file_name)
            '''
            try:
                plt.figure(figsize=(10, 10))
                plt.imshow(sample_image)
                show_mask(combined_mask, plt.gca())
                show_points(input_points, input_labels, plt.gca())
                for box in input_boxes:
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                plt.axis('off')
                saving_file_name = file_name + '_' +  str(iou_result) + '_' + str(cam_threshold) + '_' + str(sobel_threshold) + '_' + '_mask.png'
                output_file = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/best_masks/' + saving_file_name 
                plt.savefig(output_file, dpi=300)
            except Exception as e:
                print(f"Error occurred while saving best masked image")
            '''

            #mask_results.append(iou_result)
            #threshold_combi.append((cam_threshold, sobel_threshold))
            mask_results.append([iou_result, cam_threshold, sobel_threshold])
        except Exception as e:
            print(f"Error occurred for thresholds ({cam_threshold}, {sobel_threshold}): {e}")
            #mask_results.append(float(0.0))
            #threshold_combi.append((cam_threshold, sobel_threshold))
            mask_results.append([float(0.0), cam_threshold, sobel_threshold])
            continue
    
    output_file = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/mask_results/' + file_name + '_mask_results.txt'
    with open(output_file, 'w') as f:
        for result in mask_results:
            f.write(','.join(map(str, result)) + '\n')

    #sorted_indices = sorted(range(len(mask_results)), key=lambda i: mask_results[i], reverse=True)
    #sorted_mask_results = [mask_results[i] for i in sorted_indices]
    #sorted_threshold_combi = [threshold_combi[i] for i in sorted_indices]

    '''
    try:
        cam_thresholds, sobel_thresholds = zip(*threshold_combi)
        iou_results = mask_results
        bins = [np.linspace(0, 1, 21), np.linspace(0, 1, 21)]
        hist, xedges, yedges = np.histogram2d(cam_thresholds, sobel_thresholds, bins=bins, weights=iou_results)
        plt.figure(figsize=(10, 8))
        plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='viridis')
        plt.colorbar(label='IOU Result')
        plt.xlabel('CAM Threshold')
        plt.ylabel('Sobel Threshold')
        plt.title('IOU Result vs. Threshold Combinations')
        output_file = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/heatmap/' + file_name + '_heatmap.png'
        plt.savefig(output_file, dpi=300)
    except Exception as e:
        print(f"Error occurred while saving heatmap")
    '''
    #return sorted_mask_results, sorted_threshold_combi
    return None


def refinedSAM(cam_threshold, sobel_threshold, sample_image, sample_image_npy, sobel_filtered_image):
    iou_result_lst = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/iou_results.txt'

    try:
        pointAnno = pointSelection(cam_threshold, sobel_threshold, sample_image_npy, sobel_filtered_image)
        boxAnno = groundingdino(file_name)
        sam_predictor = set_up_SAM()
        sam_process_image(sam_predictor, sample_image)
        boxes_to_points = get_input(pointAnno, boxAnno)
        input_boxes = np.load(boxAnno)
        best_masks = []
        for input_boxes_idx, input_points in boxes_to_points.items():
            num_points = len(input_points)
            input_labels = np.ones(num_points, dtype=int)
            for i, point in enumerate(input_points):
                input_labels[i] = 1
            input_box = input_boxes[input_boxes_idx]
            
            if len(input_points) == 0:
                masks, scores, logits = run_sam(sam_predictor, input_boxes = input_box)
            else:
                masks, scores, logits = run_sam(sam_predictor, input_points, input_labels, input_box)
            best_mask_index = np.argmax(scores)
            best_masks.append(masks[best_mask_index])
        if best_masks:
            combined_mask = np.zeros_like(best_masks[0], dtype=bool)
            for mask in best_masks:
                combined_mask |= mask
        else:
            print("No best masks found.")
            combined_mask = np.zeros((sample_image.shape[0],sample_image.shape[1]), dtype=bool)
        iou_result = getIou(combined_mask, file_name)
        with open(iou_result_lst, 'a') as f:
            f.write(str(iou_result) + '\n')

        try:
            plt.figure(figsize=(10, 10))
            plt.imshow(sample_image)
            show_mask(combined_mask, plt.gca())
            for box in input_boxes:
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
            show_points(input_points, input_labels, plt.gca())
            plt.axis('off')
            saving_file_name = file_name + '_' +  str(iou_result) + '_' + str(cam_threshold) + '_' + str(sobel_threshold) + '_' + '_mask.png'
            output_file = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/best_masks/' + saving_file_name 
            plt.savefig(output_file, dpi=300)
        except Exception as e:
            print(f"Error occurred while saving best masked image")
            baselineModel(sample_image, sample_image_npy, iou_result_lst = iou_result_lst)
    except Exception as e:
        print(f"Error occurred for thresholds, falling back to baseline model: {e}")
        baselineModel(sample_image, sample_image_npy, iou_result_lst = iou_result_lst)

    return None


def baselineModel(sample_image, sample_image_npy, iou_result_lst = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/baseline_iou_results.txt'):
    max_retries = 0  # Maximum number of retries

    try:
        #pointAnno = baselinePointSelection(sample_image_npy)
        boxAnno = groundingdino(file_name)
        input_boxes = np.load(boxAnno)[0]
        sam_predictor = set_up_SAM()
        sam_process_image(sam_predictor, sample_image)
        #input_point = np.array([pointAnno])
        #input_label = [1]
            
        masks, scores, logits = run_sam(sam_predictor, input_boxes = input_boxes)
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]
        best_mask = np.array(best_mask)
    
        iou_result = getIou(best_mask, file_name)
        with open(iou_result_lst, 'a') as f:
            f.write(str(iou_result) + '\n')

        try:
            plt.figure(figsize=(10, 10))
            plt.imshow(sample_image)
            show_mask(best_mask, plt.gca())
            #plt.scatter(input_point[:, 0], input_point[:, 1], color='red', marker='*', s=400)
            x_min, y_min, x_max, y_max = input_boxes
            width = x_max - x_min
            height = y_max - y_min
            rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.axis('off')
            saving_file_name = file_name + '_' +  str(iou_result) + '_' + '_mask.png'
            output_file = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/best_masks/' + saving_file_name 
            plt.savefig(output_file, dpi=300)
        except Exception as e:
            print(f"Error occurred while saving best masked image")
    except Exception as e:
        print(f"Error occurred: {e}")

    return None


if __name__ == '__main__':
    
    # predict CAM and sobel thresholds in batch
    files = []
    with open("C:/Users/snack/Desktop/CAM4SAM/temp_files/classZeroSingleInstanceImages.txt", "r") as file:
        for line in tqdm(file):
            line = line.strip()
            file_name = line.replace(".npy", "")
            files.append(file_name)

    scaled_texture_df, _ = texture_analysis(files = files)
    selected_data = scaled_texture_df[['contrast', 'correlation', 'energy', 'homogeneity']]
    optimalThresholds = predictThresholds(selected_data)
    #print(optimalThresholds)

    # running refined SAM to get results
    output_directory = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/'

    threshold_idx = 0
    with open("C:/Users/snack/Desktop/CAM4SAM/temp_files/classZeroSingleInstanceImages.txt", "r") as file:
        for line in tqdm(file):
            line = line.strip()
            file_name = line.replace(".npy", "")

            print(file_name)
            sys.stdout.flush() 

            cam_threshold, sobel_threshold = optimalThresholds[threshold_idx][0], optimalThresholds[threshold_idx][1]
            sample_image, sample_image_npy = get_file_data(file_name)
            sobel_filtered_image = sobel_filter(sample_image_npy)
            refinedSAM(cam_threshold, sobel_threshold, sample_image, sample_image_npy, sobel_filtered_image)

            threshold_idx += 1
    
    output_directory = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/'
    '''
    # testing threshold ranges for a set of images
    thresholds = {
            'cam': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
            'sobel': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
            }

    with open("C:/Users/snack/Desktop/CAM4SAM/temp_files/classZeroSingleInstanceImages.txt", "r") as file:
        for line in file:
            line = line.strip()
            file_name = line.replace(".npy", "")

            print(file_name)
            sys.stdout.flush() 

            sample_image, sample_image_npy = get_file_data(file_name)
            sobel_filtered_image = sobel_filter(sample_image_npy)
            
            experiment(file_name, thresholds, sample_image, sample_image_npy, sobel_filtered_image)
            #output_file = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/mask_results/' + file_name + '_mask_results.txt'

            #output_file = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/mask_results/' + file_name + '_mask_results.txt'
            #with open(output_file, "w") as f:
            #    for i, (iou, threshold) in enumerate(zip(mask_results, threshold_combi)):
            #        line = f"Top {i+1}: IOU = {iou}, threshold = {threshold}\n"
            #        f.write(line)
    
    # testing on single image
    file_name="2009_005120"
    sample_image, sample_image_npy = get_file_data(file_name)
    sobel_filtered_image = sobel_filter(sample_image_npy)
    pointAnno = pointSelection(0.2, 0.6, sample_image_npy, sobel_filtered_image)
    #pointAnno="C:/Users/snack/Desktop/CAM4SAM/temp_files/points_of_interest.txt"
    boxAnno = groundingdino(file_name)
    sam_predictor = set_up_SAM()
    sam_process_image(sam_predictor, sample_image)
    boxes_to_points = get_input(pointAnno, boxAnno)
    
    input_boxes = np.load(boxAnno)

    best_masks = []

    for input_boxes_idx, input_points in boxes_to_points.items():
        num_points = len(input_points)
        input_labels = np.ones(num_points, dtype=int)
        for i, point in enumerate(input_points):
            input_labels[i] = 1
        input_box = input_boxes[input_boxes_idx]
        
        masks, scores, logits = run_sam(sam_predictor, input_points, input_labels, input_box)
        best_mask_index = np.argmax(scores)
        best_masks.append(masks[best_mask_index])

    # Combine the best masks
    if best_masks:
        combined_mask = np.zeros_like(best_masks[0], dtype=bool)
        for mask in best_masks:
            combined_mask |= mask
    else:
        print("No best masks found.")
        combined_mask = np.zeros((sample_image.shape[0],sample_image.shape[1]), dtype=bool)

    iou_result = getIou(combined_mask, file_name)

    # Visualization code for the combined mask
    plt.figure(figsize=(10,10))
    plt.imshow(sample_image)
    show_mask(combined_mask, plt.gca())
    for box in input_boxes:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    #show_points(input_points, input_labels, plt.gca())
    plt.title("Combined Best Masks", fontsize=18)
    plt.axis('off')
    plt.show()

    # testing baseline model
    output_directory = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/'

    with open("C:/Users/snack/Desktop/CAM4SAM/temp_files/classZeroSingleInstanceImages.txt", "r") as file:
        for line in tqdm(file):
            line = line.strip()
            file_name = line.replace(".npy", "")

            print(file_name)
            sys.stdout.flush() 

            sample_image, sample_image_npy = get_file_data(file_name)
            baselineModel(sample_image, sample_image_npy)
    '''