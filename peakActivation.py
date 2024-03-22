import os 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2


def read_results(directory_path = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/Saved_Results/mask_results'):
    data = []
    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3: 
                    iou_result, cam_threshold, sobel_threshold = parts
                    data.append((str(file_name[:11]), float(iou_result), float(cam_threshold), float(sobel_threshold)))
    return pd.DataFrame(data, columns=['file_name','IoU_Result', 'CAM_Threshold', 'Sobel_Threshold'])


def texture_analysis(directory_path = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/Saved_Results/mask_results'):
    texture_data = []

    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    image_folder = 'C:/Users/snack/Desktop/SEAM/VOC2012_data/JPEGImages/'
    for file_name in files:
        image_path = os.path.join(image_folder + file_name[:11]+ '.jpg')
        loaded_image = cv2.imread(image_path)

        if len(loaded_image.shape) > 2 and loaded_image.shape[2] > 1:
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)

        glcm = graycomatrix(loaded_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        correlation = graycoprops(glcm, 'correlation')
        energy = graycoprops(glcm, 'energy')
        homogeneity = graycoprops(glcm, 'homogeneity')
        texture_data.append((file_name[:11], float(contrast), float(correlation), float(energy), float(homogeneity)))

    return pd.DataFrame(texture_data, columns=['file_name','contrast', 'correlation', 'energy', 'homogeneity'])
    
if __name__ == "__main__":

    results_df = read_results()
    print(results_df)
    #results_df['combi'] = results_df['CAM_Threshold'].astype(str) + "_" + results_df['Sobel_Threshold'].astype(str)
    
    texture_df = texture_analysis()
    print(texture_df)

    '''
    all_combi = results_df['combi'].unique()
    filtered_df = results_df[results_df['IoU_Result'] == 0]
    combi_with_zero = filtered_df['combi'].unique()
    difference = np.setdiff1d(combi_with_zero, all_combi)
    print("Unique 'combi' values in filtered_df but not in results_df:")
    print(difference)
    '''
