import os 
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model


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
    #image_folder = '/Users/teyc/Desktop/VOCdevkit 2/VOC2012/JPEGImages/'
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

    texture_df = pd.DataFrame(texture_data, columns=['file_name', 'contrast', 'correlation', 'energy', 'homogeneity'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(texture_df[['contrast', 'correlation', 'energy', 'homogeneity']])
    scaled_df = pd.DataFrame(X_scaled, columns=['contrast', 'correlation', 'energy', 'homogeneity'])
    texture_df[['contrast', 'correlation', 'energy', 'homogeneity']] = scaled_df

    return texture_df, scaler


def mergeDf(experiment_data, texture_data):
    return pd.merge(experiment_data, texture_data, on='file_name')


def getOptimisedData(experiment_df, texture_df):
    mergedData = mergeDf(experiment_df, texture_df)
    top_2_iou_results = mergedData.groupby('file_name')['IoU_Result'].nlargest(2)
    top_2_iou_results = top_2_iou_results.reset_index()
    top_2_rows = mergedData.loc[top_2_iou_results.level_1]
    
    return top_2_rows


def multi_layer_perceptron(optimised_data):
    X = optimised_data[['contrast', 'correlation', 'energy', 'homogeneity']]
    y = optimised_data[['CAM_Threshold', 'Sobel_Threshold']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
    input_size = X_train.shape[1]
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(input_size,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=1)
    print('\n', 'Test MSE:', score)

    model.save("texture_model.keras")


def predictThresholds(X_inputs):
    saved_model = load_model("texture_model.keras")
    
    return saved_model.predict(X_inputs)


if __name__ == "__main__":

    experiment_df = read_results()
    #print(experiment_df)
    #results_df['combi'] = results_df['CAM_Threshold'].astype(str) + "_" + results_df['Sobel_Threshold'].astype(str)
    
    texture_df, scaler= texture_analysis()
    #print(texture_df)

    optimised_data = getOptimisedData(experiment_df, texture_df)
    print(optimised_data)
    
    multi_layer_perceptron(optimised_data)

    #selected_data = texture_df[['contrast', 'correlation', 'energy', 'homogeneity']].iloc[[40]]
    selected_data = texture_df[['contrast', 'correlation', 'energy', 'homogeneity']].iloc[:]
    selected_data_normalized = scaler.transform(selected_data)
    #print(selected_data_normalized)
    print(predictThresholds(selected_data_normalized))
    
    '''
    all_combi = results_df['combi'].unique()
    filtered_df = results_df[results_df['IoU_Result'] == 0]
    combi_with_zero = filtered_df['combi'].unique()
    difference = np.setdiff1d(combi_with_zero, all_combi)
    print("Unique 'combi' values in filtered_df but not in results_df:")
    print(difference)
    '''
