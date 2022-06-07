from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import random
from sklearn.metrics import classification_report
import pickle
import os

from DataLoading import ImagesLoader
from PreProcessing import PreProcessor
from FeatureExtraction import FeatureExtractor


if __name__ == '__main__':
    # Data Loading

    females_list = os.listdir('./data/Females')
    females_list = [f"./data/Females/{img_path}" for img_path in females_list]
    random.shuffle(females_list)

    males_list = os.listdir('./data/Males')
    males_list = [f"./data/Males/{img_path}" for img_path in males_list]
    random.shuffle(males_list)

    imgs_list = females_list + males_list
    y = np.array(([0] * len(females_list)) + ([1] * len(males_list)))

    assert len(imgs_list) == len(females_list) + len(males_list)
    assert len(y) == len(imgs_list)

    print(f"Number of Female Images: {len(females_list)}")
    print(f"Number of Male Images: {len(males_list)}")
    print(
        f"Ratio of Females in The Dataset: {len(females_list) / len(imgs_list)}")
    print(f"Total Number of Images: {len(imgs_list)}")

    females_split = train_test_split(
        females_list, y[:len(females_list)], test_size=0.2, random_state=42)

    males_spilt = train_test_split(
        males_list[:len(females_list)], y[len(females_list):len(females_list)*2], test_size=0.2, random_state=42)

    X_train = females_split[0] + males_spilt[0] + \
        males_list[len(females_list): len(females_list)+30]
    X_test = females_split[1] + males_spilt[1]
    y_train = np.concatenate([females_split[2], males_spilt[2], y[len(
        females_list)*2:(len(females_list)*2) + 30]])
    y_test = np.concatenate([females_split[3], males_spilt[3]])
    X_test_old = X_test

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    print(f"Number of validation examples: {len(X_test)}")

    X_train_data = ImagesLoader(X_train)
    X_test_data = ImagesLoader(X_test)

    data_pipe = Pipeline([
        ('PreProcessing', PreProcessor()),
        ("Features", FeatureExtractor()),
    ])

    model_pipe = Pipeline([
        ("Standard Scaling", StandardScaler()),
        ("PCA", PCA(n_components=80)),
        ("Estimator", SVC(C=3, class_weight={0: 1.3, 1: 1}))
    ])

    print("Generating Training Features >>>>")
    X_train_transformed = data_pipe.transform(X_train_data)
    print("Training Model >>>>")
    model_pipe.fit(X_train_transformed, y_train)
    print("Generating Validation Features >>>>")
    X_test_transformed = data_pipe.transform(X_test_data)
    print("Generating Results >>>>")
    y_pred = model_pipe.predict(X_test_transformed)

    pickle.dump(data_pipe, open('data_pipeline', 'wb'))
    pickle.dump(model_pipe, open('model_pipeline', 'wb'))

    with open('train_results.txt', "w") as out:
        out.write(classification_report(y_test, y_pred))

    with open('test_data.txt', "w") as out:
        for f in X_test:
            out.write(f)
            out.write('\n')
