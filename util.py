import os
import requests
import urllib.request
import zipfile
import io

import pandas as pd
import numpy as np

path = './data'


def fetch_data_sets():
    print('fetching test and train datasets from kaggle . . .')
    url = 'https://www.kaggle.com/c/3004/download-all'
    train_set_csv_path = os.path.join(path, 'train.csv')
    test_set_csv_path = os.path.join(path, 'test.csv')
    if os.path.exists(train_set_csv_path) and os.path.exists(test_set_csv_path):
        return
    os.makedirs(path, exist_ok=True)
    response = urllib.request.urlopen(url)  # requests.get(url)
    zipped_file_path = './data/data.zip'
    temp = open(zipped_file_path, 'wb')
    temp.write(response.read())
    zf = zipfile.ZipFile(zipped_file_path)
    zf.extractall(path=path)
    zf.close()


def load_data_sets():
    fetch_data_sets()
    train_set_csv_path = os.path.join(path, 'train.csv')
    test_set_csv_path = os.path.join(path, 'test.csv')
    return pd.read_csv(train_set_csv_path), pd.read_csv(test_set_csv_path)


def split_train_test_naive(data, test_ratio):
    # Don't use this anymore, just use sklearn's implementation 😁
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
