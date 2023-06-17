import numpy as np
import pandas as pd
import tensorflow.keras as ks
import cv2

from lib.config import *
np.random.seed(SEED)

def load_data():
    # Load DataFrame
    (x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()

    # Preprocess the image: reshape & normalize
    preprocess = lambda x: cv2.resize(x, SIZE).flatten() / 255.
    x_train = [preprocess(x) for x in x_train]
    x_test  = [preprocess(x) for x in x_test]

    # Assign the TARGET
    y_train = [1 if y == TARGET else 0 for y in y_train]
    y_test  = [1 if y == TARGET else 0 for y in y_test]

    # Balance the Dataset, select the SAMPLES
    df_train = pd.DataFrame({'image': x_train, 'label': y_train}).groupby('label')
    df_train_balanced = df_train.sample(SAMPLES, random_state=SEED).sample(frac=1, random_state=SEED)

    images_train = np.array([df_train_balanced['image'][agent:SAMPLES:N_AGENTS].tolist() for agent in range(N_AGENTS)])# [N_AGENTS, SAMPLES_per_agent, image_size]
    labels_train = np.array([df_train_balanced['label'][agent:SAMPLES:N_AGENTS].tolist() for agent in range(N_AGENTS)])# [N_AGENTS, SAMPLES_per_agent]

    images_test = np.array([df_train_balanced['image'][agent+SAMPLES::N_AGENTS].tolist() for agent in range(N_AGENTS)])
    labels_test = np.array([df_train_balanced['label'][agent+SAMPLES::N_AGENTS].tolist() for agent in range(N_AGENTS)])

    return images_train, labels_train, images_test, labels_test