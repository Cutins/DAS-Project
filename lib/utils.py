'''
Funzioni da usare insieme al "Main_Vecchio_DataSet"
'''

from tqdm import tqdm
import os
import cv2
import pandas as pd
import numpy as np

def get_data(dataset_folder, tools, target, size=(224, 224), samples=None, balanced=True):
    dataset = {'image': [], 'label': []}
    for tool in tqdm(tools, total=len(tools), leave=False):
        if tool[0] == '.':
            continue
        tool_folder = os.path.join(dataset_folder, tool)

        pos_samples = samples
        neg_samples = pos_samples // (len(tools) - 1) if balanced else samples

        assert (len(os.listdir(tool_folder)) >= pos_samples if tool == target else len(os.listdir(tool_folder)) >= neg_samples), f"Not enough data for tool {tool} - MAX: {len(os.listdir(tool_folder))}!"
        
        for idx, image_name in enumerate(os.listdir(tool_folder)):
            image_path = os.path.join(tool_folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, size)
            img = img / 255.
            img = img.flatten()
            dataset['image'].append(img)
            dataset['label'].append(int(tool == target))
            if samples is not None:
                if tool == target and idx == pos_samples:
                    break
                if tool != target and idx == neg_samples:
                    break 

    return pd.DataFrame(dataset)