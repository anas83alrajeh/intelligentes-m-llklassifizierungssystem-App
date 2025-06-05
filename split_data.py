# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 13:00:37 2025

@author: anasa
"""

import os
import random
import shutil

def split_data(source_dir, output_dir, train_ratio=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = os.listdir(category_path)
        random.shuffle(images)

        train_count = int(len(images) * train_ratio)

        train_images = images[:train_count]
        test_images = images[train_count:]

        # Erstellen der Kategorienordner innerhalb von train und test
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))

        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category, img))

if __name__ == "__main__":
    split_data(source_dir='data', output_dir='data_split', train_ratio=0.8)
    print("Die Daten wurden in train- und test-Ordner im Verzeichnis data_split/ aufgeteilt.")

