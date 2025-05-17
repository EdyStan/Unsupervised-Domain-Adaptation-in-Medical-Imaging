import os
import random
import re

def natural_sort_key(file_name):
    # Regular expression to extract numbers from the file name
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_name)]

def split_files(folder_path):
    # Get all file names in the folder
    file_names = os.listdir(folder_path)

    # Sort the file names considering numbers in the names (natural sorting)
    file_names.sort(key=natural_sort_key)

    # Calculate split sizes
    train_size = 45725
    val_size = 51460
    test_size = len(file_names)

    # Split file names into train, val, and test
    train_files = file_names[:train_size]
    val_files = file_names[train_size:val_size]
    test_files = file_names[val_size:]

    # Write file names to corresponding text files
    with open('train.txt', 'w') as train_file:
        for file_name in train_files:
            train_file.write(file_name + '\n')

    with open('val.txt', 'w') as val_file:
        for file_name in val_files:
            val_file.write(file_name + '\n')

    with open('test.txt', 'w') as test_file:
        for file_name in test_files:
            test_file.write(file_name + '\n')

    print("Files have been successfully split and written to train.txt, val.txt, and test.txt.")

# Specify your folder path here
folder_path = 'images'
split_files(folder_path)
