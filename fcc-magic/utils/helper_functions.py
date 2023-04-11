# helper_functions.py
import os

# Function to remove all files except .ipynb_checkpoints
def remove_files_except_checkpoints(directory):
    for file in os.listdir(directory):
        if file != '.ipynb_checkpoints':
            os.remove(os.path.join(directory, file))

# Function to create or clean up a directory
def dir_except_checkpoints(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        # This will remove all the files existing in the folder, if you do not want that delete line 14-16
        remove_files_except_checkpoints(directory)