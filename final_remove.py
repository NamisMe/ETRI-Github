import os
import shutil
import glob
from tqdm import tqdm

base_path0 = "./face_result/method1_group3/group1/folder1"
base_path1 = "./face_result/method1_group3/group2/folder1"            
base_path2 = "./face_result/method1_group3/group2/folder2"
base_path3 = "./face_result/method1_group3/group3/folder1"
base_path4 = "./face_result/method1_group3/group3/folder2"
base_path5 = "./face_result/method1_group3/group3/folder3"

paths = (base_path0, base_path1, base_path2, base_path3, base_path4, base_path5)

if __name__ == "__main__":
    for base_path in paths:
        for dirs in os.listdir(base_path):
            folders = os.path.join(base_path, dirs)
            [os.remove(files) for files in tqdm(glob.glob(os.path.join(folders, "*.txt")))]
    
