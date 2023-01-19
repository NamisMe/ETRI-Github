import json
import glob
import torch
import pandas as pd
from tqdm import tqdm
import gc
import os
import cv2
import numpy as np
import random
import time
from urllib.request import Request, urlopen

def memory_empty(sample2detect): # 메모리 해제
    del sample2detect
    torch.cuda.empty_cache()
    gc.collect()

def file_load(folder_num): #face-metadata load, folder_path 정의
    
    start = time.time()
    sample2detect = torch.load(f"detection_metadata/sample2detect_{folder_num}.pth", map_location=torch.device("cuda:0")) # metadata
    end = time.time()
    base_path = f"F:/NAM/laion_face_data/split_{folder_num}"
    print(f"execution duration : {(end-start) : .2f} seconds")
    print(f"base_path : {base_path}, folder_num : {folder_num}")
    
    return base_path, sample2detect

def classify(num): # 각 범위에 해당하는 이미지의 분류 후 개수 count
    global range0
    global range1
    global range2
    global range3
    global range4
    global range5
    
    if 0 < num <= 50 :
        range0 += 1
    elif 50 < num <= 100 :
        range1 += 1
    elif 100< num <= 150 :
        range2 += 1
    elif 150 < num <= 200 :
        range3 += 1
    elif 200 < num <= 250 :
        range4 += 1
    else:
        range5 += 1
        
def face_crop_classify(faces): # metadata를 통하여 faces에서 이미지를 추출. (1~N개)
    for i in range(len(faces)):
        box, _, _ = faces[i] 
        #Box
        box_lt = (int(box[0]), int(box[1]))
        box_rb = (int(box[2]), int(box[3]))
        #print(box_lt, box_rb)
        width = box_rb[0] - box_lt[0]
        height = box_rb[1] - box_lt[1]
        max_len = max(width, height) # width와 height 중 더 긴 것을 기준으로 분류
        classify(max_len)

#Main 

if __name__ == "__main__": 
    global range0
    global range1
    global range2
    global range3
    global range4
    global range5
    global sum

    range0 = 0
    range1 = 0
    range2 = 0
    range3 = 0
    range4 = 0
    range5 = 0

    folder_num_list = ('00000','00002','00005','00008','00013','00015','00017','00018','00021','00022','00024','00025','00028')
    for folder_num in folder_num_list:
        print(f"----------------------{folder_num} start-----------------------")
        #path, metadata 정의
        base_path, sample2detect = file_load(folder_num)
        folders = os.listdir(base_path)
        for folder in folders:
            e_folder = os.path.join(base_path, folder)
            json_files = glob.glob(e_folder+'/*.json')
            print(folder)
            for json_file in tqdm(json_files):
                with open(json_file) as data_file:
                    json_info = json.load(data_file)
                SAMPLE_id = json_info['SAMPLE_ID']
                key = json_info['key']
                faces=sample2detect[SAMPLE_id]
                #Face crop and resize
                face_crop_classify(faces)
        sum = (range0 + range1 + range2 + range3 + range4 + range5)
        print(f"Result for [{sum}] files")
        print(f"   [0~50] : count: {range0} ratio:{(range0/sum)*100 :.2f}%")
        print(f" [50~100] : count: {range1} ratio:{(range1/sum)*100 :.2f}%")
        print(f"[100~150] : count: {range2} ratio:{(range2/sum)*100 :.2f}%")
        print(f"[150~200] : count: {range3} ratio:{(range3/sum)*100 :.2f}%")
        print(f"[200~250] : count: {range4} ratio:{(range4/sum)*100 :.2f}%")
        print(f"[Over250] : count: {range5} ratio:{(range5/sum)*100 :.2f}%")
        
        #한 폴더 끝날 때 마다 메모리 제거

        print(f"------------------------{folder_num} end-----------------------")
        memory_empty(sample2detect)
