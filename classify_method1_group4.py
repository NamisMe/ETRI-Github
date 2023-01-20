import json
import glob
import torch
from tqdm import tqdm
import gc
import shutil
import os
import cv2
import time

def memory_empty(sample2detect):
    del sample2detect
    torch.cuda.empty_cache()
    gc.collect()

def file_load(split_folder_num):
    
    global metadata_load_time
    start = time.time()
    sample2detect = torch.load(f"detection_metadata/sample2detect_{split_folder_num}.pth", map_location=torch.device("cuda:0")) # metadata
    end = time.time()
    metadata_load_time += end-start
    base_path = f"./laion_face_data/split_{split_folder_num}"
    print(f"execution duration : {(end-start) : .2f} seconds")
    print(f"base_path : {base_path}, split_folder_num : {split_folder_num}")
    
    return base_path, sample2detect
        
def image_and_txt(save_folder, image_cropped_resized, i):
    save_image_path = os.path.join(save_folder, f'{folder}_{key}_{i}.png')
    save_txt_path = os.path.join(save_folder, f'{folder}_{key}_{i}.txt')  
    shutil.copy(txt,save_txt_path)
    cv2.imwrite(save_image_path, image_cropped_resized)  
        
def face_crop_resize(image, faces):
    for i in range(len(faces)):
        
        box, _, _ = faces[i] #face에서 box만 추출
        #Box
        box_lt = (int(box[0]), int(box[1])) #좌상단 좌표
        box_rb = (int(box[2]), int(box[3])) #우하단 좌표
        width = box_rb[0] - box_lt[0]
        height = box_rb[1] - box_lt[1]
        
        max_len = max(width, height) # width와 height 중 더 긴 것을 기준으로 분류
        
        image_cropped = image[box_lt[1]:box_rb[1], box_lt[0]:box_rb[0]]
        try:
            image_cropped_resized = cv2.resize(image_cropped, dsize=(256,256))
        except:
            continue        
        
        if max_len >= 250:
            save_folder = f"./face_result/method1_group4/folder1/{split_folder_num}/"
            image_and_txt(save_folder, image_cropped_resized, i)
        elif max_len >= 200:
            save_folder = f"./face_result/method1_group4/folder2/{split_folder_num}/"
            image_and_txt(save_folder, image_cropped_resized, i)
        elif max_len >= 150:
            save_folder = f"./face_result/method1_group4/folder3/{split_folder_num}/"
            image_and_txt(save_folder, image_cropped_resized, i)
        elif max_len >= 100:
            save_folder = f"./face_result/method1_group4/folder4/{split_folder_num}/"
            image_and_txt(save_folder, image_cropped_resized, i)
        else:
            save_folder = f"./face_result/method1_group4/"
def folder_remove_make(folder_path):
    try:
        shutil.rmtree(folder_path) # 폴더 삭제
        os.mkdir(folder_path) # 폴더 생성
    except:
        os.mkdir(folder_path) # 삭제할 폴더가 없을 경우, 폴더 생성

if __name__ == "__main__":
    
    global metadata_load_time
    metadata_load_time = 0
    folder_remove_make("./face_result/method1_group4/folder1")
    folder_remove_make("./face_result/method1_group4/folder2")
    folder_remove_make("./face_result/method1_group4/folder3")
    folder_remove_make("./face_result/method1_group4/folder4")   
    folder_remove_make("./face_result/method1_group4/group1")
    folder_remove_make("./face_result/method1_group4/group2")
    folder_remove_make("./face_result/method1_group4/group3")
    folder_remove_make("./face_result/method1_group4/group4")

    split_folder_list = []
    base_path = "./laion_face_data"
    for folders in os.listdir(base_path):
        split_folder_list.append(folders.split("_")[1])

    base_dir = "./face_result/method1_group4"
    for dirs in os.listdir(base_dir)[:4]:
        path = os.path.join(base_dir, dirs)
        for folder in split_folder_list:
            os.mkdir(os.path.join(path, folder))

    #folder_num_list = ('00000','00002','00005','00008','00013','00015','00017','00018','00021','00022','00024','00025','00028')
    entire_start = time.time()
    
    # Looking for sample(Delete when you take the full dataset)
    f_num = 10
    j_num = 100

    for split_folder_num in split_folder_list:
        print(f"----------------------{split_folder_num} start-----------------------")
        #path, metadata 정의
        base_path, sample2detect = file_load(split_folder_num)
        folders = os.listdir(base_path)
        for folder in folders[:f_num]:
            e_folder = os.path.join(base_path, folder)
            json_files = glob.glob(e_folder+'/*.json')

            for json_file in tqdm(json_files[:j_num]):
                with open(json_file) as data_file:
                    json_info = json.load(data_file)
                
                #json info 요소들
                SAMPLE_id = json_info['SAMPLE_ID']
                key = json_info['key']
                caption = json_info['caption']
                
                faces=sample2detect[SAMPLE_id]
                image=json_file.replace("json","jpg")
                image=cv2.imread(image)
                txt=json_file.replace("json","txt")

                #Face crop and resize
                face_crop_resize(image, faces) 
            print(folder)
        #한 폴더 끝날 때 마다 메모리 제거

        print(f"------------------------{split_folder_num} end-----------------------")
        memory_empty(sample2detect)
    crop_resize_end = time.time()
    print(f"metadata load time : {metadata_load_time :.2f}s")
    print(f"method1_group4 crop_resize execution time : {(crop_resize_end - entire_start - metadata_load_time) :.2f}s")
    
    copy_start = time.time()
    
    #각 folder에서 group으로 복사
    shutil.copytree("./face_result/method1_group4/folder1", "./face_result/method1_group4/group1/folder1")
    shutil.copytree("./face_result/method1_group4/folder1", "./face_result/method1_group4/group2/folder1")
    shutil.copytree("./face_result/method1_group4/folder1", "./face_result/method1_group4/group3/folder1")
    shutil.copytree("./face_result/method1_group4/folder1", "./face_result/method1_group4/group4/folder1")
    shutil.copytree("./face_result/method1_group4/folder2", "./face_result/method1_group4/group2/folder2")
    shutil.copytree("./face_result/method1_group4/folder2", "./face_result/method1_group4/group3/folder2")
    shutil.copytree("./face_result/method1_group4/folder2", "./face_result/method1_group4/group4/folder2")
    shutil.copytree("./face_result/method1_group4/folder3", "./face_result/method1_group4/group3/folder3")
    shutil.copytree("./face_result/method1_group4/folder3", "./face_result/method1_group4/group4/folder3")
    shutil.copytree("./face_result/method1_group4/folder4", "./face_result/method1_group4/group4/folder4")
    
    copy_end = time.time()
    print(f"method1_group4 copy execution time : {copy_end - copy_start :.2f}s")
    print(f"method1_group4 pure execution time : {(copy_end - entire_start - metadata_load_time) :.2f}s")
    print(f"method1_group4 total execution time : {copy_end - entire_start:.2f}s")
    print(f"method1_group4 Predicted time for full dataset : {(copy_end - entire_start - metadata_load_time)*(153/f_num)*(6700/j_num):.2f}s")
