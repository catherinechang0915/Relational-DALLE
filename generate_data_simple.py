"""
@Author: Vishaal Agartha
@Date: 11/17/2022
"""

import cv2
import numpy as np
import os
import random
import torch

from config import (
    TRAIN_DATA_GEN_DIR, VAL_DATA_GEN_DIR, TEST_DATA_GEN_DIR, 
    TRAIN_DATA_SIZE, VAL_DATA_SIZE, TEST_DATA_SIZE, 
    IMAGE_SIZE, OBJECT_SIZE,
)

'''
Generate Dataset
Logic based off of Sort Of Clevr Dataset generation
'''
QUESTION_SIZE = 16
O1_OFFSET = 0
O1_SHAPE_OFFSET = 6
O2_OFFSET = 8
O2_SHAPE_OFFSET = 14
colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]

def save_dataset(img_data, qst_data, ans_data, dirpath):
    '''
    Save each image as a .npy file. 
    Save all questions as a single .txt file.
    Save all answers as a single .npy file.
    Save all sentences as a single .txt file.
    '''
    img_dir = os.path.join(dirpath, 'images')
    qst_dir = os.path.join(dirpath, 'questions')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(qst_dir):
        os.makedirs(qst_dir)
    
    for idx, img in enumerate(img_data):
        filepath = os.path.join(img_dir, '{:05d}'.format(idx))
        np.save(filepath, img)

    for idx, qst in enumerate(qst_data):
        filepath = os.path.join(qst_dir, '{:05d}'.format(idx))
        np.save(filepath, qst)
    np.save(os.path.join(dirpath, 'answers'), np.array(ans_data))

def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+OBJECT_SIZE, IMAGE_SIZE - OBJECT_SIZE, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((OBJECT_SIZE * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_dataset(n, dirpath):
    img_data = []
    qst_data = []
    ans_data = []
    for _ in range(n):
        objects = []
        img = np.ones((IMAGE_SIZE,IMAGE_SIZE,3)) * 255
        for color_id,color in enumerate(colors):  
            center = center_generate(objects)
            if random.random()<0.5:
                start = (center[0]-OBJECT_SIZE, center[1]-OBJECT_SIZE)
                end = (center[0]+OBJECT_SIZE, center[1]+OBJECT_SIZE)
                cv2.rectangle(img, start, end, color, -1)
                objects.append((color_id,center,'r'))
            else:
                center_ = (center[0], center[1])
                cv2.circle(img, center_, OBJECT_SIZE, color, -1)
                objects.append((color_id,center,'c'))
        binary_questions = []
        binary_answers = []
        """Binary Relational questions"""
        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                question = [0]*QUESTION_SIZE
                """
                Question encoding:
                0-5 correspond to o1 color
                6-7 correspond to o1 shape

                8-14 correspond to o2 color
                14-15 correspond to o2 shape
                [R, G, B, O, K, Y, circle, rectangle, R, G, B, O, K, Y, circle, rectangle]
                """
                o1_c, o1_center, o1_shape = objects[i]
                o2_c, o2_center, o2_shape = objects[j]

                # Encode color
                question[O1_OFFSET + o1_c] = 1
                question[O2_OFFSET + o2_c] = 1

                # Encode shape
                o1_shape_idx = 0 if o1_shape == 'c' else 1
                o2_shape_idx = 0 if o2_shape == 'c' else 1
                question[O1_SHAPE_OFFSET + o1_shape_idx] = 1
                question[O2_SHAPE_OFFSET + o2_shape_idx] = 1

                # Question asks "Is <o1> above <o2>?"
                """Answer : [yes, no]"""
                if o1_center[1]<o2_center[1]:
                    answer = [1, 0]
                else:
                    answer = [0, 1]
                binary_questions.append(question)
                binary_answers.append(answer)


        img = img/255
        img_data.append(img)
        qst_data += binary_questions
        ans_data += binary_answers
    img_data = np.array(img_data)
    qst_data = np.array(qst_data)
    ans_data = np.array(ans_data)
    save_dataset(img_data, qst_data, ans_data, dirpath)   


if __name__ == '__main__':
    print("Building training dataset...")
    build_dataset(TRAIN_DATA_SIZE, TRAIN_DATA_GEN_DIR)
    print("Building validation dataset...")
    build_dataset(VAL_DATA_SIZE, VAL_DATA_GEN_DIR)
    print("Building test dataset...")
    build_dataset(TEST_DATA_SIZE, TEST_DATA_GEN_DIR)