import clip
import cv2
import numpy as np
import os
import random
import torch

from config import (
    TRAIN_DATA_GEN_DIR, VAL_DATA_GEN_DIR, TEST_DATA_GEN_DIR, 
    TRAIN_DATA_SIZE, VAL_DATA_SIZE, TEST_DATA_SIZE, 
    IMAGE_SIZE, OBJECT_SIZE,
    CLIP_MODEL
)

'''
Generate Dataset
Logic based off of Sort Of Clevr Dataset generation
'''
above_syn = ['above', 'on top of', 'upon', 'over', 'higher than']
below_syn = ['below', 'underneath', 'under', 'lower than']
colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]

color_to_word = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']

# clip model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(CLIP_MODEL, device=device)

# Helper function to create a center and shape avoiding collisions
def shape_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+OBJECT_SIZE, IMAGE_SIZE - OBJECT_SIZE, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((OBJECT_SIZE * 2) ** 2):
                    pas = False
        if pas:
            shape = 'r' if random.random() < .5 else 'c'
            return center, shape

def save_dataset(img_data, qst_data, ans_data, dirpath):
    '''
    Save each image as a .npy file. 
    Save all questions as a single .txt file.
    Save all answers as a single .npy file.
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

def build_dataset(n, dirpath):
    img_data, qst_data, ans_data = [], [], []
    for _ in range(n):
        c1 = random.randint(0,5)
        c2 = random.randint(0,5)
        # Generate 2 unique colors
        while c1 == c2:
            c2 = random.randint(0, 5)
        objects = []
        (c1_center, c1_shape) = shape_generate(objects)
        objects.append((c1,c1_center,c1_shape))
            
        # Create 2 shapes, c1 and c2 that guaranteed to be stacked
        (c2_center, c2_shape) = shape_generate(objects)
        while True:
            c1_x = c1_center[0]
            c2_x = c2_center[0]
            delta = 2*OBJECT_SIZE
            if abs(c2_x-c1_x) < delta:
                break
            else: 
                (c2_center, c2_shape) = shape_generate(objects)
        objects.append((c2, c2_center, c2_shape))

        # Create 4 other random objects
        for color_id, color in enumerate(colors):  
            if color_id == c1 or color_id == c2:
                continue
            else:
                (center, shape) = shape_generate(objects)
                objects.append((color_id, center, shape))

        # Create cv2 image
        img = np.ones((IMAGE_SIZE,IMAGE_SIZE,3)) * 255
        for o in objects:
            (color, center, shape) = o
            if shape=='r':
                start = (center[0]-OBJECT_SIZE, center[1]-OBJECT_SIZE)
                end = (center[0]+OBJECT_SIZE, center[1]+OBJECT_SIZE)
                cv2.rectangle(img, start, end, colors[color], -1)
            else:
                cv2.circle(img, center, OBJECT_SIZE, colors[color], -1)
        img_data.append(img)

        # Generate sentences
        # data = []
        c1_y = c1_center[1]
        c2_y = c2_center[1]
        c1_shape_word = 'rectangle' if c1_shape == 'r' else 'circle'
        c2_shape_word = 'rectangle' if c2_shape == 'r' else 'circle'
        # Choose 'above' synonym and 'below' synonym
        above_word = above_syn[random.randint(0, len(above_syn)-1)]
        below_word = below_syn[random.randint(0, len(below_syn)-1)]
        s1 = f'Is the {color_to_word[c1]} {c1_shape_word} {above_word} the {color_to_word[c2]} {c2_shape_word}?'
        s2 = f'Is the {color_to_word[c2]} {c2_shape_word} {below_word} the {color_to_word[c1]} {c1_shape_word}?'
        s3 = f'Is the {color_to_word[c2]} {c2_shape_word} {above_word} the {color_to_word[c1]} {c1_shape_word}?'
        s4 = f'Is the {color_to_word[c1]} {c1_shape_word} {below_word} the {color_to_word[c2]} {c2_shape_word}?'
        sentences = [s1, s2, s3, s4]
        
        for i, s in enumerate(sentences):
            ans = 0
            # c1 is above c2: s1, s2 correct. s3, s4 incorrect; c2 is above c1: s1, s2 incorrect. s3, s4 correct
            if (c1_y < c2_y and i <= 1) or (c1_y > c2_y and i > 1):
                ans = 1
            # data.append((img, s, ans))
            # encode sentence use clip
            s_token = clip.tokenize(s).to(device)
            with torch.no_grad():
                s_encoded = model.encode_text(s_token)
            qst_data.append(s_encoded.cpu().numpy())
            ans_data.append(ans)
    
    # write to file
    save_dataset(img_data, qst_data, ans_data, dirpath)   
  
if __name__ == '__main__':
    build_dataset(TRAIN_DATA_SIZE, TRAIN_DATA_GEN_DIR)
    build_dataset(VAL_DATA_SIZE, VAL_DATA_GEN_DIR)
    build_dataset(TEST_DATA_SIZE, TEST_DATA_GEN_DIR)