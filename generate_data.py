"""
@Author: Vishaal Agartha
@Date: 11/17/2022
"""

import cv2
import numpy as np
import os
import random

from config import (
    DATA_DIR,
    TRAIN_DATA_SIZE, VAL_DATA_SIZE, TEST_DATA_SIZE, 
    DALLE_TRAIN_DATA_SIZE,
    IMAGE_SIZE, OBJECT_SIZE,
)

'''
Generate Dataset
Logic based off of Sort Of Clevr Dataset generation
'''
QUESTION_SIZE = 20
O1_OFFSET = 0
O1_SHAPE_OFFSET = 6
O2_OFFSET = 8
O2_SHAPE_OFFSET = 14
IMAGE_SIZE = 64
OBJECT_SIZE = 5
QUESTION_OFFSET = 16
COLORS = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]

IDX_TO_COLOR = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']

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

def build_dataset_rn(n_rn_examples):
  img_data = []
  qst_data = []
  ans_data = []
  while len(ans_data)<n_rn_examples:
    objects = []
    img = np.ones((IMAGE_SIZE,IMAGE_SIZE,3)) * 255
    for color_id,color in enumerate(COLORS):  
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
        [R, G, B, O, K, Y, circle, rectangle, R, G, B, O, K, Y, circle, rectangle, left, right, above, below]
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

        # Encode question type
        q_type = random.randint(0, 3)
        question[QUESTION_OFFSET+q_type] = 1
        # Left
        if q_type == 0:
          # Question asks "Is <o1> left of <o2>?"
          if o1_center[0]<o2_center[0]:
            answer = 1
          else:
            answer = 0
        # Right
        elif q_type == 1:
          # Question asks "Is <o1> left of <o2>?"
          if o1_center[0]<o2_center[0]:
            answer = 0
          else:
            answer = 1
        # Above
        elif q_type == 2:
          # Question asks "Is <o1> above <o2>?"
          if o1_center[1]<o2_center[1]:
            answer = 1
          else:
            answer = 0
        # Below
        else:
          # Question asks "Is <o1> below <o2>?"
          if o1_center[1]<o2_center[1]:
            answer = 0
          else:
            answer = 1

        binary_questions.append(question)
        binary_answers.append(answer)
    img = img/255
    img_data += [img] * len(binary_questions)
    qst_data += binary_questions
    ans_data += binary_answers
  img_data = np.array(img_data[:n_rn_examples])
  qst_data = np.array(qst_data[:n_rn_examples])
  ans_data = np.array(ans_data[:n_rn_examples])
  return img_data, qst_data, ans_data

def build_dataset_dalle(n_dalle_examples):
  img_data = []
  sentence_data = []
  while len(sentence_data)<n_dalle_examples:
    objects = []
    img = np.ones((IMAGE_SIZE,IMAGE_SIZE,3)) * 255
    for color_id,color in enumerate(COLORS):  
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
    o1, o2 = random.sample(objects, 2)
    o1_c, o1_center, o1_shape = o1
    o2_c, o2_center, o2_shape = o2
    sentence = f"A {IDX_TO_COLOR[o1_c]} "
    sentence = sentence + "circle " if o1_shape == 'c' else sentence + "rectangle "
    # Choose between vertical and horizontal
    if random.random()<0.5:
      if o1_center[0]<o2_center[0]:
        sentence += "is left of "
      else:
        sentence += "is right of "
    else:
      if o1_center[1]<o2_center[1]:
        sentence += "is above "
      else:
        sentence += "is below "
    sentence += f"a {IDX_TO_COLOR[o2_c]} "
    sentence = sentence + "circle." if o2_shape == 'c' else sentence + "rectangle."
    img = img/255
    img_data.append(img)
    sentence_data.append(sentence)
  img_data = np.array(img_data)
  return img_data, sentence_data

def save_rn_data(dirpath, subpath, img_data, qst_data, ans_data):
  rn_dir_path = os.path.join(dirpath, 'rn')
  sub_dir = os.path.join(rn_dir_path, subpath)
  img_dir = os.path.join(sub_dir, 'images')
  qst_dir = os.path.join(sub_dir, 'questions')
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)
  if not os.path.exists(rn_dir_path):
    os.makedirs(rn_dir_path)
  if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)
  if not os.path.exists(qst_dir):
    os.makedirs(qst_dir)

  for idx, img in enumerate(img_data):
    filepath = os.path.join(img_dir, '{:05d}'.format(idx))
    np.save(filepath, img)

  for idx, qst in enumerate(qst_data):
    filepath = os.path.join(qst_dir, '{:06d}'.format(idx))
    np.save(filepath, qst)
  np.save(os.path.join(sub_dir, 'answers'), np.array(ans_data))

def save_dalle_data(dirpath, img_data, sentence_data):
  dalle_dir_path = os.path.join(dirpath, 'dalle')
  img_dir = os.path.join(dalle_dir_path, 'images')
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)
  if not os.path.exists(dalle_dir_path):
    os.makedirs(dalle_dir_path)
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  for idx, img in enumerate(img_data):
    sub_dir = os.path.join(img_dir, '{:05d}'.format(idx))
    if not os.path.exists(sub_dir):
      os.makedirs(sub_dir)
    filepath = os.path.join(sub_dir, '{:05d}'.format(idx))
    np.save(filepath, img)
    with open(os.path.join(filepath+'.txt'),  'w+') as f:
      f.write(sentence_data[idx] + '\n')


if __name__ == '__main__':
    print("Building training dataset...")
    img_data, qst_data, ans_data = build_dataset_rn(TRAIN_DATA_SIZE)
    print("Saving training dataset...")
    save_rn_data(DATA_DIR, 'train', img_data, qst_data, ans_data)

    print("Building val dataset...")
    img_data, qst_data, ans_data = build_dataset_rn(VAL_DATA_SIZE)
    print("Saving val dataset...")
    save_rn_data(DATA_DIR, 'val', img_data, qst_data, ans_data)

    print("Building test dataset...")
    img_data, qst_data, ans_data = build_dataset_rn(TEST_DATA_SIZE)
    print("Saving test dataset...")
    save_rn_data(DATA_DIR, 'test', img_data, qst_data, ans_data)

    print("Building DALLE training dataset...")
    dalle_img_data, sentence_data = build_dataset_dalle(DALLE_TRAIN_DATA_SIZE)
    print("Saving DALLE training dataset...")
    save_dalle_data(DATA_DIR, dalle_img_data, sentence_data)