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

def build_dataset(n):
  img_data = []
  qst_data = []
  ans_data = []
  sentence_data = []
  for _ in range(n):
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
    sentences = []
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

        # Create sentence
        sentence = f"A {IDX_TO_COLOR[o1_c]} "
        sentence = sentence + "circle " if o1_shape == "c" else sentence + "rectangle "

        # Left
        if q_type == 0:
          # Question asks "Is <o1> left of <o2>?"
          if o1_center[0]<o2_center[0]:
            sentence += "is left of "
            answer = [1, 0]
          else:
            sentence += "is right of "
            answer = [0, 1]
        # Right
        elif q_type == 1:
          # Question asks "Is <o1> right of <o2>?"
          if o1_center[0]>o2_center[0]:
            sentence += "is right of "
            answer = [1, 0]
          else:
            sentence += "is left of "
            answer = [0, 1]
        # Above
        elif q_type == 2:
          # Question asks "Is <o1> above <o2>?"
          if o1_center[1]<o2_center[1]:
            sentence += "is above "
            answer = [1, 0]
          else:
            sentence += "is below "
            answer = [0, 1]
        # Below
        else:
          # Question asks "Is <o1> below <o2>?"
          if o1_center[1]>o2_center[1]:
            sentence += "is below "
            answer = [1, 0]
          else:
            sentence += "is above "
            answer = [0, 1]

        sentence += f"a {IDX_TO_COLOR[o2_c]} "
        sentence = sentence + "circle." if o2_shape == "c" else sentence + "rectangle."
        sentences.append(sentence)
        binary_questions.append(question)
        binary_answers.append(answer)

    bgr_img = img/255
    rgb_img = bgr_img[...,::-1].copy()
    img_data.append(rgb_img)
    sentence_data += sentences
    qst_data += binary_questions
    ans_data += binary_answers

  img_data = np.array(img_data)
  qst_data = np.array(qst_data)
  ans_data = np.array(ans_data)

  return img_data, qst_data, ans_data, sentence_data

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
    filepath = os.path.join(img_dir, '{:010d}'.format(idx))
    np.save(filepath, img)

  for idx, qst in enumerate(qst_data):
    filepath = os.path.join(qst_dir, '{:010d}'.format(idx))
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
  sentence_idx = 0
  for idx, img in enumerate(img_data):
    sub_dir = os.path.join(img_dir, '{:010d}'.format(idx))
    if not os.path.exists(sub_dir):
      os.makedirs(sub_dir)
    filepath = os.path.join(sub_dir, '{:010d}'.format(idx))
    np.save(filepath, img)
    sentences = sentence_data[sentence_idx:sentence_idx + 15]
    with open(os.path.join(filepath+'.txt'),  'w+') as f:
      f.write('\n'.join(sentences))
    sentence_idx += 15


if __name__ == '__main__':
    print("Building training dataset...")
    img_data, qst_data, ans_data, sentence_data = build_dataset(TRAIN_DATA_SIZE)
    print("Saving training dataset...")
    save_rn_data(DATA_DIR, 'train', img_data, qst_data, ans_data)
    save_dalle_data(DATA_DIR, img_data, sentence_data)

    print("Building val dataset...")
    img_data, qst_data, ans_data, _ = build_dataset(VAL_DATA_SIZE)
    print("Saving val dataset...")
    save_rn_data(DATA_DIR, 'val', img_data, qst_data, ans_data)

    print("Building test dataset...")
    img_data, qst_data, ans_data, _ = build_dataset(TEST_DATA_SIZE)
    print("Saving test dataset...")
    save_rn_data(DATA_DIR, 'test', img_data, qst_data, ans_data)