import numpy as np
import os
import torch
from torch.utils.data import Dataset


class SortOfClevrDataset(Dataset):
    def __init__(self, data_dir):
        # load image
        image_dir = os.path.join(data_dir, 'images')
        image_files = sorted(os.listdir(image_dir))
        self.images = [np.transpose(np.load(os.path.join(image_dir, filename)), (2, 0, 1)) for filename in image_files]
        # load question encoding
        question_dir = os.path.join(data_dir, 'questions')
        question_files = sorted(os.listdir(question_dir))
        self.questions = [np.load(os.path.join(question_dir, filename)) for filename in question_files]
        # load answer array
        self.answers = np.load(os.path.join(data_dir, 'answers.npy'))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image = self.images[idx // 4]
        text = self.questions[idx]
        y = self.answers[idx]
        return torch.tensor(image).float(), torch.tensor(text).squeeze(0), torch.tensor(y)
