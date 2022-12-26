from random import choice
from pathlib import Path

import wandb

import numpy as np

# torch

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# vision imports

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import OpenAIDiscreteVAE, DiscreteVAE, DALLE

# config
from config import WANDB_KEY, DALLE_PATH, VAE_PATH, MODEL_DIM, DEPTH, HEADS, DIM_HEAD, GRAD_CLIP_NORM, IMAGE_TEXT_FOLDER, LEARNING_RATE, BATCH_SIZE_DALLE, EPOCHS

# Tokenizer

all_words = ['red', 'green', 'blue', 'orange', 'gray', 'yellow'] + ['circle', 'rectangle'] + ['a', 'is', 'of'] + ['above', 'below', 'right', 'left']
word_tokens = dict(zip(all_words, range(1, len(all_words) + 1)))
token2word = {v: k for k, v in word_tokens.items()}
longest_caption = 'a blue rectangle is left of a yellow circle.'
longest_caption_len = len(longest_caption.split())

print('Tokens:', word_tokens)

RESUME = DALLE_PATH is not None

if RESUME:
    print('Resuming previous run...')
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'

    loaded_obj = torch.load(DALLE_PATH)

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']

    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    else:
        vae = OpenAIDiscreteVAE()

    dalle_params = dict(        
        **dalle_params
    )

    IMAGE_SIZE = vae.image_size

else:
    if VAE_PATH is not None:
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        vae_klass = OpenAIDiscreteVAE
        vae = vae_klass()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        vae = vae,
        num_text_tokens = len(word_tokens) + 1,    # vocab size for text
        text_seq_len = longest_caption_len,
        dim = MODEL_DIM,
        depth = DEPTH,
        heads = HEADS,
        dim_head = DIM_HEAD,
        reversible = True
    )

# helpers

def save_model(path):
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'weights': dalle.state_dict()
    }

    torch.save(save_obj, path)

# dataset loading

class TextImageDataset(Dataset):
    def __init__(self, folder, image_size = 64):
        super().__init__()
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]

        image_files = [
            *path.glob('**/*.npy'),
        ]

        text_files = {t.stem: t for t in text_files}
        image_files = {i.stem: i for i in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        image = np.load(image_file)
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = choice(descriptions)
        
        words = description.replace('.','').lower().split()
        tokens = [word_tokens[i] for i in words]
        tokens += [0]*(longest_caption_len - len(tokens))
        tokenized_text = torch.Tensor(tokens).type(torch.int64).squeeze(0)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).type(torch.FloatTensor)

        return tokenized_text, image_tensor

# create dataset and dataloader

ds = TextImageDataset(
    IMAGE_TEXT_FOLDER,
    image_size = IMAGE_SIZE
)

assert len(ds) > 0, 'dataset is empty'
print(f'{len(ds)} image-text pairs found for training')

dl = DataLoader(ds, batch_size = BATCH_SIZE_DALLE, shuffle = True, drop_last = True)

for (i, el) in enumerate(dl):
    print('Data example:')
    print(el[0].shape, el[1].shape)
    break

# initialize DALL-E

dalle = DALLE(**dalle_params).cuda()

if RESUME:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(dalle.parameters(), lr = LEARNING_RATE)

# experiment tracker

model_config = dict(
    depth = DEPTH,
    heads = HEADS,
    dim_head = DIM_HEAD
)

if WANDB_KEY:
    wandb.login(key=WANDB_KEY)
    run = wandb.init(project = 'dalle_train_transformer', resume = RESUME, config = model_config)

# training

for epoch in range(EPOCHS):
    for i, (text, images) in enumerate(dl):
        text, images = map(lambda t: t.cuda(), (text, images))

        loss = dalle(text, images, return_loss = True)

        loss.backward()
        clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)

        opt.step()
        opt.zero_grad()

        log = {}

        if i % 10 == 0:
            print(epoch, i, f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }

        if i % 100 == 0:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = ' '.join([token2word[i] for i in token_list])

            image = dalle.generate_images(
                text[:1],
                filter_thres = 0.9    # topk sampling at 0.9
            )

            save_model(f'./dalle.pt')
            if WANDB_KEY:
                wandb.save(f'./dalle.pt')
                log = {
                    **log,
                    'image': wandb.Image(image, caption = decoded_text)
                }
        if WANDB_KEY:
            wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end

    if WANDB_KEY:
        model_artifact = wandb.Artifact('trained-dalle', type = 'model', metadata = dict(model_config))
        model_artifact.add_file('dalle.pt')
        run.log_artifact(model_artifact)

save_model(f'./dalle-final.pt')
if WANDB_KEY:
    wandb.save('./dalle-final.pt')
    model_artifact = wandb.Artifact('trained-dalle', type = 'model', metadata = dict(model_config))
    model_artifact.add_file('dalle-final.pt')
    run.log_artifact(model_artifact)

    wandb.finish()
