import math
from math import sqrt
import os
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# wandb
import wandb

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.utils import make_grid, save_image

# dalle classes

from dalle_pytorch import DiscreteVAE

# constants
from config import WANDB_KEY, DATA_DIR, EPOCHS, BATCH_SIZE_VAE, LEARNING_RATE, LR_DECAY_RATE, NUM_TOKENS, NUM_LAYERS, NUM_RESNET_BLOCKS, EMB_DIM, HID_DIM, STARTING_TEMP, TEMP_MIN, ANNEAL_RATE, NUM_IMAGES_SAVE 



def npy_loader(path):
    sample = torch.permute(torch.from_numpy(np.load(path)), (2, 0, 1))
    return sample.type(torch.FloatTensor)

# data
IMAGE_SIZE = 64
IMAGE_PATH = f'{DATA_DIR}/dalle/images'

dataset = DatasetFolder(
    root=IMAGE_PATH,
    loader=npy_loader,
    extensions='.npy'
)

assert len(dataset) > 0, 'folder does not contain any images'
print(f'{len(dataset)} images found for training')
dl = DataLoader(dataset, BATCH_SIZE_VAE, shuffle = True)

vae_params = dict(
    image_size = IMAGE_SIZE,
    num_layers = NUM_LAYERS,
    num_tokens = NUM_TOKENS,
    codebook_dim = EMB_DIM,
    hidden_dim   = HID_DIM,
    num_resnet_blocks = NUM_RESNET_BLOCKS
)

vae = DiscreteVAE(
    **vae_params,
    smooth_l1_loss = False,
    kl_div_loss_weight = 0
).cuda()

def save_model(path):
    save_obj = {
        'hparams': vae_params,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

# optimizer

opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

# weights & biases experiment tracking

model_config = dict(
    num_tokens = NUM_TOKENS,
    smooth_l1_loss = False,
    num_resnet_blocks = NUM_RESNET_BLOCKS,
    kl_loss_weight = 0
)

if WANDB_KEY:
    wandb.login(key=WANDB_KEY)
    run = wandb.init(
        project = 'dalle_train_vae',
        job_type = 'train_model',
        config = model_config
    )

# starting temperature

global_step = 0
temp = STARTING_TEMP

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(dl):
        images = images.cuda()

        loss, recons = vae(
            images,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        logs = {}

        if i % 100 == 0:
            k = NUM_IMAGES_SAVE

            with torch.no_grad():
                codes = vae.get_codebook_indices(images[:k])
                hard_recons = vae.decode(codes)

            images, recons = map(lambda t: t[:k], (images, recons))
            images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
            images, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (images, recons, hard_recons))
            if WANDB_KEY:
                logs = {
                    **logs,
                    'sample images':        wandb.Image(images, caption = 'original images'),
                    'reconstructions':      wandb.Image(recons, caption = 'reconstructions'),
                    'hard reconstructions': wandb.Image(hard_recons, caption = 'hard reconstructions'),
                    'codebook_indices':     wandb.Histogram(codes),
                    'temperature':          temp
                }

            save_model(f'./vae.pt')
            if WANDB_KEY:
                wandb.save('./vae.pt')

            # temperature anneal

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # lr decay

            sched.step()

        if i % 10 == 0:
            lr = sched.get_last_lr()[0]
            print(epoch, i, f'lr - {lr:6f} loss - {loss.item()}')

            logs = {
                **logs,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item(),
                'lr': lr
            }
        if WANDB_KEY:
            wandb.log(logs)
        global_step += 1

    # save trained model to wandb as an artifact every epoch's end
    if WANDB_KEY:
        model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
        model_artifact.add_file('vae.pt')
        run.log_artifact(model_artifact)

# save final vae and cleanup

save_model('./vae-final.pt')
if WANDB_KEY:
    wandb.save('./vae-final.pt')

    model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
    model_artifact.add_file('vae-final.pt')
    run.log_artifact(model_artifact)

    wandb.finish()