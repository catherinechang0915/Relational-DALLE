import torch
from tqdm import tqdm
from torchvision.utils import save_image
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer
from pathlib import Path
from einops import repeat

class RelationalDalle(torch.nn.Module):
    def __init__(self, dalle_path='./trained_models/dalle.pt'):
        super().__init__()
        # load DALL-E
        dalle_path = Path(dalle_path)
        assert dalle_path.exists(), 'trained DALL-E must exist'
        load_obj = torch.load(str(dalle_path))
        dalle_params, vae_params, weights, vae_class_name, version = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights'), load_obj.pop('vae_class_name', None), load_obj.pop('version', None)
        vae = OpenAIDiscreteVAE()

        self.dalle = DALLE(vae = vae, **dalle_params).cuda()
        self.dalle.load_state_dict(weights)

    def generate_images(self, text, batch_size=64, output_dir_name='./outputs', num_images=3):
        text_tokens = tokenizer.tokenize([text], self.dalle.text_seq_len).cuda()
        text_tokens = repeat(text_tokens, '() n -> b n', b = num_images)

        outputs = []
        for text_chunk in tqdm(text_tokens.split(batch_size), desc = f'generating images for - {text}'):
            output = self.dalle.generate_images(text_chunk, filter_thres = 0.9)
            """
            Insert RN here.
            Pseudocode:
            rn_out = RN(output, text_check)
            if rn_out == 1: # image correctly represents text
                add to output                
            """
            outputs.append(output)

        file_name = text 
        output_dir = Path(output_dir_name) / file_name.replace(' ', '_')[:(100)]
        output_dir.mkdir(parents = True, exist_ok = True)
        outputs = torch.cat(outputs)

        for i, image in tqdm(enumerate(outputs), desc = 'saving images'):
            save_image(image, output_dir / f'{i}.png', normalize=True)
            with open(output_dir / 'caption.txt', 'w') as f:
                f.write(file_name)

    def forward(self, x):
        raise NotImplemented

if __name__ == '__main__':
    dalle = RelationalDalle()
    dalle.generate_images("A blue rectangle above a red circle.")