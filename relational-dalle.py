import torch
from torchvision.utils import save_image
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer
from pathlib import Path
from einops import repeat
from model import RN
from config import TRAIN_CONFIG

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_CONFIG['cuda'] = (device == 'cuda')

class RelationalDalle(torch.nn.Module):
    def __init__(self, dalle_path='./trained_models/dalle.pth', rn_path='./trained_models/rn.pth'):
        super().__init__()
        # load DALL-E and RN
        dalle_path = Path(dalle_path)
        rn_path = Path(rn_path)
        assert dalle_path.exists(), 'trained DALL-E must exist'
        assert rn_path.exists(), 'trained RelationalNetwork must exist'
        dalle_obj = torch.load(str(dalle_path))
        dalle_params, vae_params, weights, vae_class_name, version = dalle_obj.pop('hparams'), dalle_obj.pop('vae_params'), dalle_obj.pop('weights'), dalle_obj.pop('vae_class_name', None), dalle_obj.pop('version', None)
        vae = OpenAIDiscreteVAE()

        rn_obj = torch.load(str(rn_path))

        self.dalle = DALLE(vae = vae, **dalle_params).cuda()
        self.dalle.load_state_dict(weights)

        self.rn = RN(TRAIN_CONFIG).cuda()
        self.rn.load_state_dict(rn_obj)

    def sentence_to_question(self, sentence):
        """
        Question encoding:
        0-5 correspond to o1 color
        6-7 correspond to o1 shape

        8-14 correspond to o2 color
        14-15 correspond to o2 shape
        [R, G, B, O, K, Y, circle, rectangle, R, G, B, O, K, Y, circle, rectangle]

        Sentence in format:
        A <o1 color> <o1 shape> is above <o2 color> <o2 shape>.
        """
        index_to_color = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
        words = sentence.strip().replace('.', '').split(' ')
        o1_color_idx = index_to_color.index(words[1])
        o1_shape_idx = 6 if words[2] == 'circle' else 7
        o2_color_idx = index_to_color.index(words[6]) + 8
        o2_shape_idx = 14 if words[6] == 'circle' else 15

        question = [0]*16
        question[o1_color_idx] = 1
        question[o1_shape_idx] = 1
        question[o2_color_idx] = 1
        question[o2_shape_idx] = 1
        return question

    def generate_images(self, text, batch_size=64, output_dir_name='./outputs', num_images=3):
        text_tokens = tokenizer.tokenize([text], self.dalle.text_seq_len).cuda()
        text_tokens = repeat(text_tokens, '() n -> b n', b = num_images)

        outputs = []
        question = self.sentence_to_question(text)
        for text_chunk in text_tokens.split(batch_size):
            output_img = self.dalle.generate_images(text_chunk, filter_thres = 0.9)
            rn_out = self.rn(output_img, question)
            """
            Insert RN here.
            Pseudocode:
            rn_out = RN(output, text_check)
            if rn_out == 1: # image correctly represents text
                add to output                
            """
            outputs.append(output)

        Path(output_dir_name).mkdir(parents = True, exist_ok = True)
        file_name = f"{text.replace(' ', '_')[:(100)]}.png"
        file_path = f"{output_dir_name}/{file_name}"
        outputs = torch.cat(outputs)

        for i, image in enumerate(outputs):
            save_image(image, file_path, normalize=True)

    def forward(self, x):
        raise NotImplemented

if __name__ == '__main__':
    dalle = RelationalDalle()
    sentence = "A green circle is above a red rectangle."
    dalle.generate_images(lines[0], num_images=1)
