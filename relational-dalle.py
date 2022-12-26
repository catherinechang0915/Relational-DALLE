import torch
import os
from torchvision.utils import save_image
from torchvision import transforms
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer
from pathlib import Path
from einops import repeat
from model import RN
from config import TRAIN_CONFIG, IMAGE_SIZE, DALLE_PATH, RN_PATH

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_CONFIG['cuda'] = (device == 'cuda')

class RelationalDalle(torch.nn.Module):
    def __init__(self, dalle_path=DALLE_PATH, rn_path=RN_PATH):
        super().__init__()
        # load DALL-E and RN
        dalle_path = Path(dalle_path)
        rn_path = Path(rn_path)
        assert dalle_path.exists(), 'trained DALL-E must exist'
        assert rn_path.exists(), 'trained RelationalNetwork must exist'
        dalle_obj = torch.load(str(dalle_path))
        dalle_params, vae_params, weights, vae_class_name, version = dalle_obj.pop('hparams'), dalle_obj.pop('vae_params'), dalle_obj.pop('weights'), dalle_obj.pop('vae_class_name', None), dalle_obj.pop('version', None)

        rn_obj = torch.load(str(rn_path))

        self.tokenizer = {'red': 1, 'green': 2, 'blue': 3, 'orange': 4, 'gray': 5, 'yellow': 6, 'circle': 7, 'rectangle': 8, 'a': 9, 'is': 10, 'of': 11, 'above': 12, 'below': 13, 'right': 14, 'left': 15}
        self.dalle = DALLE(**dalle_params).cuda()
        self.dalle.load_state_dict(weights)

        self.rn = RN(TRAIN_CONFIG).cuda()
        self.rn.load_state_dict(rn_obj)
        self.image_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))])


    def sentence_to_question(self, sentence):
        """
        Question encoding:
        0-5 correspond to o1 color
        6-7 correspond to o1 shape

        8-14 correspond to o2 color
        14-15 correspond to o2 shape

        15-19 correspond to q type

        [R, G, B, O, K, Y, circle, rectangle, R, G, B, O, K, Y, circle, rectangle, left, right, above, below]

        Sentence in format:
        A <o1 color> <o1 shape> is <q_type> <o2 color> <o2 shape>.

        1) Remove stop words
        2) Get o1 color, shape
        3) Get o2 color, shape
        4) Get q_type
        5) Encode
        """
        words = sentence.strip().replace('.', '').split(' ')

        index_to_color = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
        shapes = ['circle', 'rectangle']
        q_type_mapping = ['left', 'right', 'above', 'below']

        valid_words = index_to_color + q_type_mapping + shapes

        words = list(filter(lambda w: w in valid_words, words))

        o1_color_idx = index_to_color.index(words[0])
        o1_shape_idx = 6 if words[1] == 'circle' else 7

        o2_color_idx = index_to_color.index(words[3]) + 8
        o2_shape_idx = 14 if words[4] == 'circle' else 15

        q_type_idx = q_type_mapping.index(words[2])

        question = [0]*20
        question[o1_color_idx] = 1
        question[o1_shape_idx] = 1
        question[o2_color_idx] = 1
        question[o2_shape_idx] = 1
        question[16+q_type_idx] = 1
        return torch.tensor([question]).to(device)

    def generate_images(self, text, batch_size=64, output_dir_name='./outputs'):
        formatted_text = text.lower().replace('.', '')
        text_tokens = [self.tokenizer[i] for i in formatted_text.split()]
        text_tokens += [0] * (9-len(text_tokens))
        text_tokens = torch.tensor(text_tokens).unsqueeze(dim=0).cuda()
        text_tokens = repeat(text_tokens, '() n -> b n', b = 1)

        output = None
        bad_outputs = []
        question = self.sentence_to_question(text)
        while output is None:
            for text_chunk in text_tokens.split(batch_size):
                output_img = self.dalle.generate_images(text_chunk, filter_thres = 0.9)
                transformed_image = self.image_transform(output_img)
                # transformed image is (1, 3, 64, 64), question is (1, 20)
                rn_out = self.rn(transformed_image, question)
                is_correct = rn_out.data.max(1)[1].item()
                if is_correct:
                    output = output_img
                else:
                    bad_outputs.append(output_img)


        Path(output_dir_name).mkdir(parents = True, exist_ok = True)
        correct_dir_name = f'{output_dir_name}/correct'
        incorrect_dir_name = f'{output_dir_name}/incorrect'
        Path(correct_dir_name).mkdir(parents = True, exist_ok = True)
        Path(incorrect_dir_name).mkdir(parents = True, exist_ok = True)

        file_name = f"{text.replace(' ', '_')[:(100)]}.png"
        file_path = os.path.join(correct_dir_name, file_name)
        save_image(output, file_path, normalize=True)

        if len(bad_outputs):
            file_path = os.path.join(incorrect_dir_name, file_name)
            for i, image in enumerate(bad_outputs):
                save_image(image, file_path.replace('.png', f'_{i}.png'), normalize=True)

    def forward(self, x):
        raise NotImplemented

if __name__ == '__main__':
    dalle = RelationalDalle()
    f = open("dalle-test.txt", "r")
    lines = f.readlines()
    for i, l in enumerate(lines[116:]):
        if i%10 == 0:
            print(i, 'of', len(lines[116:]))
        dalle.generate_images(l.strip(), output_dir_name='../output')