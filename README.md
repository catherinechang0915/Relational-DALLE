# RelationClassifier

## Installation
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Data Generation
Modify the sizes for dataset, image size, data directory in `config.py`
```
python generate_data.py
```

## Train
Modify configuration in `config.py`. Modify the data directory 
`TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` to where the dataset were saved. Modify `MODEL_DIR` to
where you want the best performance models to be saved. Training details are specified in `TRAIN_CONFIG`. 
Do not change `relation_type`.
```
python train.py
```

## Pretrained Models

Models have been pretrained for your convenience. They can be found at the links below:

[Pretrained DALL-E (30 epochs)](https://drive.google.com/file/d/1eQJOClc_70oaPTH3bxHkpAZx_ZN1Z5nO/view?usp=sharing)

This model was trained using the `dalle-pytorch` package which can be installed via pip: `$ pip install dalle-pytorch`

Please note that this package requires `pytorch` version 1.10 or below to function. Additionally note that this model used OpenAI's open source implementation of `DiscreteVAE` which this package also provides. The model was trained using the `image_text_folder` parameter like so:

`$ python train_dalle.py --image_text_folder /path/to/data`

Where we provided 150,000 image-text pairs. We used the default configuration provided in the package of Adam optimizer with a learning rate of `3e-4` and no scheduler.

[Pretrained RN (33 epochs)](https://drive.google.com/file/d/1kruA8lPV6uULFf7nD4h4M1ryf1JKHsqY/view?usp=sharing)

This model was trained using the Relational Network implementation below. It was trained on 150,000 image-text pairs with a learning rate of `1e-4` and no scheduler.

## References

[DALL-E Implementation](https://github.com/lucidrains/DALLE-pytorch)

[DALL-E Paper](https://arxiv.org/pdf/2102.12092.pdf)

[Relational Network Implementation](https://github.com/kimhc6028/relational-networks)

[Relational Network Paper](https://proceedings.neurips.cc/paper/2017/file/e6acf4b0f69f6f6e60e9a815938aa1ff-Paper.pdf)
