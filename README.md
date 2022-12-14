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

[Pretrained RN (33 epochs)](https://drive.google.com/file/d/1kruA8lPV6uULFf7nD4h4M1ryf1JKHsqY/view?usp=sharing)

## Reference

Relational Network logic: https://github.com/kimhc6028/relational-networks
