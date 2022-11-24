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