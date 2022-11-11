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
Modify the data directory to where data were saved, modify configuration in `TRAIN_CONFIG` in `config.py`
```
python train.py
```