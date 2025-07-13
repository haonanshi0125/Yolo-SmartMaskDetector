## 基于YOLO&Pyside6的智能口罩检测系统

## Project Directory Initialization

```shell
python init_project.py
```

## Data Preparation Steps

Place the dataset in the following directories:
   - `./yolo_server/data/raw/images`
   - `./yolo_server/data/raw/original_annotations`

Run the script to split data:

```shell
python yolo_trans.py
```

Run the script to validate data:

```shell
python yolo_validate.py
```

## Model Preparation Step

Place the pretrained model in the following directory:
   - `./yolo_server/models/pretrained`

## Training, Evaluation, Infer Steps

We utilize 1 A100 GPUs for training.

Train:

```shell
python yolo_train.py
```

Val:

```shell
python yolo_model_val.py
```

Infer:

```shell
python yolo_infer.py
```

## UI System Application

```shell
python main.py
```
