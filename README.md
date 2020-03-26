# road-segmentation

This repository aims to provide a simple environment to easily train and test convolutional
neural network models for semantic segmentation task.

## Dependencies

The best way to set the environment up is to install Anaconda or Miniconda which
should manage the installation of CUDA for you.
After creating and activating conda virtual environment, install dependencies by running

```bash
$ pip install -r requirements.txt
```

## Usage

TODO

### Dataset setup

Script `data/split_data.py` does simple preprocessing and splits data into
`train`, `validation` and `test` folders.

Run `$ python data/split_data.py --help` to find out more.

### Training, Evaluating, Predicting

File `run.py` contains commands for training a model, evaluating it
on test set and creating predictions.

Repository contains many models defined in `models` folder and
imported in `run.py`.

In order to train a model, `run.py` contains command `train`.
```bash
$ python run.py train --help
```

Once the model is trained, we may evaluate it using `evaluate` command.
```bash
$ python run.py evaluate --help
```

If we want to predict test images, we may use `predict` command.
```bash
$ python run.py predict --help
```

The script also contains command `visualize` for visualizing
feature maps.
```bash
$ python run.py visualize --help
```


### Active learning

TODO
