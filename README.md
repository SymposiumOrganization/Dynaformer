# Dynaformer: A Deep Learning Model for Ageing-aware Battery Discharge Prediction
Implementation, data and pretrained models for the paper "Dynaformer: A Deep Learning Model for Ageing-aware Battery Discharge Prediction"

## How to get started
* Clone the repository: `git clone`
* Create a virtual environment: `python -m venv venv` # Note, we used python 3.6 for this project on Ubuntu 18.04
* Activate the virtual environment: `source venv/bin/activate`
* Install dependencies: `pip install -r requirements.txt`
* Install the dynaformer package: `pip3 install -e src/`
* Download the model/data `python scripts/download_model_and_sample_data.py` # It will download the weights of Dynaformer on synthetic data and the test data


## Demo
* Visualize the deme of the model (inference) via `streamlit run visualization/visualize_predictions.py`

## Getting the synthetic data 
### Generate a synthetic training dataset
In order to generate a synthetic training dataset via `python3 scripts/generate_dataset.py`. We use [Hydra](https://github.com/facebookresearch/hydra) as configuration tool. The configuration file is `configs/generate_dataset.yaml`. 
In general, if you want to generate **synthetic constant training dataset** similar the one in the paragraph of the paper "Performance Evaluation on Constant Load Profiles" you can use the following command:
```
python3 scripts/generate_dataset.py current.current_type=constant_currents N_profiles=1 N_currents=50
```
Instead if you want to generate a **synthetic variable training dataset**, similar the one in the paragraph of the paper "Performance Evaluation on Variable Load Profiles" you can use the following command:
```
python3 scripts/generate_dataset.py
```
Please take a look at the configuration file if you want to modify something more in specific.


### Download the synthetic training dataset
Alternatively you directly download our variable training dataset via `python scripts/download_training_data.py --dataset_type variable`.

You can change the parameters in config/generate_dataset.yaml or via command line. 
You can also change current.current_type to `constant_currents` to generate a constant current dataset.
For instance,  `python3 scripts/generate_dataset.py`

The generated dataset is saved in `data/variable_currents` or `data/constant_currents`.

## Training
### How to train the model
* Train the Dynaformer model via the following command:
```
python3 scripts/train.py method=dynaformer data_dir=data/variable_currents/2022-04-27/14-58-12/data method.batch_size=12
```
If you want to train the model on a different dataset, you can change the `data_dir` parameter. 

## TODO
* [X] Add a demo of the model
* [ ] Add a demo of the model on real data
* [X] Add training dataset
* [ ] Add training/testing dataset generation
* [X] Add training pipeline
* [ ] Add baseline models

## Additional information
### Pretrained models and data 
Please note that these are downloaded already with the script `download_data.py`. 
* Dynaformer `https://drive.google.com/open?id=1-_QZQ-_j_X8W_Xq_X_X_X_X_X_X`
* Synthetic data `https://drive.google.com/open?id=1-_QZQ-_j_X8W_Xq_X_X_X_X_X_X`

### System Specification
All the experiments were done with Python 3.6 with pytorch 1.9.0+cu111 on Ubuntu 18.04.
