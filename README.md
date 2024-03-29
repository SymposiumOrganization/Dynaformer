# Dynaformer: A Deep Learning Model for Ageing-aware Battery Discharge Prediction
Implementation, data and pretrained models for the paper "Dynaformer: A Deep Learning Model for Ageing-aware Battery Discharge Prediction"

## How to get started
* Clone the repository: `git clone`
* Create a virtual environment: `python -m venv venv` # Note, we used python 3.6 for this project on Ubuntu 18.04
* Activate the virtual environment: `source venv/bin/activate`
* Install dependencies: `pip install -r requirements.txt`
* Install the dynaformer package: `pip3 install -e src/`
* Download the model and the test data `python scripts/download_model_and_test_data.py` # This will download the weights of Dynaformer, and the synethetic variable test data


## Demo
* Visualize the deme of the model (inference) via `streamlit run visualization/visualize_predictions.py`

## Getting the synthetic data 
#### Generate a synthetic training dataset
In order to generate a synthetic training dataset via `python3 scripts/generate_dataset.py`. We use [Hydra](https://github.com/facebookresearch/hydra) as configuration tool. The configuration file is `configs/generate_dataset.yaml`. 
In general, if you want to generate **synthetic constant training dataset** similar the one in the paragraph of the paper "Performance Evaluation on Constant Load Profiles" you can use the following command:
```
python3 scripts/generate_dataset.py current.current_type=constant_currents current.N_profiles=1 N_currents=50
```
Instead if you want to generate a **synthetic variable training dataset**, similar the one in the paragraph of the paper "Performance Evaluation on Variable Load Profiles" you can use the following command:
```
python3 scripts/generate_dataset.py current.current_type=variable_currents current.N_profiles=6 N_currents=1000
```
Please take a look at the configuration file if you want to modify something more in specific.


The generated dataset is saved in `data/variable_currents` or `data/constant_currents` depending on the option current.current_type

## Getting the real data


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
### System Specification
All the experiments were done with Python 3.6 with pytorch 1.9.0+cu111 on Ubuntu 18.04.
