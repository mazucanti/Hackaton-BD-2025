# Hackaton-BD-2025

This is the repository to generate the output of the banana_terracota.py team

The code was made in Python 3.12.1 but it is compatible with newer versions of Python.

## Requirements

To install the requirements `pip install -r requirements.txt`.

Alternatively, one can run `pip-compile --cert=None --client-cert=None --index-url=None --output-file=requirements.txt --pip-args=None requirements.in` to use the `pip-tools` package version manager. That method need `pip-tools` to be installed with `pip install pip-tools`.

## Running the model

To run the model, first put the three parquet files in the `data` folder, then you need to execute `python3 -m banana_terracota`. After completion, check the folder `output` to se the file `result.csv` which contains the result of the model. 

The model is already trained and saved to file, but you can train your own model using `<AAAAAAAAAAAAAAAAAAAAAAAAAAA>`