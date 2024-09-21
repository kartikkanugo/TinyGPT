# VectoCalibration

Module for calibrating the probe, it reads the data from the csv file and produces the suitable cal files.

## User Installation

The released .whl files can be found in Nextcloud release folder "https://app.vectoflow.de/cloud/index.php/f/1802895"
The package can be installed using the following command
`pip install <path to .whl file>`
All dependencies will automatically install

## Release Procedure

Run the following command in your environment `python -m build --wheel`

## Debug Procedure

The package requires vectoErrors to be installed from the .whl file.
Run the following command in your environment `python install -e .`
