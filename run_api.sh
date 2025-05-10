#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate py311
cd /Users/alpsalcioglu/Desktop/PythonProject/api
pip install flask flask-cors
python app.py
