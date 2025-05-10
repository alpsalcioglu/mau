#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate py311
cd /Users/alpsalcioglu/Desktop/PythonProject
echo "Python 3.11 ortamında Streamlit uygulaması başlatılıyor..."
streamlit run app.py
