#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate py311
conda install -c conda-forge streamlit torch sentencepiece -y
pip install streamlit torch sentencepiece
python -c "import sentencepiece; print('Sentencepiece başarıyla yüklendi!')"
