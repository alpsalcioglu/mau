#!/bin/bash

# Python ortamına geçiş
source ~/miniconda3/bin/activate py311

# Doğru dizine geç
cd $(dirname "$0")

# API'yi başlat
python api/original_model_api.py
