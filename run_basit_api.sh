#!/bin/bash

# Python ortamına geçiş
source ~/miniconda3/bin/activate py311

# API'yi başlat
cd $(dirname "$0")/api
python basit_api.py
