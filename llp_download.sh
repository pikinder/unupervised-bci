#!/usr/bin/env bash

curl -# -o storage/raw_data/llp_1.zip https://zenodo.org/record/192684/files/online_study_1-7.zip
curl -# -o storage/raw_data/llp_2.zip https://zenodo.org/record/192684/files/online_study_8-13.zip

unzip storage/raw_data/llp_1.zip
unzip storage/raw_data/llp_2.zip