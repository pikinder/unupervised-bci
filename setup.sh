#!/usr/bin/env

# Download amuse data
while read p; do
  echo "Downloading $p"
  curl -# -o "storage/raw_data/$p.mat" "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-AMUSE/AMUSE_$p.mat"
done <amuse_datasets

# Preprocess amuse data
while read p; do
  echo "Processing $p"
  python  amuse_preprocess.py --subject $p
done <amuse_datasets

# Download and unzip the LLP data
curl -# -o storage/raw_data/llp_1.zip https://zenodo.org/record/192684/files/online_study_1-7.zip
curl -# -o storage/raw_data/llp_2.zip https://zenodo.org/record/192684/files/online_study_8-13.zip

unzip storage/raw_data/llp_1.zip -d storage/raw_data
unzip storage/raw_data/llp_2.zip -d storage/raw_data

# Preprocess the llp data