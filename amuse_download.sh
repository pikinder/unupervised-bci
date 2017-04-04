#!/usr/bin/env bash

while read p; do
  echo "Downloading $p"
  curl -# -o "storage/raw_data/$p.mat" "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-AMUSE/AMUSE_$p.mat"
done <amuse_datasets