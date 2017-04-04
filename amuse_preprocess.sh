#!/usr/bin/env bash

while read p; do
  echo "Processing $p"
  python  amuse_preprocess.py --subject $p
done <amuse_datasets
