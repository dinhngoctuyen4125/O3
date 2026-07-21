#!/bin/bash

python preprocess_SQ.py
python preprocess_scienceqa.py
python preprocess_scienceqa_RD.py
python preprocess_scienceqa_SD.py
python preprocess_scienceqa_random_labeling.py