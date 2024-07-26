## Installation

Make sure to use python 3.9, platform: linux-64:
```
$ conda create -n DevMut python=3.9
$ pip install -r requirements.txt
$ conda activate DevMut
```

## Usage
# 1
```
export CONTEXT_DEVICE_TARGET=GPU
export CUDA_VISIBLE_DEVICES=2,3
```
# 2
```
python mutation_test.py
```
# 3 output 
./common/log/