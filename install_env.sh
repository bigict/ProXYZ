#!/bin/bash
#
set -e

name="xyz"

help() {
  echo "usage: `basename $0` [-h] -n {name}"
  echo "options:"
  echo "    -h, --help show this help message and exit"
  echo "    -n ENVIRONMENT, --name ENVIRONMENT"
  echo "               Name of environment. (default: ${name})"
  exit $1
}

ARGS=$(getopt -o "n:h" -l "name:,help" -- "$@") || help 1
eval "set -- ${ARGS}"
while true; do
  case "$1" in
    (-n | --name) name="$2"; shift 2;;
    (-h | --help) help 0 ;;
    (--) shift 1; break;;
    (*) help 1;
  esac
done

conda create -n ${name} python=3.11 gxx=11.2.0 -c conda-forge

conda run -n ${name} \
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
conda install -n ${name} cuda-nvcc==12.8.93 -c nvidia -c conda-forge
conda run -n ${name} pip install psutil
conda run -n ${name} \
    pip install flash-attn==2.8.3.post1 --no-build-isolation
conda run -n ${name} \
    pip install torch_geometric==2.8.0.post1 -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
conda run -n ${name} \
    pip install accelerate==1.14.0 datasets==5.0.0 tokenizers==0.22.2 transformers==5.12.0
conda run -n ${name} \
    pip install graphein==1.7.8 biotite==1.6.0 foldcomp==1.0.0 lightning==2.6.5
conda run -n ${name} \
    pip install biglist==0.9.6 lmdb==2.3.0 posix-ipc==1.3.2
