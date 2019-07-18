#!/bin/bash
set -e
set -x

BASEURL=http://sterling8.d2.comp.nus.edu.sg/downloads/neuralreord

curl -L -o embeddings.tgz ${BASEURL}/embeddings.tar.gz

tar -xvzf embeddings.tgz
