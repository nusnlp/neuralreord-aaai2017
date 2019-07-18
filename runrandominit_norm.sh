#!/bin/bash
export TOOLS_DIR=tools
set -e

W2V_PREP_DIR=$TOOLS_DIR/DependencyReordering

SRC_LANG=zh.jdp
VECSIZE=100
ALPHA=0.1

python ${W2V_PREP_DIR}/ioWrapper/w2v_random_norm.py ${SRC_LANG}.vcb ${VECSIZE} embeddings.random${ALPHA}.npemb ${ALPHA}

cp ${SRC_LANG}.vcb embeddings.random${ALPHA}.vcb
