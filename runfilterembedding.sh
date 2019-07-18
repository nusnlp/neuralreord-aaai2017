#!/bin/bash
export TOOLS_DIR=tools
set -e

PREP_DIR=$TOOLS_DIR/DependencyReordering
FILTER_W2V=$PREP_DIR/ioWrapper/filter_w2v_by_word.py

SRC_LANG=zh.jdp
CFG=$1

echo "Filtering word2vec embedding files - started:" `date`
EMBED_PRE=embeddings.${CFG}
EMBED_SUF=norm.bin

EMBEDDING_FILE=${EMBED_PRE}.${EMBED_SUF}

python $FILTER_W2V ${EMBEDDING_FILE} ${SRC_LANG}.vcb \
    ${EMBED_PRE}.npemb > ${EMBED_PRE}.vcb

echo Finished: `date`
