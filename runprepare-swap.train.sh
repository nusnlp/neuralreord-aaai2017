#!/bin/bash
export TOOLS_DIR=tools
set -e

PREP_DIR=$TOOLS_DIR/DependencyReordering
TRAIN_PREP=$PREP_DIR/generateSwapTrainset-pairwithpos-withvcb.py
RAW2IDX=$PREP_DIR/raw2idx-trainset.py
NUMPIZE=$PREP_DIR/numpizeTrainset.py

SRC_LANG=zh.jdp

###################################
###   Prepare the corpus data   ###
###################################

WORD_VOCAB=wordvcb.${SRC_LANG%.jdp}

TRAIN_FILE=train.${SRC_LANG}
ALIGN_FILE=train.align
SIB_PREFIX=swaptrain.sib
PAR_PREFIX=swaptrain.par

echo Preparing W2V training data $L - started: `date`

time python ${TRAIN_PREP} ${WORD_VOCAB} ${TRAIN_FILE} ${ALIGN_FILE} ${PAR_PREFIX}.${SRC_LANG}.gz ${SIB_PREFIX}.${SRC_LANG}.gz

ITER=10
VECSIZE=100
SAMPSIZE=0
WINDOW=4
MC=4
ALPHA=0.1

VOCAB=${SRC_LANG}.vcb

echo Converting swap training data to feature indices - started: `date`

DIM=12  # the number of features in the swap classifier training data
for P in $PAR_PREFIX $SIB_PREFIX; do

python ${RAW2IDX} ${VOCAB} ${P}.${SRC_LANG}.gz ${P}.num.${SRC_LANG}.gz > ${P}.${SRC_LANG}.size
zcat ${P}.num.$SRC_LANG.gz | python ${NUMPIZE} `cat ${P}.${SRC_LANG}.size` ${DIM} ${P}.${SRC_LANG}.mmap

done

echo Finished: `date`
