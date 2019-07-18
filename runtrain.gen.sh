#!/bin/bash
export TOOLS_DIR=tools
set -x
set -e

PREP_DIR=$TOOLS_DIR/DependencyReordering
TRAINER=$PREP_DIR/nnAdapt/train_features-dropout.withmom.py # progressive learning rate update

# Data
TRAIN=train
DEV=dev
SRC_LANG=zh.jdp

GEN_PREFIX=swaptrain
VOCAB=${SRC_LANG}.vcb

PAR_TRAIN_PREFIX=swaptrain.par
PAR_DEV_PREFIX=swapdev.par
SIB_TRAIN_PREFIX=swaptrain.sib
SIB_DEV_PREFIX=swapdev.sib

###################################
###   Prepare the corpus data   ###
###################################
CFG=dep
ILR=5  # inverse learning rate = 1/LR
LR=`echo 1/${ILR} | bc -l`

MODE=$W2V_MODE
# Model hyperparameters
VECSIZE=100
HIDDENS=200,100
DROPOUTS=0.5,0,0  # embedding layer dropout, hidden layer 1 dropout, hidden layer 2 dropout
W2V_PREFIX=embeddings.${CFG}

VCB_FILE=${W2V_PREFIX}.vcb
W2V_FILE=${W2V_PREFIX}.npemb.mat.mmap

PAR_BATCH=128
SIB_BATCH=128
EPOCHS=100

export OMP_NUM_THREADS=1
M1=$1  # GPU machine number to train head-child (par) model
M2=$2  # GPU machine number to train siblings (sib) model
SEED=$3

MODEL_DIR=models/pair-${CFG}-v${VECSIZE}-H${HIDDENS}-d${DROPOUTS}-ilr${ILR}-rs${SEED}
PAR_MODEL=$MODEL_DIR/model-par
SIB_MODEL=$MODEL_DIR/model-sib


if [ ! -e ${PAR_TRAIN_PREFIX}.${SRC_LANG}.mmap ]; then
    ${PREP_DIR}/raw2idx-trainset.py ${VOCAB} ${PAR_TRAIN_PREFIX}.${SRC_LANG}.gz ${PAR_TRAIN_PREFIX}.num.${SRC_LANG}.gz > ${PAR_TRAIN_PREFIX}.size
    zcat ${PAR_TRAIN_PREFIX}.num.${SRC_LANG}.gz | python ${PREP_DIR}/numpizeTrainset.py `cat ${PAR_TRAIN_PREFIX}.size` 12 ${PAR_TRAIN_PREFIX}.${SRC_LANG}.mmap
fi

if [ ! -e ${SIB_TRAIN_PREFIX}.${SRC_LANG}.mmap ]; then
    ${PREP_DIR}/raw2idx-trainset.py ${VOCAB} ${SIB_TRAIN_PREFIX}.${SRC_LANG}.gz ${SIB_TRAIN_PREFIX}.num.${SRC_LANG}.gz > ${SIB_TRAIN_PREFIX}.size
    zcat ${SIB_TRAIN_PREFIX}.num.${SRC_LANG}.gz | python ${PREP_DIR}/numpizeTrainset.py `cat ${SIB_TRAIN_PREFIX}.size` 12 ${SIB_TRAIN_PREFIX}.${SRC_LANG}.mmap
fi

if [ ! -e ${PAR_DEV_PREFIX}.${SRC_LANG}.mmap ]; then
    ${PREP_DIR}/raw2idx-trainset.py ${VOCAB} ${PAR_DEV_PREFIX}.${SRC_LANG}.gz ${PAR_DEV_PREFIX}.num.${SRC_LANG}.gz > ${PAR_DEV_PREFIX}.size
    zcat ${PAR_DEV_PREFIX}.num.${SRC_LANG}.gz | python ${PREP_DIR}/numpizeTrainset.py `cat ${PAR_DEV_PREFIX}.size` 12 ${PAR_DEV_PREFIX}.${SRC_LANG}.mmap
fi

if [ ! -e ${SIB_DEV_PREFIX}.${SRC_LANG}.mmap ]; then
    ${PREP_DIR}/raw2idx-trainset.py ${VOCAB} ${SIB_DEV_PREFIX}.${SRC_LANG}.gz ${SIB_DEV_PREFIX}.num.${SRC_LANG}.gz > ${SIB_DEV_PREFIX}.size
    zcat ${SIB_DEV_PREFIX}.num.${SRC_LANG}.gz | python ${PREP_DIR}/numpizeTrainset.py `cat ${SIB_DEV_PREFIX}.size` 12 ${SIB_DEV_PREFIX}.${SRC_LANG}.mmap
fi

mkdir -p ${MODEL_DIR}

echo Training back-propagation - started: `date`
source activate py27-tf12
export THEANO_FLAGS=base_compiledir=/tmp/christian/theano.NOBACKUP,mode=FAST_RUN,floatX=float32,on_unused_input=warn,deterministic=more,device=cuda${M1}
python ${TRAINER} -vcb ${VCB_FILE} -emb ${W2V_FILE} -tr ${PAR_TRAIN_PREFIX}.${SRC_LANG}.mmap -dev ${PAR_DEV_PREFIX}.${SRC_LANG}.mmap \
    -out ${PAR_MODEL} -lr ${LR} -H ${HIDDENS} -mb ${PAR_BATCH} -do ${DROPOUTS} -E ${EPOCHS} -rs ${SEED} &> ${MODEL_DIR}/trace-par.log &

export THEANO_FLAGS=base_compiledir=/tmp/christian/theano.NOBACKUP,mode=FAST_RUN,floatX=float32,on_unused_input=warn,deterministic=more,device=cuda${M2}
python ${TRAINER} -vcb ${VCB_FILE} -emb ${W2V_FILE} -tr ${SIB_TRAIN_PREFIX}.${SRC_LANG}.mmap -dev ${SIB_DEV_PREFIX}.${SRC_LANG}.mmap \
    -out ${SIB_MODEL} -lr ${LR} -H ${HIDDENS} -mb ${SIB_BATCH} -do ${DROPOUTS} -E ${EPOCHS} -rs ${SEED} &> ${MODEL_DIR}/trace-sib.log &

wait
source deactivate
echo Training back-propagation - finished: `date`
