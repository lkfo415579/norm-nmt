#!/bin/bash -v

MARIAN=~/competence/build

MARIAN_TRAIN=$MARIAN/marian
MARIAN_DECODER=$MARIAN/marian-decoder
MARIAN_VOCAB=$MARIAN/marian-vocab
MARIAN_SCORER=$MARIAN/marian-scorer

# set chosen gpus
GPUS="0 1 2 3 4 5 6 7"
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

SRCL=zh
TGTL=en
TERM=Common
MODEL_NAME=BASE_DEEP
ID=2
MODEL_DIR=$MODEL_NAME\_$SRCL\-$TGTL
TRAIN=corpus.bpe
VALID=../dev/2017.bpe
CORPUS_DIR=en-zh
OUTPUT_DIR=output

mkdir -p $MODEL_DIR
mkdir -p $OUTPUT_DIR

# create shared vocabulary
# if [ ! -e $MODEL_DIR"/vocab."$SRCL$TGTL".yml" ]
# then
#     cat $CORPUS_DIR/$TRAIN.$SRCL $CORPUS_DIR/$TRAIN.$TGTL | $MARIAN_VOCAB --max-size 66000 > $MODEL_DIR/vocab.$SRCL$TGTL.yml
# fi

# train model
    $MARIAN_TRAIN \
        --model $MODEL_DIR/model_revo.npz --type transformer \
        --train-sets $CORPUS_DIR/$TRAIN.$SRCL $CORPUS_DIR/$TRAIN.$TGTL \
        --max-length 140 \
        --vocabs $MODEL_DIR/vocab.$SRCL.yml $MODEL_DIR/vocab.$TGTL.yml \
        --mini-batch-fit -w 9050 --maxi-batch 2000 \
        --early-stopping 10 --cost-type=ce-mean-words \
        --valid-freq 2500 --save-freq 2500 --disp-freq 1 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $CORPUS_DIR/$VALID.$SRCL $CORPUS_DIR/$VALID.$TGTL \
        --valid-script-path "bash ./validate-"$SRCL\-$TGTL".sh" \
        --valid-translation-output $OUTPUT_DIR/$MODEL_NAME.tf.$SRCL$TGTL.single --quiet-translation \
        --valid-mini-batch 30 \
        --beam-size 12 --normalize 1.0 \
        --log $MODEL_DIR/train.log --valid-log $MODEL_DIR/valid.log \
        --enc-depth 12 --dec-depth 12 \
        --transformer-heads 8 \
        --transformer-postprocess-emb d \
        --transformer-postprocess dan \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0003 --lr-warmup 8000 --lr-decay-inv-sqrt 8000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 \
        --devices $GPUS --sync-sgd --seed $ID$ID$ID$ID --keep-best --overwrite \
        --exponential-smoothing --after-batches 150000 --print_mod --disp-label-counts \
        --update_cycle 1 --ds-init
        # --exponential-smoothing --after-batches 120000
        # --sr-freq-file cl/$SRCL\-freq.txt cl/$SRCL\-cdf_base.npz 40000 0.01

./final_exam.zhen.sh dev/newstest2017-zhen-src.pre.zh.bpe $MODEL_DIR "$GPUS"
