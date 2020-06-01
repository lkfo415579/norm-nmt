MARIAN=~/NBCL-marian/build
MARIAN_TRAIN=$MARIAN/marian
MARIAN_VOCAB=$MARIAN/marian-vocab

# set chosen gpus
GPUS="0 1 2"
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS

SRCL=en
TGTL=de
TERM=News
ID=2
MODEL_DIR=NBCL
TRAIN=corpus
VALID=newstest2013.bpe
CORPUS_DIR=data
OUTPUT_DIR=output

mkdir -p $MODEL_DIR output
# CBCL
# mkdir -p cl/rarity
# # python CL_tools/build_cdf_mod.py $CORPUS cl/rarity/en-rarity
# # NBCL
# mkdir -p cl/mod
# cat $TRAIN.$SRCL | $MARIAN_VOCAB > cl/vocab.$SRCL.yml
# python CL_tools/process_fasttext.py -i $TRAIN.$SRCL -o $SRCL.emb -v $cl/vocab.$SRCL.yml -w ~/fast/fasttext
# python CL_tools/build_cdf_mod.py --emb_vector $SRCL.emb.orig.vec $TRAIN.$SRCL cl/mod/$SRCL-mod
#
# mkdir -p $MODEL_DIR
# mkdir -p $OUTPUT_DIR

# create common vocabulary
# if [ ! -e $MODEL_DIR"/vocab."$SRCL$TGTL".yml" ]
# then
#     cat $DATA.$SRCL $DATA.$TGTL | $MARIAN_VOCAB --max-size 66000 > $MODEL_DIR/vocab.$SRCL$TGTL.yml
# fi
# train model
    $MARIAN_TRAIN \
        --model $MODEL_DIR/model_revo.npz --type transformer \
        --train-sets $CORPUS_DIR/$TRAIN.$SRCL $CORPUS_DIR/$TRAIN.$TGTL \
        --max-length 140 \
        --vocabs $MODEL_DIR/vocab.$SRCL.yml $MODEL_DIR/vocab.$TGTL.yml \
        --mini-batch-fit -w 9250 --maxi-batch 5000 \
        --early-stopping 10 --cost-type=ce-mean-words \
        --valid-freq 2500 --save-freq 2500 --disp-freq 1 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $CORPUS_DIR/$VALID.$SRCL $CORPUS_DIR/$VALID.$TGTL \
        --valid-script-path "bash ./validate-"$SRCL\-$TGTL".sh" \
        --valid-translation-output $OUTPUT_DIR/$ID.tf.$SRCL$TGTL.single --quiet-translation \
        --valid-mini-batch 64 \
        --beam-size 6 --normalize 0.6 \
        --log $MODEL_DIR/train.log --valid-log $MODEL_DIR/valid.log \
        --enc-depth 6 --dec-depth 6 \
        --transformer-heads 8 \
        --transformer-postprocess-emb d \
        --transformer-postprocess dan \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
        --devices $GPUS --sync-sgd --seed $ID$ID$ID$ID --keep-best --overwrite \
        --exponential-smoothing --no-nccl --filter_corpus 0.85 \
        --sr-freq-file cl/mod/en-mod.txt cl/mod/en-mod-cdf_base.npz 2.5 0.01 mod d 0.5 \
        --after-batches 100000

./decode_validate.sh $MODEL_DIR > $MODEL_DIR/result

# NOTES:
# CBCL : --sr-freq-file cl/rarity/en-freq.txt cl/rarity/en-cdf_base.npz 25000 0.01 cl d
# NBCL : --sr-freq-file cl/mod/en-mod.txt cl/mod/en-mod-cdf_base.npz 2.5 0.01 mod d 0.5
# params : word_stat_file CDF_file [CL:step, MOD: ratio] starting_corpus_percentage mode(cl|mod) [dynamic_weight] [dynamic_ratio]
