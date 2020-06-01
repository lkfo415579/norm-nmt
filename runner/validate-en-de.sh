#!/bin/bash

SRCL=en
TGTL=de
TERM=News
VALID=data/newstest2013.tc.$TGTL

cat $1 | sed 's/@@ //g' \
    | ~/NBCL-marian/runner/multi-bleu.perl -lc $VALID \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
