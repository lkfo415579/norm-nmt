testset=data/newstest2014.bpe.en
testset_ref=data/newstest2014.tc.de
cat $testset | build/marian-decoder -m $1/model_revo.npz.best-translation.npz -v \
$1/vocab.en.yml $1/vocab.de.yml -b 6 -n 0.6 --mini-batch 100 -d 6 7 -o output.txt



cat output.txt | sed 's/@@ //g' \
    | ~/GOD.util/moses-scripts/scripts/generic/multi-bleu.perl -lc $testset_ref
#     | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
