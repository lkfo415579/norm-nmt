python ~/GOD.util/performance/competence/process_fasttext.py -i corpus.bpe.en -o en.emb -v ../BASE_2_REVO_en-ro/vocab.en.yml -w ~/fast/fasttext
python ~/GOD.util/performance/competence/build_cdf_mod.py --emb_vector en.emb.orig.vec corpus.bpe.en en-mod
