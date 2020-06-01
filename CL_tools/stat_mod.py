# -*- encoding=utf-8 -*-
import numpy as np
import codecs
import sys

print "python .py vocab.txt iter*.npz"

out = codecs.open('output', 'w', encoding='utf-8')

vocab = codecs.open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').readlines()
vocab = [s.split(":")[0] for s in vocab]

both = []
for i in range(2, len(sys.argv)):
    name = sys.argv[i]
    iter = name.split(".")[1][4:]
    both.append((name, iter))
both.sort(key=lambda x: int(x[1]))
print both
models_names = [x[0] for x in both]
models_iters = [x[1] for x in both]
# loading emb
vocab_stats = [[] for _ in range(len(vocab))]
for name in models_names:
    model = np.load(name)
    # encoder_Wemb
    print "=" * 100
    Wemb = model['encoder_Wemb']
    print Wemb, "===>", Wemb.shape, "===>", name
    vocab_id = 0
    for word in Wemb:
        t = [x * x for x in word]
        mod = np.sum(t) ** 0.5
        vocab_stats[vocab_id].append(mod)
        vocab_id += 1
    # print vocab_stats

# output stats
out.write("ID:" + ",".join(models_iters) + '\n')
for id, v_data in enumerate(vocab_stats):
    out.write("%s:%s\n" % (vocab[id], str(v_data)))
