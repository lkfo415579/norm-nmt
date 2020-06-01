import sys
import math

print("usage: x.py norm_f freq_f corpus")
norm_f = open(sys.argv[1], 'r').readlines()
freq_f = open(sys.argv[2], 'r').readlines()
corpus = open(sys.argv[3], 'r').readlines()
print("Loaded", len(corpus))


def readV(f, t=""):
    vocab = {}
    for i, l in enumerate(f):
        word, score = l.split()
        if t == 'norm':
            i = len(f) - i
        vocab[word] = (float(score), i)
    return vocab


norm_vocab = readV(norm_f, "norm")
freq_vocab = readV(freq_f)

# print all common vocab
# for v in norm_vocab:
#   print(v, norm_vocab[v][0], math.log(freq_vocab[v][0]), "NORM_ID:%d" % norm_vocab[v][1], "FREQ_ID:%d" % freq_vocab[v][1])


def cal_score(t, line, vocab):
    line = line.split()
    SUM = 0.
    for word in line:
        p = vocab[word][0]
        if t == 'cl':
            SUM += math.log(p)
        else:
            SUM += p

    if t == 'cl':
        SUM = -1 * SUM
    return SUM


# sys.exit()
# main
search = "&#124;"
for i, line in enumerate(corpus):
    # if i % 5000 == 0:
    #     print('.', end='')
    line = line.strip()
    cl_score = cal_score('cl', line, freq_vocab)
    norm_score = cal_score('norm', line, norm_vocab)
    # if line.find(search) != -1 and norm_score > cl_score * 3:
    # norm_score is easy but cl_score is difficult
    # if norm_score * 2.8 < cl_score:
    # norm_score is difficult but cl_score is easy
    # if norm_score > cl_score * 0.5 and norm_score < 100:
    # find errors sentences
    if norm_score > 800:
        print("%d ||| %s ||| CL: %f ||| MOD: %f" % (i, line, cl_score, norm_score))
