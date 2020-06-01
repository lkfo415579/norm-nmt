import sys
import math

print("usage: x.py norm_f freq_f source_bpe_corpus")
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
# search = "&#124;"
len_data, cl_data, norm_data = [], [], []
for i, line in enumerate(corpus):
    # if i % 5000 == 0:
    #     print('.', end='')
    line = line.strip()
    len_score = float(len(line))
    cl_score = cal_score('cl', line, freq_vocab)
    norm_score = cal_score('norm', line, norm_vocab)
    len_data.append((len_score, line, i))
    cl_data.append((cl_score, line, i))
    norm_data.append((norm_score, line, i))

len_data.sort(key=lambda x: x[0])
cl_data.sort(key=lambda x: x[0])
norm_data.sort(key=lambda x: x[0])
print(len_data[:10])
# READ TESTSET
BASE = open('BASE.output', 'r').readlines()
CL = open('CL-sw.output', 'r').readlines()
NORM = open('NORM.output', 'r').readlines()
TEST = open('newstest2014.tc.de', 'r').readlines()
SRC = open('newstest2014.tc.en', 'r').readlines()

def generate(data, INPUT, name):
  r = 0
  for i in range(1, 4):
    portion = int(len(corpus)/3)
    tmp = data[r:i * portion]
    r = i * portion
    o_f = open('workshop/%s%d.txt' % (name, i), 'w')
    t_f = open('workshop/t%d.txt' % i, 'w')
    s_f = open('workshop/src%d.txt' % i, 'w')
    for l in tmp:
      score, line, id = l
      o_f.write(INPUT[id])
      t_f.write(TEST[id])
      s_f.write(SRC[id])

if sys.argv[4] == '0':
    # LEN DATA
    print("LENGTH_BASED")
    generate(len_data, BASE, 'base')
    generate(len_data, CL, 'cl')
    generate(len_data, NORM, 'norm')
elif sys.argv[4] == '1':
    # CL DATA
    print("CL_BASED")
    generate(cl_data, BASE, 'base')
    generate(cl_data, CL, 'cl')
    generate(cl_data, NORM, 'norm')
elif sys.argv[4] == '2':
    # NORM DATA
    print("NORM_BASED")
    generate(norm_data, BASE, 'base')
    generate(norm_data, CL, 'cl')
    generate(norm_data, NORM, 'norm')



