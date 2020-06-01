import numpy as np
import sys

cdf = 'de-rarity-cdf_base.npz'
rarity_t = open('de-rarity.txt', 'r').readlines()
rarity = {}
for l in rarity_t:
    word, pos = l.split()
    rarity[word] = float(pos)

base = np.load(cdf)['base'][:-1]
cdf = np.load(cdf)['cdf']

def get_cdf_by_sent(sent):
    words = sent.split()
    score = 0.
    for word in words:
        if word in rarity:
            score += np.log(rarity[word])
        else:
            print(word)
    score = -score
    # print("s:", score)
    for idx, b in enumerate(base):
        if score <= b:
            return cdf[idx]
    return 1.


for ll in sys.stdin:
    ll = ll.strip()
    # print(ll)
    print(get_cdf_by_sent(ll))
