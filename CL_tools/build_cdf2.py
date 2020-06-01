#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import math
import numpy as np

def count_words(filename):
    counter = collections.Counter()

    with open(filename, "r") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))

    return words, counts


def control_symbols(string):
    if not string:
        return []
    else:
        return string.strip().split(",")


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    # pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    pairs = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    words, ids = list(zip(*pairs))

    # total freq
    T_freq = sum(ids)

    with open(name, "w") as f:
        for i, word in enumerate(words):
            # f.write(word + " " + str(ids[i]) + "\n")
            f.write(word + " " + "%.16f" % (ids[i] / T_freq) + "\n")
        # write total freq

def cal_cdf_model(corpus, vocab):
    pairs = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    words, ids = list(zip(*pairs))

    freq_dict = {}
    for word, id in zip(words, ids):
        freq_dict[word] = id

    T_freq = sum(ids)
    data = []
    debug = 0
    with open(corpus, "r") as f:
        for line in f.readlines():
            line = line.split()
            SUM = 0
            for w in line:
                p = freq_dict[w] / T_freq
                if p != 0:
                    SUM += math.log(p)
            SUM = -SUM
            data.append(SUM)
            # if SUM < 5.718:
            #     debug += 1
            #  print (SUM)
    # data contains all sum log
    # bins='auto'
    v, base = np.histogram(data, bins=np.arange(1000))
    print ("data:", data[:50])
    print ("value", v[:50])
    base = base.astype(np.float32)
    print ("base:", base[:50])
    print ("highest value:", base[-1])
    print ("len of base:", len(base))
    # print ("debug:", debug)
    cdf = np.cumsum(v)
    cdf = cdf / len(data)
    cdf = cdf.astype(np.float32)
    print ("cdf:", cdf, cdf.dtype)
    print ("outputing cdf and bases.")
    # res = {"cdf": cdf, "base": base}
    np.savez(args.output + "-cdf_base.npz", cdf=cdf, base=base)



def parse_args():
    parser = argparse.ArgumentParser(description="Create vocabulary")

    parser.add_argument("corpus", help="input corpus")
    parser.add_argument("output", default="vocab.txt",
                        help="Output vocabulary name")
    parser.add_argument("--limit", default=0, type=int, help="Vocabulary size")
    parser.add_argument("--control", type=str, default="",
                        help="Add control symbols to vocabulary. "
                             "Control symbols are separated by comma.")

    return parser.parse_args()

args=parse_args()

def main():
    vocab = {}
    limit = args.limit
    count = 0

    words, counts = count_words(args.corpus)
    ctrl_symbols = control_symbols(args.control)

    for sym in ctrl_symbols:
        vocab[sym] = len(vocab)

    for word, freq in zip(words, counts):
        if limit and len(vocab) >= limit:
            break

        if word in vocab:
            print("Warning: found duplicate token %s, ignored" % word)
            continue

        # vocab[word] = len(vocab)
        # print(word, freq)
        vocab[word] = freq
        count += freq

    save_vocab(args.output, vocab)
    cal_cdf_model(args.corpus, vocab)

    print("Total words: %d" % sum(counts))
    print("Unique words: %d" % len(words))
    print("Vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))


if __name__ == "__main__":
    main()
