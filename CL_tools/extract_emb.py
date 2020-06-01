import numpy as np
import yaml
import sys
from sys import getsizeof

SORT = True
model = np.load(sys.argv[1])

f = open(sys.argv[2], 'r', encoding='utf-8', errors='ignore').readlines()
output = open('emb.txt', 'w')

vocab = {}
for i, l in enumerate(f):
    vocab[i] = l.split(": ")[0].strip("\"")
display_key = "encoder_Wemb"

print("LOADED model:%s" % sys.argv[1])
if SORT:
    model = sorted(list(model.items()), key=lambda v: getsizeof(v[1]))
    print("sorted keys...")

all_s = 0
all_parameters = 0
for key in model:
    if SORT:
        name = key[0]
        shape = key[1].shape
        values = key[1]
    else:
        name = key
        shape = model[key].shape
        values = model[key]
    print("Name:", name, end=' ')
    print("Shapes:", shape, end=' ')
    all_parameters += (shape[0] * shape[1]) if len(shape) == 2 else shape[0]
    s = getsizeof(values)
    print("Size:", s, "," + str(s / 1024 ** 2) + "MB")
    all_s += s
    # display value
    if display_key and name == display_key:
        output.write("%d %d\n" % (len(vocab), values.shape[1]))
        for i, v in enumerate(values):
            v = [str(x) for x in v]
            output.write(vocab[i] + " " + " ".join(v) + "\n")


print("=" * 100)
# display config
try:
    if SORT:
        yaml_text = [a for a in model if a[0] == "special:model.yml"][0][1]
    else:
        yaml_text = model["special:model.yml"]
    s = ""
    for char in yaml_text:
        s += chr(char)
    print(s)
except:
    pass

print("=" * 100)
# Total size
print("Toal Size: %d, %dMB, %s parameters" % (all_s, all_s / 1024 ** 2, all_parameters))
