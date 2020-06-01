import numpy as np
import yaml
import sys
from sys import getsizeof

SORT = True
model = np.load(sys.argv[1])
Sel_key = sys.argv[2]
print "LOADED model:%s" % sys.argv[1]
if SORT:
    model = sorted(model.items(), key=lambda v: getsizeof(v[1]))
    print "sorted keys..."

all_s = 0
for key in model:
    if SORT:
        name = key[0]
        shape = key[1].shape
        values = key[1]
    else:
        name = key
        shape = model[key].shape
        values = model[key]
    print "Name:", name,
    print "Shapes:", shape,
    s = getsizeof(values)
    print "Size:", s, "," + str(s / 1024 ** 2) + "MB"
    all_s += s
    if name == Sel_key:
        print "values :", values
        v = np.square(values)
        print "values square:", v
        v = np.sum(v, -1)
        print "values sum:", v
        v = np.sqrt(v)
        print "values sqrt:", v
        v = np.sum(v, 0)
        print "values sum:", v
        break
print "=" * 100
# display config
try:
    if SORT:
        yaml_text = [a for a in model if a[0] == "special:model.yml"][0][1]
    else:
        yaml_text = model["special:model.yml"]
    s = ""
    for char in yaml_text:
        s += chr(char)
    print s
except:
    pass

print "=" * 100
# Total size
print "Toal Size: %d, %dMB" % (all_s, all_s / 1024 ** 2)
