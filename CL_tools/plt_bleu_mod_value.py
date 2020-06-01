import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')

plt.rc('font', size=15, family='Times New Roman')
print("python x.py valid.log mod_file")
ENG = ticker.EngFormatter(sep='')
ENG.ENG_PREFIXES[3] = 'K'

def draw_bleu(bleu_f, name, plt_name, color):
    bleu = {}
    for l in bleu_f:
        if l.find('translation') != -1:
            l = l.split()
            step = int(l[7])
            score = float(l[11])
            bleu[step] = score
    # follow the name, fit bleu to name
    BLEU = []
    for n in name:
        BLEU.append(bleu[n])
    # BLEU
    BLEU = BLEU[:TOTAL_STEP]
    # "BLEU_" + plt_name
    print("ax2, step:%d" % len(name))
    curve = ax2.plot(name, BLEU, label='BLEU', color=color, marker='^')
    print(bleu)
    return curve
        
def draw(MOD_log, plt_name, color):
    r = 0
    data = []
    name = []
    mod = []
    for line in MOD_log:
        # print (line)
        if r % 2 == 0:
            t = line.split(".")[1][4:]
            name.append(int(t))
        else:
            mod.append(float(line))
        r += 1
    for i, n in enumerate(name):
          data.append((n, mod[i]))

    data.sort()
    data = data[:TOTAL_STEP]
    step = [x[0] for x in data]
    mod = [x[1] for x in data]
    # normlization
    # mod = [x[1] / 23615 for x in data]
    # plt_name + "_NORM"
    print("ax1, step:%d" % len(step))
    curve = ax1.plot(step, mod, label='Norm', color=color, marker='s')
    print(data)
    print("=" * 100)
    return step, curve

TOTAL_STEP = 40
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel("Norm of Source Embedding")
ax2 = plt.twinx()
ax2.set_ylabel("BLEU Score")
#
ax1.yaxis.set_major_formatter(ENG)
ax2.xaxis.set_major_formatter(ENG)
#
colors = {1:["b", "g"], 3:["r", "c"], 5:["m", "y"]}
names = {1: "BASE", 3: "CL", 5: "MOD"}
for i in range(1, len(sys.argv), 2):
    bleu_f = open(sys.argv[i], 'r').readlines()
    MOD_log = open(sys.argv[i + 1], 'r').readlines()
    step, mod_curve = draw(MOD_log, names[i], colors[i][0])
    bleu_curve = draw_bleu(bleu_f, step, names[i], colors[i][1])
#
ax1.set_xlabel("Training Step")
h = [bleu_curve[0], mod_curve[0]]
plt.legend(handles=h, loc="lower right")
# fig.legend(loc="lower right", bbox_to_anchor=(1,0), bbox_transform=ax1.transAxes)
# plt.title("NORM + BLEU")
# plt.locator_params(nbins=10)
ax1.locator_params(nbins=10)
# ax1.yaxis.set_major_locator(MaxNLocator(nbins=10))
# ax2.yaxis.set_major_locator(MaxNLocator(nbins=10))

plt.savefig('MOD.pdf',format="pdf",bbox_inches='tight')
plt.show()

