import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
import numpy as np
import sys

print ("python x.py model.emb.txt")
ENG = ticker.EngFormatter(sep='')
ENG.ENG_PREFIXES[3] = 'K'
print(ENG.format_eng(50000))
# fontP = font_manager.FontProperties()
# fontP.set_family('SimHei')
# fontP.set_size(12)
#
# matplotlib.rcParams['font.family'] = "Times New Roman"
plt.rc('font', size=15, family='Times New Roman')

f = open(sys.argv[1], 'r').readlines()[1:]

x_data = [x for x in range(len(f))]
y_data = []

for i, data in enumerate(f):
    data = data.split()[1:]
    data = [float(x) for x in data]
    data = np.array(data)
    data = np.square(data)
    data = np.sum(data, 0)
    data = np.sqrt(data)
    y_data.append(data)

x_data = x_data[:20000]
y_data = y_data[:20000]

# reformat
# for i, d in enumerate(x_data):
#   x_data[i] = str(ENG.format_eng(d)).replace(" ", "").replace("k", "K")
# print(x_data[:10])
# plt.title(sys.argv[1])
plt.gca().xaxis.set_major_formatter(ENG)
plt.plot(x_data, y_data)
plt.locator_params(nbins=5)
plt.ylabel("Norm of Word Vector")
plt.xlabel("Word Frequency Ranking")


# save figure
plt.savefig(sys.argv[1]+'.pdf',format="pdf",bbox_inches='tight')

plt.show()