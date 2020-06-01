import matplotlib.pyplot as plt
import numpy as np
import sys

print "python x.py cdf_base.npz T"
f = np.load(sys.argv[1])
plt.title(sys.argv[1])
base = f['base']
cdf = f['cdf']
plt.plot(base[:-1], cdf)
# plt.show()

# make competence graph
g = plt.figure(2)
c0 = 0.001
T = int(sys.argv[2])
def c(t):
    tmp = t * ((1 - c0 * c0) / T) + c0 * c0
    c_square = tmp ** 0.5
    return min(1.0, c_square)

t_data = [t for t in range(T)]
c_data = [c(t) for t in range(T)]

plt.title("Competence")
plt.plot(t_data, c_data)
plt.show()
