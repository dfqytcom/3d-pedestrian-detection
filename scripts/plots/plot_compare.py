"""
Function to compare different results in a plot.
"""

import matplotlib.pyplot as plt
import pickle
import sys
import os
import numpy as np

if len(sys.argv) < 4:
    print('Usage: {} <directory> <history> <train/eval>'.format(sys.argv[0]))
    exit()

# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(figsize=(8,6))
plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
epochs = np.arange(0, 251)

str = ''
if sys.argv[-1] == 'train' or sys.argv[-1] == 'eval':
    str += sys.argv[-2] + ' ' + sys.argv[-1]
    for i in np.arange(1, len(sys.argv) - 2):
        h = pickle.load(open(os.path.join(sys.argv[i], sys.argv[-1] + '_' + sys.argv[-2] + '_history.p'), 'rb'))
        if sys.argv[-1] == 'train':
            plt.plot(epochs - 0.5, h, label=sys.argv[i])
        else:
            plt.plot(epochs, h, label=sys.argv[i], lw=2)
else:
    str += sys.argv[-1]
    for i in np.arange(1, len(sys.argv) - 1):
        h = pickle.load(open(os.path.join(sys.argv[i], sys.argv[-1] + '_history.p'), 'rb'))
        plt.plot(epochs, h, label=sys.argv[i])

# plt.ylim(top=1.0)
plt.xlim(right=100)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Model Loss', fontsize=15)
plt.legend(fontsize=13)
plt.savefig('Model_Loss_Modelnet.png')
plt.show()
