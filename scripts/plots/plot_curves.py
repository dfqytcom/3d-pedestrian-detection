"""
Function to plot different curves of a result.
"""

import matplotlib.pyplot as plt
import pickle
import sys
import os
import numpy as np

if len(sys.argv) < 3:
    print('Usage: {} <directory> <history>'.format(sys.argv[0]))
    exit()

if sys.argv[2] == "precision":
    history1 = os.path.join(sys.argv[1], 'precision_history.p')
    history2 = os.path.join(sys.argv[1], 'recall_history.p')
else:
    history1 = os.path.join(sys.argv[1], 'eval_' + sys.argv[2] + '_history.p')
    history2 = os.path.join(sys.argv[1], 'train_' + sys.argv[2] + '_history.p')

h1 = pickle.load(open(history1,'rb'))
h2 = pickle.load(open(history2,'rb'))

print(len(h1))

epochs = np.arange(0, len(h1))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(figsize=(8,6))
plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)

# plt.plot(epochs, h1, label='Validation')
# plt.plot(epochs - 0.5, h2, label='Train')

plt.plot(epochs, h1, label='Precision')
plt.plot(epochs, h2, label='Recall')

# plt.ylim(top=1.0)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Value', fontsize=15)
plt.legend(fontsize=13)
plt.savefig('fig.png')
plt.show()
