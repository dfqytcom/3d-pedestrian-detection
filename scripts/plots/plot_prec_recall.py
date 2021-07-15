"""
Function to plot precision vs recall curves.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

with open('../results/precision_recall/precision_recall_confidence_thresholds.txt', 'r') as f:
    x = f.read().splitlines()

precision = np.array([float(i) for i in x[0].split(',')])
recall = np.array([float(i) for i in x[1].split(',')])
conf = np.array([float(i) for i in x[2].split(',')])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig,ax = plt.subplots()

plt.figure(figsize=(8,6))
plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)

plt.axis([0.5, 0.9, 0.8, 1])
# plt.title('Precision-recall curve')
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
sc = plt.scatter(recall, precision, c=conf, cmap='winter')
cbar = plt.colorbar()
cbar.set_label('Confidence threshold', fontsize=15)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([str(conf[ind["ind"]])]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor('green')

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
