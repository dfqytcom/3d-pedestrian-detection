"""
Function to plot histograms.
"""
import matplotlib.pyplot as plt
import sys

filename = "../results/histograms/bbp/txt/bbp_width_histogram.txt"

with open(filename, 'r') as f:
    lines=f.readlines()
x=[]
y=[]
for i in lines:
    x.append(i.split(',')[0])
    y.append(int(i.split(',')[1]))

plt.bar(x, y)
plt.xlabel("Example")
plt.ylabel("Example")
plt.title("Example-example histogram")
plt.savefig('../results/histograms/bbp/png/example.png')
plt.show()
