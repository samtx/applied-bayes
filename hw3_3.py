# HW 2, Problem 3.3

import numpy as np
from scipy.stats import gamma, poisson
from scipy.special import gamma as gammafn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw3_3'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 10
titleFontSize = 14

y_a = np.array([12,9,12,14,13,13,15,8,15,6])
y_b = np.array([11,11,10,9,9,8,7,10,6,8,8,9,7])

def part_a():
    pass
    
def part_b(fname=fname):
    fname = fname + '_b'
    sumY = np.sum(y_b)
    n0_set = range(1,51)
    E = np.zeros(len(n0_set))
    for i, n0 in enumerate(n0_set):         
        # calculate posterior expectation of thetaB.
        a = np.float(12*n0 + sumY)
        b = np.float(n0)
        E[i] = a/b
    fig = plt.figure()
    plt.plot(n0_set,E,'o-',lw=2)
    plt.xlabel(r'$n_0$', fontsize=labelFontSize)
    plt.ylabel(r'$E[\theta_B \mid \mathbf{y}_b, n_0]$', fontsize=labelFontSize)
    plt.title('3.3b', fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_b()
    elif len(sys.argv) > 1:
        if sys.argv[1] == 'a':
            print('part a...')
            part_a()
        elif sys.argv[1] == 'b':
            print('part b...')
            part_b()
        elif sys.argv[1] == 'c':
            print('part c...')
            part_c()
        elif sys.argv[1] == 'd':
            print('part d...')
            part_d()
        elif sys.argv[1] == 'e':
            print('part e...')
            part_e()
