# HW 4, Problem 4.1

import numpy as np
from scipy.stats import beta, binom, bernoulli
from scipy.special import gamma as gammafn
from scipy.special import gammaln
from scipy.misc import comb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw4_1'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 12
titleFontSize = 14

def lgamma(x):
    return gammafn(np.log(x))

def combln(N,k):
    return gammaln(N+1)-gammaln(N-k+1)-gammaln(k+1)

def part_a(fname=fname):
    fname = fname + '_a'
    n = 5000
    P1 = beta.rvs(58,44,size=n) # beta(57+1,100-57+1)
    P2 = beta.rvs(31,21,size=n)  # beta(30+1,50-30+1) 
    y = 0.0
    for i in range(n):
        if (P1[i] < P2[i]):
            y += 1.0
    print("Pr(theta1 < theta2) = {:6.4f}".format(y/float(n)))
            

def part_e(fname=fname):
    pass

if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_a()
    elif len(sys.argv) > 1:

        if sys.argv[1] == 'a':
            print('part a...')
            part_a()
        elif sys.argv[1] == 'a1':
            print('part a1...')
            part_a1()
        elif sys.argv[1] == 'a2':
            print('part a2...')
            part_a2()
        elif sys.argv[1] == 'a3':
            print('part a3...')
            part_a3()
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
