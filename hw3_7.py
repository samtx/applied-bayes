# HW 2, Problem 3.7

import numpy as np
from scipy.stats import beta, binom
from scipy.special import gamma as gammafn
from scipy.special import gammaln
from scipy.misc import comb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw3_7'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 10
titleFontSize = 14

def lgamma(x):
    return gammafn(np.log(x))
    
def combln(N,k):
    return gammaln(N+1)-gammaln(N-k+1)-gammaln(k+1)
    
def part_c(fname=fname):
    fname = fname + '_c'
    # x = 20
    x = np.arange(0,257)
    # print(lgamma(295))
    # print(gammaln(295))
    def p(y):
        A = gammaln(17)-gammaln(3)-gammaln(14)+combln(257,y)
        B = gammaln(3+y)+gammaln(292-y)-gammaln(295)
        # print(A)
        # print(B)
        return np.exp(A+B)
            
    fig = plt.figure()
    P = p(x)
    plt.plot(x, P,lw=2)
    plt.xlabel(r'$y_2$',fontsize=labelFontSize)
    plt.ylabel(r'$p(Y_2 = y_2 \mid Y_1 = 2)$',fontsize=labelFontSize)
    plt.title('3.7c'+r' $p(Y_2 = y_2 \mid Y_1 = 2)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    print('mean = '+ str(np.mean(P)))
    print('stdv = '+ str(np.std(P)))
    

def part_d(fname=fname):
    fname = fname + '_d'
    # x = 20
    x = np.arange(0,257)
    # print(lgamma(295))
    # print(gammaln(295))
    
    fig = plt.figure()
    P = binom.pmf(x,257,2.0/15.0)
    plt.plot(x, P, lw=1.5)
    plt.xlabel(r'$y_2$',fontsize=labelFontSize)
    plt.ylabel(r'$p(Y_2 = y_2 \mid Y_1 = 2)$',fontsize=labelFontSize)
    plt.title('3.7d'+r' $p(Y_2 = y_2 \mid \theta = 2/15)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    print('mean = '+ str(np.mean(P)))
    print('stdv = '+ str(np.std(P)))
        

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
