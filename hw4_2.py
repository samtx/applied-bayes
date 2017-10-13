# HW 4, Problem 4.2

import numpy as np
from scipy.stats import gamma, poisson
# from scipy.special import gamma as gammafn
# from scipy.special import gammaln
# from scipy.misc import comb
# import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw4_2'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 12
titleFontSize = 14

# def lgamma(x):
#     return gammafn(np.log(x))

# def combln(N,k):
#     return gammaln(N+1)-gammaln(N-k+1)-gammaln(k+1)

def part_a(fname=fname):
    fname = fname + '_a'
    n = 100000
    # pA = random.gammavariate(120, scale=1.0/10, size=n)
    # pB = random.gammavariate(12, scale=1.0/1, size=n)
    alpha=237
    beta=20 
    pA = gamma.rvs(alpha, scale=1.0/beta, size=n)
    alpha=125
    beta=14 
    pB = gamma.rvs(alpha, scale=1.0/beta, size=n)
    y = 0.0
    for i in range(n):
        if (pB[i] < pA[i]):
            y += 1.0
    print("Pr(thetaB < thetaA) = {:.5f}".format(y/float(n)))
            
            
def part_b(fname=fname):
    fname += '_b'
    n0_list = np.arange(1,250)
    n = int(1e5)
    alpha=237
    beta=20 
    pA = gamma.rvs(alpha, scale=1.0/beta, size=n)
    Pr = np.zeros(n0_list.size) 
    for j, n0 in enumerate(n0_list):
        y = 0.0
        alpha=12*n0
        beta=n0 
        pB = gamma.rvs(alpha, scale=1.0/beta, size=n)
        for i in range(n):
            if (pB[i] < pA[i]):
                y += 1.0
        Pr[j] = y/float(n)
    plt.figure()
    plt.plot(n0_list, Pr, lw=1.5)
    plt.xlabel(r'$n_0$',fontsize=labelFontSize)
    plt.ylabel(r'$Pr(\theta_B < \theta_A \mid y_A, y_B)$',fontsize=labelFontSize)
    plt.title('4.2b',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
        

def part_c(fname=fname):
    fname = fname + '_c'
    n = int(1e4)
    # repeat part a
    # y = 0.0
    # for i in range(n):
    #     thetaA = gamma.rvs(237, scale=1.0/20)
    #     yA = poisson.rvs(thetaA)    
    #     thetaB = gamma.rvs(125, scale=1.0/14)
    #     yB = poisson.rvs(thetaB)
    #     if (yB < yA):
    #         y += 1.0
    # print("Pr(yB < yA) = {:.5f}".format(y/float(n)))
    
    # repeat part b
    n0_list = np.arange(1,250)
    Pr = np.zeros(n0_list.size) 
    for j, n0 in enumerate(n0_list):
        print n0
        y = 0.0
        for i in range(n):
            thetaA = gamma.rvs(237, scale=1.0/20)
            yA = poisson.rvs(thetaA)  
            thetaB = gamma.rvs(12*n0, scale=1.0/n0)
            yB = poisson.rvs(thetaB)  
            if (yB < yA):
                y += 1.0
        Pr[j] = y/float(n)
    plt.figure()
    plt.plot(n0_list, Pr, lw=1.5)
    plt.xlabel(r'$n_0$',fontsize=labelFontSize)
    plt.ylabel(r'$Pr(\tilde{Y}_B < \tilde{Y}_A \mid y_A, y_B)$',fontsize=labelFontSize)
    plt.title('4.2c',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_a()
    elif len(sys.argv) > 1:

        if sys.argv[1] == 'a':
            print('part a...')
            part_a()
        # elif sys.argv[1] == 'a1':
        #     print('part a1...')
        #     part_a1()
        # elif sys.argv[1] == 'a2':
        #     print('part a2...')
        #     part_a2()
        # elif sys.argv[1] == 'a3':
        #     print('part a3...')
        #     part_a3()
        elif sys.argv[1] == 'b':
            print('part b...')
            part_b()
        elif sys.argv[1] == 'c':
            print('part c...')
            part_c()
        # elif sys.argv[1] == 'd':
        #     print('part d...')
        #     part_d()
        # elif sys.argv[1] == 'e':
        #     print('part e...')
        #     part_e()
