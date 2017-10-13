# HW 3, Problem 3.8

import numpy as np
from scipy.stats import beta, binom, bernoulli
from scipy.special import gamma as gammafn
from scipy.special import gammaln
from scipy.misc import comb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw3_8'
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
    # x = 20
    x = np.linspace(0,1,100)
    fig = plt.figure()
    P1 = P = beta.pdf(x,4,4)
    P2 = beta.pdf(x,2,4)
    P3 = beta.pdf(x,4,2)
    plt.plot(x, P, lw=1.5, label='Beta(4,4)')
    plt.plot(x, P2, lw=1.5, label='Beta(2,4)')
    plt.plot(x, P3, lw=1.5, label='Beta(4,2)' )
    # plt.plot(x, 0.2*P1+0.4*P2+0.4*P3, lw=1.5)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta)$',fontsize=labelFontSize)
    plt.title('3.8a',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.legend()
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    print('mean = '+ str(np.mean(P)))
    print('stdv = '+ str(np.std(P)))

    fig = plt.figure()
    plt.plot(x, 0.2*P1+0.4*P2+0.4*P3, lw=1.5)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta)$',fontsize=labelFontSize)
    plt.title(r'3.8a $p(\theta)=0.2Beta(4,4)+0.4Beta(2,4)+0.4Beta(4,2)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    # plt.legend()
    plt.savefig(fname+'2.'+imgFmt, format=imgFmt)


def part_c(fname=fname):
    fname = fname + '_c'
    # x = 20
    x = np.linspace(0,1,100)
    y = 24
    n = 50
    fig = plt.figure()
    def prior(x):
        return 0.2*beta.pdf(x,4,4)+0.4*beta.pdf(x,4,2)+0.4*beta.pdf(x,2,4)
    def likelihood(x,y,n):
        return (x**y)*(1-x)**(n-y)
    P = prior(x)*likelihood(x,y,n)
    plt.plot(x, P, lw=1.5)
    # plt.plot(x, 0.2*P1+0.4*P2+0.4*P3, lw=1.5)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta \mid y)$',fontsize=labelFontSize)
    plt.title('3.8c',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    # plt.legend()
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    print('mean = '+ str(np.mean(P)))
    print('stdv = '+ str(np.std(P)))


def part_d(fname=fname):
    fname = fname + '_d'
    # x = 20
    x = np.linspace(0,1,100)
    y = 23
    n = 50
    fig = plt.figure()
    def prior(x):
        return 0.2*beta.pdf(x,4,4)+0.4*beta.pdf(x,4,2)+0.4*beta.pdf(x,2,4)
    def likelihood(x,y,n):
        return (x**y)*(1-x)**(n-y)
    P = prior(x)*likelihood(x,y,n)
    plt.plot(x, P, lw=1.5)
    # plt.plot(x, 0.2*P1+0.4*P2+0.4*P3, lw=1.5)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta \mid y)$',fontsize=labelFontSize)
    plt.title('3.8d',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    # plt.legend()
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
