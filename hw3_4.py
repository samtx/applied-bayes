# HW 2, Problem 3.4

import numpy as np
from scipy.stats import beta, binom
from scipy.special import gamma as gammafn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw3_4'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 10
titleFontSize = 14

def part_a(fname=fname):
    part_a1(fname=fname)
    
def part_a1(fname=fname):
    fname = fname + '_a'
    a = 2
    b = 8
    x = np.linspace(beta.ppf(0.0, a, b), beta.ppf(1.0, a, b), 100)
    rv = beta(a, b)
    fig = plt.figure()
    plt.plot(x, rv.pdf(x),lw=2)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta \mid a=2, b=8)$',fontsize=labelFontSize)
    plt.title('3.4a'+r' $p(\theta)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'1'+'.'+imgFmt, format=imgFmt)
    
def part_a2(fname=fname):
    fname = fname + '_a2'
    m = 5
    theta_set = np.linspace(1/float(m),1-1/float(m),m)
    print(theta_set)
    n = 34
    x = np.arange(n+1)
    fig = plt.figure()
    hh = []
    for i, theta in enumerate(theta_set):
        h, = plt.plot(x, binom.pmf(x, n, theta), lw=1.5, label=r'$\theta='+'{:.2f}'.format(theta)+'$')
        hh.append(h)
    plt.xlabel(r'$y$',fontsize=labelFontSize)
    plt.ylabel(r'$p(y \mid \theta)$',fontsize=labelFontSize)
    plt.title('3.4a'+r' $p(y \mid \theta)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.legend()
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    
def part_a3(fname=fname):
    fname = fname + '_a3'
    # m = 5
    # theta_set = np.linspace(1/float(m),1-1/float(m),m)
    # print(theta_set)
    a = 2
    b = 8
    n = 34
    y = 15
    a2 = y + a
    b2 = n + b - y
    x = np.linspace(0,1, 100)
    fig = plt.figure()
    plt.plot(x, beta.pdf(x, a2, b2),lw=1.5)
    # hh = []
    # for i, theta in enumerate(theta_set):
    #     h, = plt.plot(x, binom.pmf(x, n, theta), lw=1.5, label=r'$\theta='+'{:.2f}'.format(theta)+'$')
    #     hh.append(h)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta \mid y)$',fontsize=labelFontSize)
    plt.title('3.4a'+r' $p(\theta \mid y)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    # plt.legend()
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    print('95% CI = ',beta.interval(0.05,a2,b2))

def part_b(fname=fname):
    part_b1()
    part_b2()
    part_b3()

    
def part_b1(fname=fname):
    fname = fname + '_b'
    a = 8
    b = 2
    x = np.linspace(beta.ppf(0.0, a, b), beta.ppf(1.0, a, b), 100)
    rv = beta(a, b)
    fig = plt.figure()
    plt.plot(x, rv.pdf(x),lw=2)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta \mid a=8, b=2)$',fontsize=labelFontSize)
    plt.title('3.4b'+r' $p(\theta)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'1'+'.'+imgFmt, format=imgFmt)
    
def part_b2(fname=fname):
    fname =fname + '_b2'
    m = 5
    theta_set = np.linspace(1/float(m),1-1/float(m),m)
    print(theta_set)
    n = 34
    x = np.arange(n+1)
    fig = plt.figure()
    hh = []
    for i, theta in enumerate(theta_set):
        h, = plt.plot(x, binom.pmf(x, n, theta), lw=1.5, label=r'$\theta='+'{:.2f}'.format(theta)+'$')
        hh.append(h)
    plt.xlabel(r'$y$',fontsize=labelFontSize)
    plt.ylabel(r'$p(y \mid \theta)$',fontsize=labelFontSize)
    plt.title('3.4b'+r' $p(y \mid \theta)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.legend()
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    
def part_b3(fname=fname):
    fname = fname + '_b3'
    # m = 5
    # theta_set = np.linspace(1/float(m),1-1/float(m),m)
    # print(theta_set)
    a = 8
    b = 2
    n = 34
    y = 15
    a2 = y + a
    b2 = n + b - y
    x = np.linspace(0,1, 100)
    fig = plt.figure()
    plt.plot(x, beta.pdf(x, a2, b2),lw=1.5)
    # hh = []
    # for i, theta in enumerate(theta_set):
    #     h, = plt.plot(x, binom.pmf(x, n, theta), lw=1.5, label=r'$\theta='+'{:.2f}'.format(theta)+'$')
    #     hh.append(h)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta \mid y)$',fontsize=labelFontSize)
    plt.title('3.4b'+r' $p(\theta \mid y)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    # plt.legend()
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    print('95% CI = ',beta.interval(0.05,a2,b2))

def part_c(fname=fname):
    fname = fname + '_c'
    x = np.linspace(0,1,100)
    def p(t):
        return 0.25*gammafn(10)/(gammafn(2)*gammafn(8))*(3*t*(1-t)**7+t**7*(1-t))
    fig = plt.figure()
    plt.plot(x, p(x),lw=2)
    plt.xlabel(r'$\theta$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta)$',fontsize=labelFontSize)
    plt.title('3.4c'+r' $p(\theta)$',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    
def part_d(fname=fname):
    pass

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
