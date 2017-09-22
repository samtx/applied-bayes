# HW 2, #3.1

from scipy.stats import bernoulli, beta
from scipy.special import gamma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# Take 100 samples from bernoulli(theta)
# assume sum(y)=57
# Determine probabilities of theta =[0.0, 0.1, ..., 0.9, 1.0] and plot them
# Compute Pr(Sum(y)=57 | theta)

fname = 'hw3_1'
imgFmt = 'png'
labelFontSize = 18
tickFontSize = 14
titleFontSize = 22

def prob_sum57(theta):
    N = 100
    return (theta**57)*(1-theta)**(N-57)

# use uniform(0,1) as prior distribution
def prob_prior(theta):
    return 1

# Bayes constant
def bayes_constant():
    a = 58
    b = 100 - a + 1
    return gamma(a)*gamma(b)/gamma(a+b)

def main():
    probs = np.zeros(11)
    THETAS = np.arange(0,1.1,0.1)
    for i, theta in enumerate(THETAS):
        probs[i] = prob_sum57(theta)*prob_prior(theta)/bayes_constant()
        print(i, probs[i])

def part_b(fname=fname):
    fname = fname + '_b'
    probs = np.zeros(11)
    THETAS = np.arange(0,1.1,0.1)
    for i, theta in enumerate(THETAS):
        probs[i] = prob_sum57(theta)
    fig = plt.figure()
    plt.plot(THETAS,probs,'o-',lw=2)
    plt.xlabel(r'$\theta$', fontsize=labelFontSize)
    plt.ylabel(r'$Pr(\sum_{y_i} = 57 \mid \theta)$', fontsize=labelFontSize)
    plt.title('3.1b', fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt, dpi=300)

def part_c(fname=fname):
    fname = fname + '_c'
    probs = np.zeros(11)
    THETAS = np.arange(0,1.1,0.1)
    for i, theta in enumerate(THETAS):
        probs[i] = prob_sum57(theta)*prob_prior(theta)/bayes_constant()
    fig = plt.figure()
    plt.plot(THETAS,probs,'o-',lw=2)
    plt.xlabel(r'$\theta$', fontsize=labelFontSize)
    plt.ylabel(r'$Pr(\sum_{y_i} = 57 \mid \theta)$', fontsize=labelFontSize)
    plt.title('3.1c', fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt, dpi=300)

def part_d(fname=fname):
    fname = fname + '_d'
    THETAS = np.linspace(0,1,500)
    probs = np.zeros(THETAS.shape)
    for i, theta in enumerate(THETAS):
        probs[i] = prob_sum57(theta)*prob_prior(theta)/bayes_constant()
    fig = plt.figure()
    plt.plot(THETAS,probs,'o-',lw=2)
    plt.xlabel(r'$\theta$', fontsize=labelFontSize)
    plt.ylabel(r'$Pr(\sum_{y_i} = 57 \mid \theta)$', fontsize=labelFontSize)
    plt.title('3.1d', fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt, dpi=300)

def part_e(fname=fname):
    a = 1 + 57
    b = 1 + 100 - 57

    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)

    # plt.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
    # plt.title('beta pdf')

    rv = beta(a, b)
    fig = plt.figure()
    plt.plot(x, rv.pdf(x),'o-',lw=2)
    plt.title('frozen beta pdf',fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt, dpi=300)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_b()
    elif len(sys.argv) > 1:
        if sys.argv[1] == 'b':
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