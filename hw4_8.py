# HW 4, Problem 4.8

import numpy as np
from scipy.stats import gamma, poisson, bayes_mvs
# from scipy.special import gamma as gammafn
# from scipy.special import gammaln
# from scipy.misc import comb
# import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw4_8'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 12
titleFontSize = 14

# def lgamma(x):
#     return gammafn(np.log(x))

# def combln(N,k):
#     return gammaln(N+1)-gammaln(N-k+1)-gammaln(k+1)

str_bach = (
    "1 0 0 1 2 2 1 5 2 0 0 0 0 0 0 1 1 1 0 0 0 1 1 2 1 3 2 0 0 3 0 0 0 2 "
    "1 0 2 1 0 0 1 3 0 1 1 0 2 0 0 2 2 1 3 0 0 0 1 1")
str_nobach = (
    "2 2 1 1 2 2 1 2 1 0 2 1 1 2 0 2 2 0 2 1 0 0 3 6 1 6 4 0 3 2 0 1 0 0 0 3 0 "
    "0 0 0 0 1 0 4 2 1 0 0 1 0 3 2 5 0 1 1 2 1 2 1 2 0 0 0 2 1 0 2 0 2 4 1 1 1 "
    "2 0 1 1 1 1 0 2 3 2 0 2 1 3 1 3 2 2 3 2 0 0 0 1 0 0 0 1 2 0 3 3 0 1 2 2 2 "
    "0 6 0 0 0 2 0 1 1 1 3 3 2 1 1 0 1 0 0 2 0 2 0 1 0 2 0 0 2 2 4 1 2 3 2 0 0 "
    "0 1 0 0 1 5 2 1 3 2 0 2 1 1 3 0 5 0 0 2 4 3 4 0 0 0 0 0 0 2 2 0 0 2 0 0 1 "
    "1 0 2 1 3 3 2 2 0 0 2 3 2 4 3 3 4 0 3 0 1 0 1 2 3 4 1 2 6 2 1 2 2")

data_bach = np.fromstring(str_bach, sep=" ")
data_nobach = np.fromstring(str_nobach, sep=" ")

# sum_bach = np.sum(data_bach)
# sum_nobach = np.sum(data_nobach)

# n_bach = data_bach.size
# n_nobach = data_nobach.size

# print("Bach:    sum={0:3f}  n={1:3d}".format(np.sum(data_bach), data_bach.size))
# print("No Bach: sum={0:3f}  n={1:3d}".format(np.sum(data_nobach), data_nobach.size))


def part_a(fname=fname):
    fname = fname + '_a'
    nsamps = int(5000)
    yA = np.zeros(nsamps)
    yB = np.zeros(nsamps)
    for i in range(nsamps):   # do Monte Carlo
        thetaA = gamma.rvs(56, scale=1.0/59)
        yA[i] = poisson.rvs(thetaA)    
        thetaB = gamma.rvs(307, scale=1.0/219)
        yB[i] = poisson.rvs(thetaB)
    
    bins = range(9)
    plt.figure()
    plt.hist(yA, bins=bins, normed=True)
    plt.xlabel(r'$\tilde{Y}_A$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\tilde{Y}_A \mid \mathbf{y}_A)$',fontsize=labelFontSize)
    plt.title(r"4.8a  Bachelor's Degree",fontsize=titleFontSize)
    plt.xticks(bins, fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'bach_density.'+imgFmt, format=imgFmt)
    print(yA)
    
    plt.figure()
    plt.hist(yB, bins=bins, normed=True)
    plt.xlabel(r'$\tilde{Y}_B$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\tilde{Y}_B \mid \mathbf{y}_B)$',fontsize=labelFontSize)
    plt.title(r"4.8a  No Bachelor's Degree",fontsize=titleFontSize)
    plt.xticks(bins, fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'nobach_density.'+imgFmt, format=imgFmt)
    
            
            
def part_b(fname=fname):
    fname += '_b'
    nsamps = int(1e5)
    thetaA = np.zeros(nsamps)
    thetaB = np.zeros(nsamps)
    yA = np.zeros(nsamps)
    yB = np.zeros(nsamps)
    for i in range(nsamps):   # do Monte Carlo
        thetaA[i] = gamma.rvs(56, scale=1.0/59)
        yA[i] = poisson.rvs(thetaA[i])    
        thetaB[i] = gamma.rvs(307, scale=1.0/219)
        yB[i] = poisson.rvs(thetaB[i])        
    thetaBA = thetaB - thetaA
    yBA = yB - yA
    
    plt.figure()
    plt.hist(thetaBA, bins=16, normed=True)
    plt.xlabel(r'$\theta_B-\theta_A$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta_B-\theta_A)$',fontsize=labelFontSize)
    plt.title(r"4.8b  $\theta_B-\theta_A$",fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'thetaBA.'+imgFmt, format=imgFmt)
    
    plt.figure()
    plt.hist(yBA, bins=16, normed=True)
    plt.xlabel(r'$\tilde{Y}_B-\tilde{Y}_A$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\tilde{Y}_B-\tilde{Y}_A)$',fontsize=labelFontSize)
    plt.title(r"4.8b  $\tilde{Y}_B-\tilde{Y}_A$",fontsize=titleFontSize)
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'yBA.'+imgFmt, format=imgFmt)
    
    thetaCI = bayes_mvs(thetaB - thetaA, alpha=0.95)
    yCI = bayes_mvs(yB - yA, alpha=0.95)
    print("thetaCI = {0}".format(thetaCI[0][1]))
    print("yCI     = {0}".format(yCI[0][1]))
    
    thetaCI2 = (np.percentile(thetaBA, 2.5), np.percentile(thetaBA, 97.5)) 
    yCI2 = (np.percentile(yBA, 2.5), np.percentile(yBA, 97.5)) 
    print("thetaCI2 = {0}".format(thetaCI2))
    print("yCI2     = {0}".format(yCI2))
    
    

def part_c(fname=fname):
    fname = fname + '_c'
    rv = poisson(1.4)  # poisson pmf with theta=1.4
    x = range(8)
    plt.figure()
    plt.hist(data_nobach, bins=8.5, alpha=0.5, normed=True, label='Empirical Data')
    plt.plot(x, rv.pmf(x), lw=3, marker='o', ms=10, label=r'Poisson($\hat{\theta}_B=1.4$)')
    # plt.xticks(range(8))
    plt.xlim(-0.5,8)
    plt.xlabel(r'Y',fontsize=labelFontSize)
    plt.ylabel(r'P',fontsize=labelFontSize)
    plt.title(r"4.8c",fontsize=titleFontSize)
    plt.legend(loc="best")
    plt.xticks(fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)
    
    
def part_d(fname=fname):
    fname += '_d'
    nsamp = 5000
    nB = 218
    thetaB = gamma.rvs(307, scale=1.0/219, size=nsamp)
    yB_zeros = np.zeros(nsamp, dtype=np.int)
    yB_ones = np.zeros(nsamp, dtype=np.int)
    for i in range(nsamp):
        yB = poisson.rvs(thetaB[i], size=nB)
        yB_zeros[i] = int(np.bincount(yB)[0])
        yB_ones[i] = int(np.bincount(yB)[1])
    plt.figure()
    plt.plot(range(nsamp), yB_zeros, label='Zero children')
    plt.xlabel('Sample Number', fontsize=labelFontSize)
    plt.legend(loc='best')
    plt.savefig(fname+'_zeros.'+imgFmt, format=imgFmt)
    
    a = yB_zeros[0]
    b = yB_ones[0]
    
    plt.figure()
    plt.scatter(yB_zeros, yB_ones)
    plt.plot(a, b , 'or', ms=10)
    plt.xlabel('Zero Children', fontsize=labelFontSize)
    plt.ylabel('One Child', fontsize=labelFontSize)
    plt.title('4.8d',fontsize=titleFontSize)
    # plt.legend(loc='best')
    plt.savefig(fname+'_scatter.'+imgFmt, format=imgFmt)
    
    print('zero={0}, one={1}'.format(a,b))
    

        
    
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
        elif sys.argv[1] == 'd':
            print('part d...')
            part_d()
        # elif sys.argv[1] == 'e':
        #     print('part e...')
        #     part_e()
