# HW 6, Problem 6.1

import numpy as np
from scipy.stats import gamma, poisson, bayes_mvs
from numpy.linalg import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw6_1'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 12
titleFontSize = 14

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


def part_d(fname=fname):
    fname = fname + '_d'
    nsamps = int(5000)
    gibbs_iter = int(5000)
    burn_in = 1000
    yA = np.zeros(nsamps)
    yB = np.zeros(nsamps)
    ybarA = np.mean(data_bach)
    n = data_bach.size
    ybarB = np.mean(data_nobach)
    m = data_nobach.size
    at = 2.0
    bt = 1.0
    ag_list = (8, 16, 32, 64, 128)
    p = np.zeros(len(ag_list))
    # for idx, ag in enumerate(ag_list):
    #     ag = np.float(ag)
    #     bg = ag
    #     print('ag=',ag)
    #     # set initial values for gma and theta
    #     gma = np.ones(nsamps)*1.0
    #     theta = np.ones(nsamps)*1.0
    #     tol = np.float(1e-4)  # relative tolerance
    #     # Do Gibbs sampling to get theta and gamma
    #     for i in range(gibbs_iter):
    #         gma_old = gma
    #         theta_old = theta
    #         for j in range(nsamps):
    #             theta[j] = gamma.rvs(n*ybarA+m*ybarB+at, scale=1.0/(n+m*gma[j]+bt))
    #             gma[j] = gamma.rvs(m*ybarB+ag, scale=1.0/(m*theta[j]+bg))
    #         if ((norm(gma)-norm(gma_old))/norm(gma_old) < tol) and (i > burn_in):  # burn in
    #             print('break... i=',i)
    #             break
    #     thetaA = theta
    #     thetaB = theta * gma  # gma = thetaA/thetaB
    #     p[idx] = np.mean(thetaB - thetaA)
    #     print('p (ag=',ag,') = ',p[idx])
    # print(p)
    p = [0.38144128,  0.33382475,  0.26846618,  0.20047005,  0.13274658]
    plt.figure()
    plt.plot(ag_list, p, '-o')
    plt.xlabel(r'$a_{\gamma}, b_{\gamma}$',fontsize=labelFontSize)
    plt.ylabel(r'$p(\theta_B -\theta_A \mid \mathbf{y}_B, \mathbf{y}_A)$',fontsize=labelFontSize)
    plt.title(r"6.1d",fontsize=titleFontSize)
    plt.xticks(ag_list, fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)

            
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
    
    
# def part_d(fname=fname):
#     fname += '_d'
#     nsamp = 5000
#     nB = 218
#     thetaB = gamma.rvs(307, scale=1.0/219, size=nsamp)
#     yB_zeros = np.zeros(nsamp, dtype=np.int)
#     yB_ones = np.zeros(nsamp, dtype=np.int)
#     for i in range(nsamp):
#         yB = poisson.rvs(thetaB[i], size=nB)
#         yB_zeros[i] = int(np.bincount(yB)[0])
#         yB_ones[i] = int(np.bincount(yB)[1])
#     plt.figure()
#     plt.plot(range(nsamp), yB_zeros, label='Zero children')
#     plt.xlabel('Sample Number', fontsize=labelFontSize)
#     plt.legend(loc='best')
#     plt.savefig(fname+'_zeros.'+imgFmt, format=imgFmt)
    
    
    
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

    # plt.figure()
    np.histogram2d(yB_zeros, yB_ones)
    plt.xlabel('Y = 0', fontsize=labelFontSize)
    plt.ylabel('Y = 1', fontsize=labelFontSize)
    # plt.legend(loc='best')
    plt.title('4.8d', fontsize=titleFontSize)
    plt.savefig(fname+'_hist2d.'+imgFmt, format=imgFmt)

    
    print('zero={0}, one={1}'.format(a,b))
    

        
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_d()
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
