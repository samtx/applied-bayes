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
        
    # # print(p)
    p = [0.38144128,  0.33382475,  0.26846618,  0.20047005,  0.13274658]
    plt.figure()
    plt.plot(ag_list, p, '-o')
    plt.xlabel(r'$a_{\gamma}, b_{\gamma}$',fontsize=labelFontSize)
    plt.ylabel(r'$E(\theta_B -\theta_A \mid \mathbf{y}_B, \mathbf{y}_A)$',fontsize=labelFontSize)
    plt.title(r"6.1d",fontsize=titleFontSize)
    plt.xticks(ag_list, fontsize=tickFontSize)
    plt.yticks(fontsize=tickFontSize)
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)

            

        
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_d()
    else:
        part_d()

