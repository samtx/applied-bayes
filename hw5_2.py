# HW 5, Problem 5.2

import numpy as np
from scipy.stats import gamma, poisson, bayes_mvs, invgamma, norm
# from scipy.special import gamma as gammafn
# from scipy.special import gammaln
# from scipy.misc import comb
# import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fname = 'hw5_2'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 12
titleFontSize = 14


def part_a(fname=fname):
    fname = fname + '_a'
    nsamps = int(1e5)
    mu0 = 75
    var0 = 100
    params = (1,2,4,8,16,32)
    yavgA = 75.2  # sample mean
    sA = 7.3  # sample std
    yavgB = 77.5
    sB = 8.1
    n = 16.0   # number of data samples
    p = np.zeros(len(params))
    for i, prm in enumerate(params):
        nu0 = prm
        k0 = prm        
        nuN = nu0 + n
        kN = k0 + n
        muN_A = (k0*mu0+n*yavgA)/kN
        s2A = sA**2
        varN_A = 1/nuN*(nu0*var0+(n-1)*s2A+k0*n/kN*(yavgA-mu0)**2)
        muN_B = (k0*mu0+n*yavgB)/kN
        s2B = sB**2
        varN_B = 1/nuN*(nu0*var0+(n-1)*s2B+k0*n/kN*(yavgB-mu0)**2)
        thetaA = np.zeros(nsamps)
        thetaB = np.zeros(nsamps)
        # sig2 = np.zeros([nsamps,2])
        for j in range(nsamps):
            sig2A = 1/gamma.rvs(nuN*0.5, scale=1/(nuN*0.5*varN_A))
            sig2B = 1/gamma.rvs(nuN*0.5, scale=1/(nuN*0.5*varN_B))
            thetaA[j] = norm.rvs(muN_A, sig2A/kN)
            thetaB[j] = norm.rvs(muN_B, sig2B/kN)
        p[i] = np.mean( np.less(thetaA, thetaB) )
    print('Number of samples: {}'.format(nsamps))
    plt.figure()
    plt.bar(params,p)
    plt.xlabel(r'$(\kappa_0,\nu_0)$', fontsize=labelFontSize)
    plt.ylabel(r'$P(\theta_A < \theta_B \mid y_a, y_b)$', fontsize=labelFontSize)
    plt.ylim([0.6,0.68])
    plt.title('5.2',fontsize=titleFontSize)
    # plt.legend(loc='best')
    plt.savefig(fname+'.'+imgFmt, format=imgFmt)

    # plt.figure()
    # np.histogram2d(yB_zeros, yB_ones)
    # plt.xlabel('Y = 0', fontsize=labelFontSize)
    # plt.ylabel('Y = 1', fontsize=labelFontSize)
    # # plt.legend(loc='best')
    # plt.title('4.8d', fontsize=titleFontSize)
    # plt.savefig(fname+'_hist2d.'+imgFmt, format=imgFmt)    

def part_b(fname=fname):
    fname = fname + '_b'
    nsamps = int(1e5)
    mu0 = 5.0
    var0 = 4.0
    k0 = 1.0
    nu0 = 2.0
    theta = np.zeros((nsamps,3))
    sig2 = np.zeros((nsamps,3))
    for i, sch in enumerate(['school1','school2','school3']):
        yavg = np.mean(data[sch])  # sample mean
        s2 = np.var(data[sch])  # sample variance
        std = np.std(data[sch])  # sample standard dev
        n = np.float(data[sch].size)   # number of data samples
        nuN = nu0 + n
        kN = k0 + n
        muN = (k0*mu0+n*yavg)/kN
        varN = 1/nuN*(nu0*var0+(n-1)*s2+k0*n/kN*(yavg-mu0)**2)
        # postVar_mean, postVar_var = invgamma.stats(nuN*0.5, scale=1/(nuN*0.5*varN), moments='mv')
        # postVar_std = postVar_var**0.5

        # print(theta)
        for j in range(nsamps):
            sig2[j,i] = 1/gamma.rvs(nuN*0.5, scale=1/(nuN*0.5*varN))
            theta[j,i] = norm.rvs(muN, sig2[j,i]/kN)
        # print theta
    sch_sets = [(1,2,3),(2,3,1),(2,1,3),(1,3,2),(3,2,1),(3,1,2)]
    p = np.zeros(len(sch_sets))
    # print theta
    # for j in range(nsamps):
    for i, perm in enumerate(sch_sets):
        # if (theta[j,perm[0]-1] < theta[j,perm[1]-1] < theta[j,perm[2]-1]):
        #     p[i] += 1
        p[i] = np.sum(np.logical_and(np.less(theta[:,perm[0]-1],theta[:,perm[1]-1]),np.less(theta[:,perm[1]-1],theta[:,perm[2]-1])))/nsamps
    for i, perm in enumerate(sch_sets):
        print('sch_set:',perm,'  p={:.5f}'.format(p[i]/nsamps))
    
def part_c(fname=fname):
    fname = fname + '_c'
    nsamps = int(1e5)
    mu0 = 5.0
    var0 = 4.0
    k0 = 1.0
    nu0 = 2.0
    theta = np.zeros((nsamps,3))
    sig2 = np.zeros((nsamps,3))
    Ypos = np.zeros((nsamps,3))
    for i, sch in enumerate(['school1','school2','school3']):
        yavg = np.mean(data[sch])  # sample mean
        s2 = np.var(data[sch])  # sample variance
        std = np.std(data[sch])  # sample standard dev
        n = np.float(data[sch].size)   # number of data samples
        nuN = nu0 + n
        kN = k0 + n
        muN = (k0*mu0+n*yavg)/kN
        varN = 1/nuN*(nu0*var0+(n-1)*s2+k0*n/kN*(yavg-mu0)**2)
        # postVar_mean, postVar_var = invgamma.stats(nuN*0.5, scale=1/(nuN*0.5*varN), moments='mv')
        # postVar_std = postVar_var**0.5

        # print(theta)
        for j in range(nsamps):
            sig2[j,i] = 1/gamma.rvs(nuN*0.5, scale=1/(nuN*0.5*varN))
            theta[j,i] = norm.rvs(muN, sig2[j,i]/kN)
            Ypos[j,i] = norm.rvs(theta[j,i], sig2[j,i])
        # print theta
    sch_sets = [(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)]
    p = np.zeros(len(sch_sets))
    # print theta
    # for j in range(nsamps):
    for i, perm in enumerate(sch_sets):
        # if (theta[j,perm[0]-1] < theta[j,perm[1]-1] < theta[j,perm[2]-1]):
        #     p[i] += 1
        p[i] = np.mean( np.less( Ypos[:,perm[0]-1].flatten(), Ypos[:,perm[1]-1].flatten(), Ypos[:,perm[2]-1].flatten() ) )
    for i, perm in enumerate(sch_sets):
        print('sch_set:',perm,'  p={:f}'.format(p[i]))
    
    
def part_d(fname=fname):
    fname += '_d'
    nsamps = int(1e5)
    mu0 = 5.0
    var0 = 4.0
    k0 = 1.0
    nu0 = 2.0
    theta = np.zeros((nsamps,3))
    sig2 = np.zeros((nsamps,3))
    Ypos = np.zeros((nsamps,3))
    for i, sch in enumerate(['school1','school2','school3']):
        yavg = np.mean(data[sch])  # sample mean
        s2 = np.var(data[sch])  # sample variance
        std = np.std(data[sch])  # sample standard dev
        n = np.float(data[sch].size)   # number of data samples
        nuN = nu0 + n
        kN = k0 + n
        muN = (k0*mu0+n*yavg)/kN
        varN = 1/nuN*(nu0*var0+(n-1)*s2+k0*n/kN*(yavg-mu0)**2)
        # postVar_mean, postVar_var = invgamma.stats(nuN*0.5, scale=1/(nuN*0.5*varN), moments='mv')
        # postVar_std = postVar_var**0.5

        # print(theta)
        for j in range(nsamps):
            sig2[j,i] = 1/gamma.rvs(nuN*0.5, scale=1/(nuN*0.5*varN))
            theta[j,i] = norm.rvs(muN, sig2[j,i]/kN)
            Ypos[j,i] = norm.rvs(theta[j,i], sig2[j,i])
    p1 = np.mean(np.logical_and( np.greater(theta[:,0],theta[:,1]), np.greater(theta[:,0],theta[:,2]) ))
    p2 = np.mean(np.logical_and( np.greater( Ypos[:,0], Ypos[:,1]), np.greater( Ypos[:,0], Ypos[:,2]) ) )
    print(p1)
    print(p2)
    

        
    
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
