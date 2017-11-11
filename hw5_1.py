# HW 5, Problem 5.1

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

fname = 'hw5_1'
imgFmt = 'png'
dpi = 300
labelFontSize = 16
tickFontSize = 12
titleFontSize = 14

# def lgamma(x):
#     return gammafn(np.log(x))

# def combln(N,k):
#     return gammaln(N+1)-gammaln(N-k+1)-gammaln(k+1)

school1_data = np.array([2.11,9.75,13.88,11.3,8.93,15.66,16.38,4.54,8.86,11.94,12.47,11.11,11.65,14.53,9.61,7.38,3.34,9.06,9.45,5.98,7.44,8.5,1.55,11.45,9.73])
school2_data = np.array([0.29,1.13,6.52,11.72,6.54,5.63,14.59,11.74,9.12,9.43,10.64,12.28,9.5,0.63,15.35,5.31,8.49,3.04,3.77,6.22,2.14,6.58,1.11])
school3_data = np.array([4.33,7.77,4.15,5.64,7.69,5.04,10.01,13.43,13.63,9.9,5.72,5.16,4.33,12.9,11.27,6.05,0.95,6.02,12.22,12.85])
data = {
    'school1': school1_data,
    'school2': school2_data,
    'school3': school3_data
}

# sum_bach = np.sum(data_bach)
# sum_nobach = np.sum(data_nobach)

# n_bach = data_bach.size
# n_nobach = data_nobach.size

# print("Bach:    sum={0:3f}  n={1:3d}".format(np.sum(data_bach), data_bach.size))
# print("No Bach: sum={0:3f}  n={1:3d}".format(np.sum(data_nobach), data_nobach.size))


def part_a(fname=fname):
    fname = fname + '_a'
    nsamps = int(1e6)
    mu0 = 5.0
    var0 = 4.0
    k0 = 1.0
    nu0 = 2.0
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
        theta = np.zeros(nsamps)
        sig2 = np.zeros(nsamps)
        for j in range(nsamps):
            sig2[j] = 1/gamma.rvs(nuN*0.5, scale=1/(nuN*0.5*varN))
            theta[j] = norm.rvs(muN, sig2[j]/kN)
        postTheta_mean = np.mean(theta)
        postThetaCI = (np.percentile(theta, 2.5), np.percentile(theta, 97.5))
        sig = sig2**0.5
        postVar_mean = np.mean(sig)
        postVarCI = (np.percentile(sig, 2.5), np.percentile(sig, 97.5)) 
        print('Number of samples: {}'.format(nsamps))
        print('school={:d} n={:d} yavg={:.4f} s2={:.5f} std={:.5f}'.format(i+1,int(n),yavg,s2,std))
        print('Theta_mean={:}    95 CI = [{:}]'.format(postTheta_mean,postThetaCI))
        print('  Var_mean={:}    95 CI = [{:}]'.format(postVar_mean, postVarCI))
        

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
