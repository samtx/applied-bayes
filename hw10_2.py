# HW 9, Problem 10.2

import numpy as np
from scipy.stats import invwishart, multivariate_normal, uniform, norm
from numpy.linalg import inv
# from numpy import dot
# from numpy.linalg import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


fname = 'hw10_2'
imgFmt = 'png'
dpi = 400
labelFontSize = 12
tickFontSize = 12
titleFontSize = 14

print('get data')
# Get data from bluecrab.dat file
f = open('msparrownest.dat','r')
data = np.loadtxt(f)
f.close()

print('finished getting data')
# x = data[:,0]  # first column
# y = data[:,1]  # second column
print(np.mean(data[:,0]))
print(np.mean(data[:,1]))
print(data.shape[0])
y = data[:,0]
n = data.shape[0]

def sum_of_squares(data, theta):
    n = data.shape[0]
    SUM = 0.0
    for i in range(n):
        y = data[i,:]
        # print(data)
        # print(y)
        SUM += np.dot(np.array(y-theta).T, (y-theta))
    return SUM

def proposal(theta_sA,theta_sB,lmbda):
    theta_starA =uniform.rvs(theta_sA-lmbda/2, lmbda)
    theta_starB =uniform.rvs(theta_sB-lmbda/2, lmbda)
    return (theta_starA, theta_starB)

def likelihood(a,b):
    a = np.float(a)
    b = np.float(b)
    n = data.shape[0]
    y = data[:,0]
    x = data[:,1]
    pr = np.zeros(n)
    for i in range(n):
        pr[i] = np.exp(y[i]*(a+b*x[i]))/(1+np.exp(a+b*x[i]))
    return np.prod(pr)
    
def prior(thetaA,thetaB,mu=0.015,std=0.001):
    var = std**2
    pA = norm.pdf(thetaA,mu,var)
    pB = norm.pdf(thetaB,mu,var)
    return (pA, pB)

def fun(theta):
    prior(theta)
    return likelihood(a, b)


def part_c(fname=fname):
    fname = fname + '_a'
    met_iters = int(1e5)  # number of posterior samples
    burn_in = int(1e3)
    lmbda = 0.01
    thetaA = 0.0
    thetaB = 0.0
    sampsA = np.zeros(1)
    sampsB = np.zeros(1)
    for i in range(n):
        
        # get proposal from previous theta
        thetastarA, thetastarB = proposal(thetaA, thetaB, lmbda)
        
        # compute r = f(thetastar)/f(theta)
        rA = fun(thetastarA)/fun(thetaA)
        rB = fun(thetastarA)/fun(thetaB)
        
        # compute acceptance/rejection u
        u = uniform.rvs(0,1,n=2)
        
        if u[0] < rA:
            thetaA = thetastarA
        if u[1] < rB:
            thetaB = thetastarB
            
        sampsA.append(thetaA)
        sampsB.append(thetaB)
        
    
        
    
    

def metropolis(f, proposal, old):
    """
    basic metropolis algorithm, according to the original,
    (1953 paper), needs symmetric proposal distribution.
    """
    u = uniform.rvs(theta-lmbda/2, lmbda)  #
    
    
    if u < r:
      theta_splus1 = theta_star
    else:
        theta_splus1 = theta_s
        
    new = proposal(old)
    alpha = np.min([f(new)/f(old), 1])
    u = np.random.uniform()
    # _cnt_ indicates if new sample is used or not.
    cnt = 0
    if (u < alpha):
        old = new
        cnt = 1
    return old, cnt


    print('begin metropolis')
    for i in range(met_iters):
        if i in (1,2,3,4,5,6,7,8,9,10,1e4,2e4,3e4,4e4,5e4,6e4,7e4,8e4,9e4,10e4):
            print(i)  # print every 10% iterations
        new
        a=inv(Lmda0)
        a=inv(SIGMA[j])
        a=inv(inv(Lmda0)+n*inv(SIGMA[j]))
        muN = np.dot(inv(inv(Lmda0)+n*inv(SIGMA[j])), (np.dot(inv(Lmda0), mu0) + n*np.dot(inv(SIGMA[j]),ybar)))
        LmdaN   =    inv(inv(Lmda0)+n*inv(SIGMA[j]))
        THETA[j] = multivariate_normal.rvs(mean=muN, cov=LmdaN)  # theta posterior

        Sth = sum_of_squares(blue_data, THETA[j])
        Sn = S0 + Sth
        SIGMA[j] = invwishart.rvs(df=nuN, scale=inv(Sn))              # sigma posterior
        print('one gibbs iter')


    np.savez('blue_data_gibbs',THETA=THETA, SIGMA=SIGMA)


    # plt.figure()
    # plot_acf(beta)
    # plt.ylabel(r'$\beta$',fontsize=labelFontSize)
    # plt.savefig(fname+'_beta.'+imgFmt, format=imgFmt)
    # plt.close()

    # plt.figure()
    # plot_acf(c)
    # plt.ylabel(r'$c$',fontsize=labelFontSize)
    # plt.savefig(fname+'_c.'+imgFmt, format=imgFmt)
    # plt.close()

    # beta_samps = beta[-3000:-1]
    # betaCI =  (np.percentile(beta_samps, 2.5), np.percentile(beta_samps, 97.5))
    # p_beta_gt_0 = np.mean(np.greater(beta_samps, 0))
    # print('betaCI=',betaCI)
    # print('p_beta_gt_0=',p_beta_gt_0)

    # for i in range(n):
    #     data = z[:,i]
    #     print(data)
    #     # corr, lags = xcorr(z[:,i].flatten(), norm='coeff')
    #     corr = acf(z[:,i].flatten(),unbiased=True)
    #     # corr = autocorr(z[:,i].flatten())
    #     print('plot acf for i=',i)
    #     print('corr=')
    #     print(corr)
    #     plt.figure()
    #     plt.plot(range(corr.size),corr)
    #     plt.xlabel('lags', fontsize=labelFontSize)
    #     plt.ylabel(r'auto correlation of $z_{'+str(i+1)+'}$',fontsize=labelFontSize)
    #     plt.savefig(fname+'_z'+str(i+1)+'.'+imgFmt, format=imgFmt)
    #     plt.close('all')



# if __name__ == "__main__":
#     if len(sys.argv) == 1:
#         part_a()
#         print('running...')
#     elif len(sys.argv) > 1:

#         if sys.argv[1] == 'a':
#             print('part a...')
#             part_a()
#         elif sys.argv[1] == 'b':
#             print('part b...')
#             part_b()
#         elif sys.argv[1] == 'c':
#             print('part c...')
#             part_c()
#         elif sys.argv[1] == 'd':
#             print('part d...')
#             part_d()
