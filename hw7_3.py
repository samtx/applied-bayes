# HW 7, Problem 7.3

import numpy as np
from scipy.stats import norm, truncnorm, invwishart, multivariate_normal
from numpy.linalg import inv
# from numpy import dot
# from numpy.linalg import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


fname = 'hw7_3'
imgFmt = 'png'
dpi = 400
labelFontSize = 12
tickFontSize = 12
titleFontSize = 14

# Get data from bluecrab.dat file
f = open('bluecrab.dat','r')
blue_data = np.loadtxt(f)
f.close()

# Get data from orangecrab.dat file
f = open('orangecrab.dat','r')
orange_data = np.loadtxt(f)
f.close()

# x = data[:,0]  # first column
# y = data[:,1]  # second column

def sum_of_squares(data, theta):
    n = data.shape[0]
    SUM = 0.0
    for i in range(n):
        y = data[i,:]
        # print(data)
        # print(y)
        SUM += np.dot(np.array(y-theta).T, (y-theta))
    return SUM

def part_a(fname=fname):
    fname = fname + '_a'
    nsamps = int(1e4)  # number of posterior samples
    gibbs_iter = int(1e5)
    burn_in = int(3e3)

    # Blue Crab data
    mu0 = np.mean(blue_data, axis=0)            # sample mean
    Lmda0 = np.cov(blue_data, rowvar=False)     # sample covariance matrix
    S0 = Lmda0
    nu0 = 4.0
    n = blue_data.shape[0]                      # number of empirical samples
    nuN = nu0 + n
    ybar = mu0
    eye3d = np.array([ [[ 1., 0.],[0.,1.]] ])
    SIGMA = np.repeat(eye3d,nsamps,axis=0)      # initial values
    THETA = np.zeros((nsamps,1,2))

    for i in range(gibbs_iter):

        if i in (1,2,3,4,5,6,7,8,9,10,1e4,2e4,3e4,4e4,5e4,6e4,7e4,8e4,9e4,10e4):
            print(i)  # print every 10% iterations

        for j in range(nsamps):
            a=inv(Lmda0)
            a=inv(SIGMA[j])
            a=inv(inv(Lmda0)+n*inv(SIGMA[j]))
            muN = np.dot(inv(inv(Lmda0)+n*inv(SIGMA[j])), (np.dot(inv(Lmda0), mu0) + n*np.dot(inv(SIGMA[j]),ybar)))
            LmdaN   =    inv(inv(Lmda0)+n*inv(SIGMA[j]))
            THETA[j] = multivariate_normal.rvs(mean=muN, cov=LmdaN)  # theta posterior

            Sth = sum_of_squares(blue_data, THETA[j])
            Sn = S0 + Sth
            SIGMA[j] = invwishart.rvs(df=nuN, scale=inv(Sn))              # sigma posterior



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



if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_a()
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
