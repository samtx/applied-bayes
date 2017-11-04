# HW 6, Problem 6.3

import numpy as np
from scipy.stats import norm, truncnorm
# from numpy.linalg import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# for autocorrelation function
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

# from TurbulenceTools import xcorr

fname = 'hw6_3'
imgFmt = 'png'
dpi = 300
labelFontSize = 12
tickFontSize = 12
titleFontSize = 14

# Get data from divorce.dat file
f = open('divorce.dat','r')
data = np.loadtxt(f, dtype=np.int)
f.close()
x = data[:,0]  # first column
y = data[:,1]  # second column


def autocorr(x):
    corr = np.zeros(x.size)
    for t in range(x.size):
        corr[t] = np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))
    return corr

def part_c(fname=fname):
    fname = fname + '_c'
    gibbs_iter = int(10000)  
    burn_in = int(1)
    tau2beta = tau2c = 16.0
    n = x.size   # number of empirical data samples 
    sum_x2 = np.sum(x**2)
    sum_x = np.sum(x)
    
    # Initialize parameter matrices for storing Gibbs iterations
    y1_idx = np.where(y==1)[0]  # samples where y=1
    y0_idx = np.where(y==0)[0]  # samples where y=0
    if y1_idx.size+y0_idx.size != n:
        print('error! C')
    
    beta = np.zeros(gibbs_iter)
    c = np.zeros(gibbs_iter)
    z = np.zeros((gibbs_iter,n))
    
    # Pick initial values for beta, c, z_i
    beta[0] = 0.0
    c[0] = 0.0
    z[0,y1_idx] = 1.0
    z[0,y0_idx] = -1.0
    z0_max = np.max(z[0,y0_idx])
    z1_min = np.min(z[0,y1_idx])
    # print('z0_max=',z0_max)
    # print('z1_min=',z1_min)

    # for i in range(n):
    #     if int(y[i]) == 1:
    #         z[i,0] = -1.0
    #     if int(y[i]) == 0:
    #         z[i,0] = 1.0
    #     else:
    #         print('error! A')
    # print(z[:,0])
    
    # Gibbs iterations
    for j in range(1,gibbs_iter):
        
        if j%500==0:
            print 'j=',j
            
        # calculate beta

        sum_zx = np.sum(z[j,:]*x)
        beta[j] = norm.rvs(loc=sum_zx/(1/(tau2beta**2)+sum_x2), scale=1/(1/(tau2beta**2)+sum_x2)) 
    
        # calculate c
        z0_max = np.max(z[j-1,y0_idx])
        z1_min = np.min(z[j-1,y1_idx])
        # print('y0_idx=',y0_idx)
        # print('z[y0_idx,j-1]=',z[y0_idx,j-1])
        # print('j=',j)
        # print('z0_max=',z0_max)
        # print('z1_min=',z1_min)
        c[j] = truncnorm.rvs(z0_max, z1_min, loc=0.0, scale=tau2c)
        
        # calculate z_i
        for i in y0_idx:
            # if j%500==0:
                # print 'hello 0'
            z[j,i] = truncnorm.rvs(c[j], np.inf, loc=beta[j]*x[i], scale=1.0)
        for i in y1_idx:
            # if j%500==0:
                # print 'hello 1'
            z[j,i] = truncnorm.rvs(-np.inf, c[j], loc=beta[j]*x[i], scale=1.0)

    # Remove burn-in iterations
    # z = np.delete(z,range(burn_in),0)
    # beta = np.delete(beta,range(burn_in),0)
    # c = np.delete(c,range(burn_in),0)

    autocorr, confint = acf(beta, alpha=0.05)
    print(confint)
    print(confint.shape)
    print(z)
    plt.figure()
    plot_acf(beta)
    plt.ylabel(r'$\beta$',fontsize=labelFontSize)
    plt.savefig(fname+'_beta.'+imgFmt, format=imgFmt)
    plt.close()

    plt.figure()
    plot_acf(c)
    plt.ylabel(r'$c$',fontsize=labelFontSize)
    plt.savefig(fname+'_c.'+imgFmt, format=imgFmt)
    plt.close()

    beta_samps = beta[-3000:-1]
    betaCI =  (np.percentile(beta_samps, 2.5), np.percentile(beta_samps, 97.5))
    p_beta_gt_0 = np.mean(np.greater(beta_samps, 0))
    print('betaCI=',betaCI)
    print('p_beta_gt_0=',p_beta_gt_0)

    for i in range(n):
        data = z[:,i]
        print(data)
        # corr, lags = xcorr(z[:,i].flatten(), norm='coeff')
        corr = acf(z[:,i].flatten(),unbiased=True)
        # corr = autocorr(z[:,i].flatten())
        print('plot acf for i=',i)
        print('corr=')
        print(corr)
        plt.figure()
        plt.plot(range(corr.size),corr)
        plt.xlabel('lags', fontsize=labelFontSize)
        plt.ylabel(r'auto correlation of $z_{'+str(i+1)+'}$',fontsize=labelFontSize)
        plt.savefig(fname+'_z'+str(i+1)+'.'+imgFmt, format=imgFmt)
        plt.close('all')

    
    # plt.figure()
    # plt.plot(range(gibbs_iter),c)
    # plt.xlabel('iterations',fontsize=labelFontSize)
    # plt.ylabel(r'$c$',fontsize=labelFontSize)
    # plt.xticks(fontsize=tickFontSize)
    # plt.yticks(fontsize=tickFontSize)
    # plt.savefig(fname+'_c.'+imgFmt, format=imgFmt)

    # ax = fig.add_subplot(222)  # Plot c
    # ax.plot(range(gibbs_iter),c)
    # ax.set_xlabel('iterations',fontsize=labelFontSize)
    # ax.set_ylabel(r'$c$',fontsize=labelFontSize)
    # # ax.set_title(r".1d",fontsize=titleFontSize)
    # # ax.set_xticklabels(fontsize=tickFontSize)
    # # ax.set_yticklabels(fontsize=tickFontSize)
    
    # ax = fig.add_subplot(223)  # Plot z_24
    # ax.plot(range(gibbs_iter),z[23,:])
    # ax.set_xlabel('iterations',fontsize=labelFontSize)
    # ax.set_ylabel(r'$z_{24}$',fontsize=labelFontSize)
    # # ax.set_title(r".1d",fontsize=titleFontSize)
    # # ax.set_xticklabels(fontsize=tickFontSize)
    # # ax.set_yticklabels(fontsize=tickFontSize)
    
    # ax = fig.add_subplot(224)  # Plot z_25
    # ax.plot(range(gibbs_iter),z[24,:])
    # ax.set_xlabel('iterations',fontsize=labelFontSize)
    # ax.set_ylabel(r'$z_{25}$',fontsize=labelFontSize)
    # # ax.set_title(r".1d",fontsize=titleFontSize)
    # # ax.set_xticklabels(fontsize=tickFontSize)
    # # ax.set_yticklabels(fontsize=tickFontSize)
    
    # fig.savefig(fname+'.'+imgFmt, format=imgFmt)

            

        
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        part_c()
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
