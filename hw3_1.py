# HW 2, #3.1

from scipy.stats import bernoulli, beta
from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt

# Take 100 samples from bernoulli(theta)
# assume sum(y)=57
# Determine probabilities of theta =[0.0, 0.1, ..., 0.9, 1.0] and plot them
# Compute Pr(Sum(y)=57 | theta)

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

def part_b():
    probs = np.zeros(11)
    THETAS = np.arange(0,1.1,0.1)
    for i, theta in enumerate(THETAS):
        probs[i] = prob_sum57(theta)
    plt.plot(THETAS,probs)
    plt.xlabel(r'$\theta$')
    plt.title('3.1b')
    # plt.ylabel(r'$Pr(\Sum y_i = 57 \mid \theta)$')
    plt.show()

def part_c():
    probs = np.zeros(11)
    THETAS = np.arange(0,1.1,0.1)
    for i, theta in enumerate(THETAS):
        probs[i] = prob_sum57(theta)*prob_prior(theta)/bayes_constant()
    plt.plot(THETAS,probs)
    plt.xlabel(r'$\theta$')
    plt.title('3.1c')
    # plt.ylabel(r'$Pr(\Sum y_i = 57 \mid \theta)$')
    plt.show()

def part_d():
    THETAS = np.linspace(0,1,500)
    probs = np.zeros(THETAS.shape)
    for i, theta in enumerate(THETAS):
        probs[i] = prob_sum57(theta)*prob_prior(theta)/bayes_constant()
    plt.plot(THETAS,probs)
    plt.xlabel(r'$\theta$')
    plt.title('3.1d')
    # plt.ylabel(r'$Pr(\Sum y_i = 57 \mid \theta)$')
    plt.show()

def part_e():
    a = 1 + 57
    b = 1 + 100 - 57

    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)

    # plt.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
    # plt.title('beta pdf')

    rv = beta(a, b)
    plt.plot(x, rv.pdf(x))
    plt.title('frozen beta pdf')
    plt.show()

if __name__ == "__main__":
    # part_b()
    # part_c()
    # part_d()
    part_e()
    # main()
