# Check if my own solutions to the evolutionarily singular strategy, using T as the trait space, match those in Fig.~2b of Parker et al. (2013)
#
# Reference:
#   Parker, G. A., Lessells, C. M. and Simmons, L. W. (2013). Sperm competition games: a general model
#   for precopulatory male–male competition, Evolution: International Journal of Organic Evolution
#   67(1): 95–109.
#
# I had to use a numerical root finder to get the solution for the risk model with M finite


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar


# parameter values from Fig. 2
# ===

r = 1; D = 1; R = 10; M = 2 


# functions for the steady states
# ===


# risk model
# ---

# M finite

# To find T*, the root of this equation has to be solved numerically.
# This very long equation was obtained using symbolic maths software Sage.
TR_M = lambda T_h, n_h, a, M: -M**2*R*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)*r**2 - M**2*T_h**(2*a)*T_h**(a - 1)*T_h**(-a + 1)*n_h*(n_h-1)**2*r**2 + \
        M*R*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)**2*r**2 - 2*M**2*R*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)*r - \
        M**2*T_h**(2*a)*T_h**(a - 1)*T_h**(-a + 1)*n_h*(n_h-1)**2*r + M*R*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)**2*r + M**2*R*T_h**(a - 1)*T_h**a*a*n_h*r**2 - \
        M*R*T_h**(2*a - 1)*a*n_h*(n_h-1)**2*r**2 - M**2*R*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1) + 2*M**2*R*T_h**(a - 1)*T_h**a*a*n_h*r + \
        M*R*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)*r - M*R*T_h**(2*a - 1)*a*n_h*(n_h-1)**2*r + M*R*T_h**(2*a - 1)*a*n_h*(n_h-1)*r**2 + \
        M**2*R*T_h**(a - 1)*T_h**a*a*n_h + M*R*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1) + M*R*T_h**(2*a - 1)*a*n_h*(n_h-1)*r - \
        M*R*T_h**(2*a - 1)*a*n_h*r**2 -2*M*R*T_h**(2*a - 1)*a*n_h*r - M*R*T_h**(2*a - 1)*a*n_h - \
        (M**2*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1)**2*r**2 - M*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1)**2*r**2 + \
        M**2*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1) - M*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1))*T_h**2 + \
        (M**2*R*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1)**2*r**2 + \
        M**2*T_h**(3*a)*T_h**(a - 1)*T_h**(-a + 1)*T_h**(-a - 1)*n_h*(n_h-1)**2*r**2 - M*R*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1)**2*r**2 + \
        M**2*R*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1) + M**2*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)*r**2 - M*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)**2*r**2 + \
        M**2*T_h**(3*a)*T_h**(a - 1)*T_h**(-a + 1)*T_h**(-a - 1)*n_h*(n_h-1) - M*R*T_h**(2*a)*T_h**(a - 1)*T_h**(-a - 1)*a*n_h*(n_h-1) + \
        2*M**2*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)*r - M*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)**2*r - M**2*T_h**(a - 1)*T_h**a*a*n_h*r**2 + \
        M*T_h**(2*a - 1)*a*n_h*(n_h-1)**2*r**2 + M**2*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1) - 2*M**2*T_h**(a - 1)*T_h**a*a*n_h*r - \
        M**2*T_h**(a - 1)*T_h**a*n_h*(n_h-1)*r - M*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1)*r + M*T_h**(2*a - 1)*a*n_h*(n_h-1)**2*r - \
        M*T_h**(2*a - 1)*a*n_h*(n_h-1)*r**2 - M**2*T_h**(a - 1)*T_h**a*a*n_h - M**2*T_h**(a - 1)*T_h**a*n_h*(n_h-1) - M*T_h**(a - 1)*T_h**a*a*n_h*(n_h-1) - \
        M*T_h**(2*a - 1)*a*n_h*(n_h-1)*r + M*T_h**(2*a - 1)*a*n_h*r**2 + 2*M*T_h**(2*a - 1)*a*n_h*r + M*T_h**(2*a - 1)*a*n_h)*T_h

# M inifite

TRoo = lambda n_h, a: ((R*a*n_h**2 - 3*R*a*n_h + 3*R*a)*r**2 + R*a - 2*(R*a*n_h - 2*R*a)*r)/((a*n_h**2 - 3*a*n_h + 3*a)*r**2 - ((2*a + 1)*n_h - n_h**2 - 4*a)*r + a)


# intensity model
# ---

# M finite

TI_M = lambda n_h, a, M: (M - 1)*R*a/((M - 1)*a + M*n_h - M)

# M infinite

TIoo = lambda n_h, a: R*a / (a + n_h - 1)


# plot the results
# ===

nhVS = np.linspace(1.01, 1.99, 10) # special points for the one that needs a numerical solution
nhVI = np.linspace(2, 10, 50)
nhVR = np.linspace(1, 2, 50)
aV = [0.2, 1, 5]; lsV = ['dashed', 'solid', 'dotted']

for a, ls in zip(aV, lsV):

    TI_MV = np.array([ TI_M(n_h, a, M) for n_h in nhVI ])
    yaxis = TI_MV/R
    plt.plot(nhVI, yaxis, ls=ls, color='black', label=r'$a = ' + str(a) + '$')

    TIooV = np.array([ TIoo(n_h, a) for n_h in nhVI ])
    yaxis = TIooV/R
    plt.plot(nhVI, yaxis, ls=ls, color='black')

    # this one needs a numerical root finder to solve
    TR_MV = np.array([ root_scalar(lambda T_h: TR_M(T_h, n_h, a, M), bracket=[1e-6, R-1e-6], method='brentq').root 
        for n_h in nhVS ])
    yaxis = TR_MV/R
    plt.plot(nhVS, yaxis, ls=ls, color='brown')

    TRooV = np.array([ TRoo(n_h, a) for n_h in nhVR ])
    yaxis = TRooV/R
    plt.plot(nhVR, yaxis, ls=ls, color='black')

plt.ylim((1,0))
plt.axvline(2, ls='dashed', lw=3, color='black')
plt.grid(True)
plt.legend(loc='best')
plt.xscale("log")
plt.xlabel(r'mean no. of matings, $n^*$')
plt.title(r'analytically obtained singular strategies')
plt.ylabel(r'pre-copulatory expenditure, $T^*/R$')
#plt.show()
plt.tight_layout()
plt.savefig('replicate_2b_rootfinding.pdf')
plt.close()
