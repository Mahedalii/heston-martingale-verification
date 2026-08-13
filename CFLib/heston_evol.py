#!/usr/bin/env python3

from math import *
import numpy as np
from time import time
try:
    from cir_obj import cir_obj
    from cir_evol import cir_euler, QT_cir_evol, fe_cir_evol
except ModuleNotFoundError:
    from CFLib.cir_obj import cir_obj
    from CFLib.cir_evol import cir_euler, QT_cir_evol, fe_cir_evol
# -----------------------------------------------------

def __mc_heston__( rand, vol, intVol, cir, rho, tf, N  ):

    '''
    @parms intVol: volatility integral trajectory
    @parms cir   : CIR object
    @parms rho   : correlation between vol and underlying innovations
    @parms tf    : schedule of the underlying trajectory
    @parms N     : number of underlying trajectories
    '''

    # length of the volatility trajectory
    # (including initial point)
    L   = len(intVol)
    th  = cir.theta
    k   = cir.kappa
    eta = cir.sigma
    nu  = vol
    I   = intVol

    # underlying trajectorie
    S  = np.ndarray(shape = (L, N), dtype=np.double ) # S[N, L] in fortran matrix notation

    xi = rand.normal( loc = 0.0, scale = 1.0, size=(L-1, N))

    # prime with So the starting value of each trajectory
    S[0] = 1.0

    for n in range(1,L):
        DI   = I[n] - I[n-1]
        Dt   = tf[n]-tf[n-1]
        X    = -.5 * DI + (rho/eta)*( nu[n] - nu[n-1] - k*( th*Dt - DI) ) + sqrt((1. - rho*rho)*DI)*xi[n-1]
        try:
            S[n] = S[n-1]*np.exp(X)
        except ValueError as e:
           print(f"S[n]: {S[n].shape}, X: {X.shape}") 
           raise e

    return S

# ----------------------------------------------------

def mc_heston( rand, vol, intVol, cir, rho, tf, N  ):
    return __mc_heston__( rand, vol, intVol, cir, rho, tf, N  )

def heston_trj    ( rand
                  , heston
                  , tf     # SChedule for the output result
                  , dt     # step per vol inegration
                  , NV     # number of vol trajectories
                  , NS=1   # number of S trajectory per vol trajectory
                  ):

    cir = heston.cir
    rho = heston.rho
    #
    # Computes NV Cir trajectories
    # vol and Ivol have the geometry ( Nt+1, NV)
    # vol[n] = r(t_n)
    # Ivol[n] = \int_0^{t_n} r(s) ds
    #
    vol, Ivol = QT_cir_evol( rand, cir, tf, dt, NV)
    vol = vol.T
    Ivol = Ivol.T

    S = np.zeros( shape=(len(tf),NV,NS) )
    for n in range(NV):
        s = mc_heston( rand, vol[n], Ivol[n], cir, rho, tf, NS )
        S[:,n,:] = s
    
    res = S if NS > 1 else np.reshape(S, shape=(len(tf),-1))
    return res
