import math
import numpy as np
from math import *

def pr_x_lt_w ( model, Xc, vW, off, t):

    '''
        Digital option according to the SINC algorithm
        the acronym stands for P(x < w )
        @params model: the model we use to compute the probability
        @params Xc   : the cutoff imposed to the pdf
        @params vW   : the log of monenyness K/Fw
        @params off  : states whether we are workingin the risk neutral measure ( off = 0 )
                       or in the M(t,T) measure ( off = -i/2PI )
    '''

    m = 1
    vTot = 0.0
    while True:
        c_k    = 2*math.pi*( m/(2*Xc) + off )
        c_phi  = model.cf(c_k, t)
        vTh    = math.pi * m * vW/Xc;
        vDelta = (np.cos(vTh)*c_phi.imag - np.sin(vTh)*c_phi.real)/m; 
        vTot  += vDelta
        if np.fabs(vDelta/vTot).max() < 1.e-08: break
        m += 2
    return .5 - 2.*vTot/math.pi
# --------------------------------------------------------------------

'''
Computes the price of vanilla options for the model 'model'

@params model   : model object that admits the method 'cf', returning the complex array
                  describing the characteristic function of the model
@params vSTrike : a scalar or  an np.array of strikes for the options
@params T       : the maturity of the options involved
@params Xc      : The Xc associated to this maturity
@returns        : a dictionary in the form  {"put": v_1, "call":  v_2, "pCn": v_3, "pAn": v_4}
                  where v_1, ..., v_4 have the same geometry as vStrike
'''
def ft_opt(model, vStrike, T, Xc):

    if T < 1.0e-08:
        vPut  = np.maximum( vStrike - 1., 0)
        vCall = np.maximum( 1. - vStrike, 0)
        vCn  = np.where(vStrike > 1., 1., 0)
        vAn  = np.where(vStrike > 1., 1., 0)
    else:

        vW       = np.log(vStrike)

        #
        # cash or nothing option in the terminal measure
        #
        off = complex(0.0, 0.0)
        vCn = pr_x_lt_w( model, Xc, vW, off, T)

        #
        # cash or nothing option in the S(T) measure
        #
        off = complex(0.0, -1/(2*math.pi))
        vAn = pr_x_lt_w( model, Xc, vW, off, T)

        
        vPut  = vStrike*vCn - vAn; 
        vCall = vPut + (1. - vStrike);

    return {"put": vPut, "call":  vCall, "pCn": vCn, "pAn": vAn}
# --------------------------------------------------------------------
