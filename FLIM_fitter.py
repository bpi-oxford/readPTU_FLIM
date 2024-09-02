# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:02:42 2024

@author: narai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls , lsq_linear
from scipy.optimize import minimize as minimize_s
# from lmfit import minimize, Parameters #,report_fit, fit_report, report_errors,

#%%


def Convol(irf, x):
    """
    Convolves the instrumental response function (irf) with the decay function x.
    Assumes periodicity with the length of x.

    Parameters:
        irf : array-like
            Instrumental response function.
        x : array-like
            Decay function.

    Returns:
        y : array-like
            Result of the convolution, assuming periodicity.
    """
    
    irf = np.array(irf).flatten()
    x = np.array(x)
    
    mm = np.mean(irf[-10:]) # background estimate
    
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    p = x.shape[0]
    n = len(irf)
    
    if p > n:
        irf = np.concatenate([irf, mm * np.ones(p - n)])
    else:
        irf = irf[:p]
    
    y = np.real(np.fft.ifft(np.fft.fft(irf)[:, None] * np.fft.fft(x, axis=0), axis=0))
    
    t = (np.arange(n) % p)
    y = y[t, :]
    
    return y.squeeze()

def IRF_Fun(p,t,pic=None):
    """
    Computes the following model IRF:
    IRF(t_0, w, a, dt, T1, T2, t) = 
           1/Q [ exp(- t'^2/(2 w^2) +
                 a P H(t'') exp(- t''/T1) (1 - exp(-t''/T2)) ] 
     
    where:
      t'  = t - t_0
      t'' = t - t_0 - dt
      Q   = sqrt(2 pi w^2) + a T1(1+T1/T2)^{T2/T1}
      P   = (1-T2/T1)(1+T1/T2)^{T2/T1}
    
    Parameters:
        p : array-like
            Parameters of the model [t_0, w1, T1, T2, a, b, dt]
        t : array-like
            Time vector
        pic : int, optional
            If provided, determines the type of plot (1 for linear, other for log).
    
    Returns:
        z : array-like
            Computed IRF values
            
    (see: Walther, K.A. et al, Mol. BioSyst. (2011) doi:10.1039/c0mb00132e)

    """

    t = np.array(t).reshape(-1)  # Ensure t is a column vector
    p = np.array(p).reshape(-1)  # Ensure p is a column vector

    t_0 = p[0]
    w1  = p[1]
    T1  = p[2]
    T2  = p[3]
    a   = p[4]
    b   = p[5]
    dt  = p[6]

    t1 = t - t_0
    t2 = t - t_0 - dt
    t3 = t + t_0

    H = np.ones(t.shape)
    H[t < (t_0 + dt)] = 0

    G = np.zeros(t.shape)
    G[t < (t_0 - dt)] = 1

    IRF = np.array([
        np.exp(-t1**2 / (2 * w1)),
        G * np.exp(t3 / T1),
        H * (np.exp(-t2 / T1) * (1 - np.exp(-t2 / T2)))
    ])

    IRF = np.ones((len(t), 1)) * np.array([1, b, a]) * IRF.T / np.ones((len(t), 1)) @ np.sum(IRF, axis=1, keepdims=True).T

    tm = 0.5 * np.max(IRF[:, 0])

    IRF[IRF[:, 1] > tm, 1] = tm
    IRF[IRF[:, 2] > tm, 2] = tm

    IRF[np.isnan(IRF)] = 0
    IRF = np.sum(IRF, axis=1) / np.sum(np.sum(IRF))
    IRF[IRF < 0] = 0

    tmp, t0 = np.max(IRF), np.argmax(IRF)
    tmp = IRF[:t0]
    tmp = np.diff(tmp)
    tmp[tmp < 0] = 0
    tmp = np.concatenate(([0], np.cumsum(tmp)))
    IRF[:t0] = tmp
    z = np.sum(IRF) / np.sum(np.sum(IRF))

    if pic is not None:
        if pic == 1:
            plt.plot(t, z, 'r')
            plt.draw()
        else:
            plt.semilogy(t, z)
            plt.draw()

    return z    


def TCSPC_Fun(p, t, y=None, para=None):
    """
    Fits the following model function to TCSPC-Data
    
    Parameters:
        p : array-like
            Parameters of the model
        t : array-like
            Time vector
        y : array-like, optional
            Data to fit the model to
        para : array-like, optional
            Additional parameters to incorporate into `p`
    
    Returns:
        err : float
            Error metric for the fit
        c : array-like
            Coefficients resulting from the fit
        zz : array-like
            Matrix containing the model functions
        z : array-like
            Fitted data points
    """
    
    p = np.array(p).flatten()

    if para is not None:
        para = np.array(para).flatten()
        n = len(para)
        if n > 6:
            p = np.concatenate([para, p])
        else:
            p = np.concatenate([p[:7-n], para, p[7-n:]])

    if y is None or len(t) < len(y):
        c = t
        t = np.array(y).flatten()
        
        nex = len(p) - 7
        tau = p[7:]
        
        IRF = IRF_Fun(p[:7], t)
        
        zz = np.zeros((len(t), nex + 1))
        zz[:, 0] = np.ones_like(t)
        
        for i in range(nex):
            tmp = Convol(IRF, np.exp(- (t - p[0]) / tau[i]) / tau[i])
            zz[:, i+1] = tmp[:len(t)]
        
        err = np.zeros((len(t), c.shape[1]))
        for j in range(c.shape[1]):
            err[:, j] = zz @ c[:, j]

    else:
        t = np.array(t).flatten()
        y = np.array(y)
        
        m, n = y.shape
        if m < n:
            y = y.T
            m, n = y.shape
        
        valid_idx = np.isfinite(np.sum(y, axis=1))
        t = t[valid_idx]
        y = y[valid_idx, :]
        
        nex = len(p) - 7
        
        IRF = IRF_Fun(p[:7], t)
        tau = p[7:]
        t1 = t - p[0]
        
        zz = np.zeros((len(t), nex + 1))
        zz[:, 0] = np.ones_like(t)
        
        for i in range(nex):
            tmp = Convol(IRF, np.exp(- t1 / tau[i]) / tau[i])
            zz[:, i+1] = tmp[:len(t)]
        
        c = np.zeros((zz.shape[1], n))
        z = np.zeros_like(y)
        
        for j in range(n):
            # res = lsq_linear(zz, y[:, j], bounds=(0, np.inf))
            # c[:, j] = res.x
            c[:, j], _ = nnls(zz, y[:, j])
            z[:, j] = zz @ c[:, j]
        
        err = np.sum(np.sum((y - z) ** 2 / (10 + np.abs(z))))
    
    return err, c, zz, z


def Calc_mIRF(head, tcspc):
    """
    Calculate the mIRF (instrumental response function) for TCSPC data.
    
    Parameters:
        head : object or dict
            Contains metadata including Resolution and SyncRate.
        tcspc : ndarray
            TCSPC data array.
    
    Returns:
        IRF : ndarray
            Calculated mIRF array.
    """

    maxres = np.max(head['Resolution'])
    Resolution = max([maxres, 0.032])
    # Pulse = 1e9 / head['SyncRate']

    tau = Resolution * (np.arange(1, tcspc.shape[1] + 1) - 0.5)
    IRF = np.zeros_like(tcspc)
    nex = 2

    _, t0_idx = np.max(tcspc, axis=1), np.argmax(tcspc, axis=1)
    t0 = tau[min(t0_idx.min(), len(tau) - 1)]

    w1 = 0.03**2
    T1 = 0.050
    T2 = 0.10
    a = 0.005
    b = 0.1
    dt = 0.0

    for PIE in range(tcspc.shape[2]):

        p = np.array([t0, w1, T1, T2, a, b, dt, 1, 2])
        pl = np.array([t0 - 2.5, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, -0.3] + [0] * nex)
        pu = np.array([t0 + 2.5, 1, 1, 1, 0.01, 0.5, 0.5] + [10] * nex)

        tc = np.sum(tcspc[:, :, PIE], axis=1)
        ord = np.argsort(tc)[::-1]

        ch = 0
        ind = ord[ch]
        y = tcspc[ind, :, PIE]

        err = np.zeros(10)

        p_array = np.zeros((len(p), 10))

        for casc in range(10):
            s = err.argmin()
            r0 = p_array[:, s]
            for sub in range(10):
                rf = r0 * (2 ** (1.1 * (np.random.rand(len(r0)) - 0.5) / casc))
                rf = np.clip(rf, pl, pu)
                res = minimize_s(lambda x: TCSPC_Fun(x, tau, y), rf, bounds=list(zip(pl, pu)))
                p_array[:, sub] = res.x
                err[sub] = TCSPC_Fun(p_array[:, sub], tau, y)

        err1 = err.min()
        p1 = np.mean(p[:, err == err1], axis=1)
        _, c1, _, tmp1 = TCSPC_Fun(p1, tau, y)

        IRF[ind, :, PIE] = IRF_Fun(p1[:7], tau)

        para = p1[1:7]
        p = np.concatenate([[p1[0]], p1[7:]])
        pl = np.array([0] + [0] * nex)
        pu = np.array([3] + [10] * nex)

        for ch in range(1, tcspc.shape[0]):
            ind = ord[ch]
            y = tcspc[ind, :, PIE]

            err = np.zeros(10)
            p_array = np.zeros((len(p), 10))

            for casc in range(10):
                s = err.argmin()
                r0 = p_array[:, s]
                for sub in range(10):
                    rf = r0 * (2 ** (1.05 * (np.random.rand(len(r0)) - 0.5) / casc))
                    rf = np.clip(rf, pl, pu)
                    res = minimize_s(lambda x: TCSPC_Fun(x, tau, y, para), rf, bounds=list(zip(pl, pu)))
                    p_array[:, sub] = res.x
                    err[sub] = TCSPC_Fun(p_array[:, sub], tau, y, para)

            err1 = err.min()
            p1 = np.mean(p_array[:, err == err1], axis=1)
            _, c1, _, tmp1 = TCSPC_Fun(p1, tau, y, para)


            IRF[ind, :, PIE] = IRF_Fun(np.concatenate([[p1[0]], para, p1[1:]]), tau)

    IRF[IRF < 0] = 0

    return IRF