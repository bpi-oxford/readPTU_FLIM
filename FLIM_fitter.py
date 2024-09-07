# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:02:42 2024

@author: narai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls , lsq_linear
from scipy.optimize import minimize as minimize_s
from scipy.linalg import lstsq
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

def IRF_Fun(p,t,pic = None):
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

    IRF = (np.ones((len(t), 1)) * np.array([1, b, a]) * IRF.T) / (np.ones((len(t), 1)) @ (np.sum(IRF, axis=1, keepdims=True).T))

    tm = 0.5 * np.max(IRF[:, 0])

    IRF[IRF[:, 1] > tm, 1] = tm
    IRF[IRF[:, 2] > tm, 2] = tm

    IRF[np.isnan(IRF)] = 0
    IRF = (np.sum(IRF, axis=1) / np.sum(np.sum(IRF)))[:,np.newaxis]
    IRF[IRF < 0] = 0

    t0 = np.argmax(IRF)
    tmp = IRF[:t0]
    tmp = np.diff(tmp)
    tmp[tmp < 0] = 0
    tmp = np.concatenate(([0], np.cumsum(tmp)))
    IRF[:t0] = tmp
    z = np.sum(IRF,axis=1) / np.sum(np.sum(IRF))

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
        tauT = p[7:]
        
        IRF = IRF_Fun(p[:7], t)
        
        zz = np.zeros((len(t), nex + 1))
        zz[:, 0] = 1
        
        for i in range(nex):
            tmp = Convol(IRF, np.exp(- (t - p[0]) / tauT[i]) / tauT[i])
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
        tauT = p[7:]
        t1 = t - p[0]
        
        zz = np.zeros((len(t), nex + 1))

        zz[:, 0] = 1
        
        for i in range(nex):
            tmp = Convol(IRF, np.exp(- t1 / tauT[i]) / tauT[i])
            # print( tmp.shape)
            zz[:, i+1] = tmp[:len(t)]
        
        c = np.zeros((zz.shape[1], n))
        z = np.zeros_like(y)
        
        # if np.isnan(np.sum(zz)) or np.isinf(np.sum(zz)):
            # print('nan / inf detected in IRF')
        for j in range(n):
            # res = lsq_linear(zz, y[:, j], bounds=(0, np.inf))
            # c[:, j] = res.x
            try:
                if not (np.isnan(np.sum(zz)) or np.isnan(np.sum(y[:,j]))):
                    c[:, j],_ = nnls(zz, y[:, j])
                    z[:, j] = zz @ c[:, j]
                else:
                    c[:,j] = 0
                    z[:, j] = 0
            except:
                print("fitting error: ",j)
                c[:,j] = 0
                z[:, j] = 0
                    
        
        # err = np.sum(np.sum((y - z) ** 2 / (10^-3 + np.abs(z))))
        # if np.isnan(err):
        #     err = 10^200
        err = np.sum((y-z)**2)  
        # print(err)
    
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

    np.seterr(divide='ignore', over='ignore', invalid='ignore')
    
    maxres = np.max(head['MeasDesc_Resolution']*1e9)
    Resolution = max([maxres, 0.1])
    # Pulse = 1e9 / head['SyncRate']

    tau = Resolution * (np.arange(tcspc.shape[1] ) + 0.5)
    IRF = np.zeros(tcspc.shape)
    nex = 2

    t0_idx = np.argmax(tcspc, axis=1)
    t0 = tau[min(t0_idx.min(), len(tau) - 1)]

    w1 = 0.03**2
    T1 = 0.050
    T2 = 0.10
    a = 0.005
    b = 0.1
    dt = 0.0

    for PIE in range(tcspc.shape[2]):

        p = np.array([t0, w1, T1, T2, a, b, dt, 1, 2])
        pl = np.array([t0 - 2.5, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, -0.3] + [0.1] * nex)
        pu = np.array([t0 + 2.5, 1, 1, 1, 0.01, 0.5, 0.5] + [10] * nex)

        tc = np.sum(tcspc[:, :, PIE], axis=1)
        ord = np.argsort(tc)[::-1]

        ch = 0
        ind = ord[ch]
        y = tcspc[ind, :, PIE][:,np.newaxis]

        err = np.zeros(10)

        p_array = np.zeros((len(p), 10))

        for casc in range(10):
            if casc==0:
                r0 = p
            else:
                s = err.argmin()
                r0 = p_array[:, s]
                
            for sub in range(10):
                rf = r0 * (2 ** (1.1 * (np.random.rand(len(r0)) - 0.5) / (casc+1)))
                rf = np.clip(rf, pl, pu)
                res = minimize_s(lambda x: TCSPC_Fun(x, tau, y.astype(np.float64))[0], rf, bounds=list(zip(pl, pu)))
                p_array[:, sub] = res.x
                err[sub] = TCSPC_Fun(p_array[:, sub], tau, y.astype(np.float64))[0]

        err1 = err.min()
        p1 = np.mean(p_array[:, err == err1], axis=1)
        _, c1, _, tmp1 = TCSPC_Fun(p1, tau, y.astype(np.float64))

        IRF[ind, :, PIE] = IRF_Fun(p1[:7], tau)

        para = p1[1:7]
        p = np.concatenate([[p1[0]], p1[7:]])
        pl = np.array([0] + [.1] * nex)
        pu = np.array([3] + [10] * nex)

        for ch in range(0, tcspc.shape[0]):
            ind = ord[ch]
            y = tcspc[ind, :, PIE][:,np.newaxis]
            print(len(y))
            err = np.zeros(10)
            p_array = np.zeros((len(p), 10))

            for casc in range(10):
                if casc==0:
                    r0 = p
                else:
                    s = err.argmin()
                    r0 = p_array[:, s]
                for sub in range(10):
                    rf = r0 * (2 ** (1.05 * (np.random.rand(len(r0)) - 0.5) / (casc+1)))
                    rf = np.clip(rf, pl, pu)
                    res = minimize_s(lambda x: TCSPC_Fun(x, tau, y.astype(np.float64), para)[0], rf, bounds=list(zip(pl, pu)))
                    p_array[:, sub] = res.x
                    err[sub] = TCSPC_Fun(p_array[:, sub], tau, y.astype(np.float64), para)[0]

            err1 = err.min()
            p1 = np.mean(p_array[:, err == err1], axis=1)
            _, c1, _, tmp1 = TCSPC_Fun(p1, tau, y.astype(np.float64), para)


            IRF[ind, :, PIE] = IRF_Fun(np.concatenate([[p1[0]], para, p1[1:]]), tau)

    IRF[IRF < 0] = 0
    np.seterr(divide='warn', over='warn', invalid='warn')
    
    return IRF


def PIRLSnonneg(x, y, max_num_iter=10):
    """
    Poisson Iterative Reweighted Least Squares (PIRLS) for non-negative beta.
    
    Solves X * beta = Y for Poisson-distributed data Y with non-negative beta.
    
    Parameters:
    x : array_like
        Predictors matrix (independent variables).
    y : array_like
        Response vector (dependent variable, Poisson-distributed).
    max_num_iter : int, optional
        Maximum number of iterations. Default is 10.
    
    Returns:
    beta : ndarray
        Solution vector.
    k : int
        Actual number of iterations used.
    """
    
    n = len(y)
    TINY = 0.1 / n  # Small regularization value
    w = np.zeros((n, n))
    
    # Initial guess using non-negative least squares
    beta_last = lsq_linear(x, y, bounds=(0, np.inf)).x
    
    for k in range(max_num_iter):
        # Update the weight matrix
        w[np.diag_indices_from(w)] = 1. / np.maximum(x @ beta_last, TINY)
        
        # Update beta using weighted least squares
        xt_w = x.T @ w
        beta = lsq_linear(xt_w @ x, xt_w @ y, bounds=(0, np.inf)).x
        
        # Check for convergence
        delta = beta - beta_last
        if np.sum(delta ** 2) < 1e-10:
            break
        
        beta_last = beta
    
    return beta, k

def MLFit(param, y, irf, p, plt_flag = None):
    """
    Computes the Maximum Likelihood (ML) error between the data y and the computed values.
    
    Parameters:
    param : array_like
        Parameters where:
        param[0] is the color shift between irf and y,
        param[1] is the irf offset,
        param[2:] are the decay times (tau).
    irf : array_like
        Measured Instrumental Response Function (IRF).
    y : array_like
        Measured fluorescence decay curve.
    p : int
        Time between laser excitations (in TCSPC channels).

    Returns:
    err : float
        Maximum Likelihood error.
    """
    
    n = len(irf)
    t = np.arange(1, n+1)
    tp = np.arange(1, p+1)
    c = param[0]
    tau = np.array(param[1:])
    
    # Matrix x calculation
    x = np.exp(-(tp[:, None] - 1) * (1.0 / tau)) @ np.diag(1.0 / (1 - np.exp(-p / tau)))
    irs = (1 - c + np.floor(c)) * irf[(t - np.int_(c) - 1) % n] + \
         (c - np.floor(c)) * irf[(t - int(np.ceil(c)) - 1) % n]
    
    # Perform the convolution using the Convol function
    z = Convol(irs, x)
    
    # Add a constant term
    # z = np.hstack([np.ones((z.shape[0], 1)), z])
    z = np.column_stack((np.ones(len(z)), z))
   
    # Perform non-negative least squares to fit A
    # A = lsq_linear(z, y, bounds=(0, np.inf)).x
    A,_ = PIRLSnonneg(z,y, 10) 
    
    # Recompute z using the estimated coefficients A
    z = z @ A
    
    # Calculate the error using Maximum Likelihood approach
    ind = y > 0
    err = np.sum(y[ind] * np.log(y[ind] / z[ind]) - y[ind] + z[ind]) / (n - len(tau))
    
    return err

def LSFit(param, y, irf, p, plt_flag = None):
    """
   LSFIT(param, irf, y, p) returns the Least-Squares deviation between the data y 
   and the computed values.
   
   LSFIT assumes a function of the form:
   
       y = yoffset + A(1)*convol(irf,exp(-t/tau(1)/(1-exp(-p/tau(1)))) + ...
   
   param(1) is the color shift value between irf and y.
   param(2) is the irf offset.
   param(3:...) are the decay times.
   
   irf is the measured Instrumental Response Function.
   y is the measured fluorescence decay curve.
   p is the time between two laser excitations (in number of TCSPC channels).
   
   Args:
   param : array
       The parameters for the fit (color shift, irf offset, decay times)
   irf : array
       The measured Instrumental Response Function.
   y : array
       The measured fluorescence decay curve.
   p : int
       The time between two laser excitations (in number of TCSPC channels).
   
   Returns:
   err : float
       The least-squares deviation between the measured data and computed values.
   """
    n = len(irf)
    t = np.arange(1, n+1)
    tp = np.arange(1, p+1)
    c = param[0]
    tau = np.array(param[1:])
    
    # Matrix x calculation
    x = np.exp(-(tp[:, None] - 1) * (1.0 / tau)) @ np.diag(1.0 / (1 - np.exp(-p / tau)))
    irs = (1 - c + np.floor(c)) * irf[(t - np.int_(c) - 1) % n] + \
         (c - np.floor(c)) * irf[(t - int(np.ceil(c)) - 1) % n]
         
    z = Convol(irs, x)
    # Add column of ones to z for fitting
    z = np.column_stack((np.ones(len(z)), z))
    # Linear least squares solution for A
    # A, _, _, _ = lstsq(z, y)
    # A,_ = nnls(z, y)
    A,_ = PIRLSnonneg(z,y,10)
    # print(A.shape)
    # Generate fitted curve
    z = z @ A
    
    if plt_flag is not None:
        # plt.semilogy(t, irs / np.max(irs) * np.max(y), label="irs")
        plt.semilogy(t, y, 'bo', label="y")
        plt.semilogy(t, z, label="fitted z")
        plt.legend()
        plt.draw()
        plt.pause(0.001)
        
        
    # Error calculation (Least-squares deviation)
    TINY = 10**-10
    err = np.sum((z+TINY - y) ** 2 / np.abs(z+TINY)) / (n - len(tau))
    
    return err

def FluoFit(irf, y, p, dt, tau, lim = None, init = None, flag_ml = True, plt_flag = 1):
    """
    The function FLUOFIT performs a fit of a multi-exponential decay curve.
    The function arguments are:
    irf 	= 	Instrumental Response Function
    y 	= 	Fluorescence decay data
    p 	= 	Time between laser exciation pulses (in nanoseconds)
    dt 	= 	Time width of one TCSPC channel (in nanoseconds)
    tau 	= 	Initial guess times
    lim   = 	limits for the lifetimes guess times
    init	=	Whether to use a initial guess routine or not  (not implemented yet!!!)

    The return parameters are:
    c 	=	Color Shift (time shift of the IRF with respect to the fluorescence curve)
    offset	=	Offset
    A	    =   Amplitudes of the different decay components
    tau	=	Decay times of the different decay components
    dc	=	Color shift error
    doffset	= 	Offset error
    dtau	=	Decay times error
    irs	=	IRF, shifted by the value of the colorshift
    zz	    Fitted fluorecence component curves
    t     =   time axis
    chi   =   chi2 value
    """

    irf = np.array(irf).flatten()
    offset = 0;
    y = np.array(y).flatten()
    n = len(irf); 
    c = 0 # this will change if colorshift correction is necessary
    m = len(tau)
    
    if lim is None:
        lim_min =  np.array([0.01] * m)
        lim_max =  np.array([100.0] * m)
    else:
        lim_min = lim[:m]
        lim_max = lim[m:]
    
    lim_min /= dt
    lim_max /= dt    
    p /= dt
    tp = np.arange(1,p+1) # time axis
    t = np.arange(1,n+1)
    tau /= dt
    
    param = np.concatenate(([c], tau)) 
    # ecay times and Offset are assumed to be positive.
    paramin = np.concatenate(([-1/dt], lim_min))
    paramax =np.concatenate(([1/dt], lim_max))
    
    if flag_ml is True:
        res = minimize_s(lambda x: MLFit(x, y.astype(np.float64), irf, np.floor(p + 0.5)), param, bounds=list(zip(paramin, paramax)))
    else:    
        res = minimize_s(lambda x: LSFit(x, y.astype(np.float64), irf, np.floor(p + 0.5)), param, bounds=list(zip(paramin, paramax)))
    
    
    xfit = res.x
    c = xfit[0]
    tau = xfit[1:]
    x = np.exp(-(tp[:, None] - 1) * (1.0 / tau)) @ np.diag(1.0 / (1 - np.exp(-p / tau)))
    irs = (1 - c + np.floor(c)) * irf[(t - np.int_(c) - 1) % n] + \
         (c - np.floor(c)) * irf[(t - int(np.ceil(c)) - 1) % n]
         
    z = Convol(irs, x);
    z = np.column_stack((np.ones(len(z)), z))
    # Linear least squares solution for A
    # A, _, _, _ = lstsq(z, y) 
    TINY = 10**-10
    # A,_ = nnls(z, y)
    A,_ = PIRLSnonneg(z,y,10)
    zz = z*A;
    z = z @ A      
    if plt_flag is not None:
        plt.semilogy(t, y, 'bo', label="y")
        plt.semilogy(t, z, label="fitted z")
        plt.legend()
        plt.draw()
        plt.pause(0.001)
    chi = np.sum((y-z-TINY)**2/ np.abs(z+TINY))/(n-m);
    t = dt*t;
    tau1 = dt*tau
    c = dt*c
    offset = zz[0]
    A = A[1:]    
    
    return tau1, A, c, z, zz, offset, irs, t, chi
    # 
    # 
    
        
    
    