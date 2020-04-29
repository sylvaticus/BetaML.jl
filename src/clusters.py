
# Python porting (partial)

import numpy as np
import math


# The EM algorithm (Lecture/segment 16.5 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

""" PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
def normalFixedSd(x,μ,σsq):
   return  (1/(2*np.pi*σsq)**(len(x)/2)) * np.exp(-1/(2*σsq)*np.linalg.norm(x-μ)**2)


# 16.5 The E-M Algorithm

def em(X,K,pO=None,μ0=None,σsq0=None,tol=0.0001,msgStep=10):
    """
      em(X,K,p0,μ0,σsq0,tol,msgStep)
    
    Compute Expectation-Maximisation algorithm to identify K clusters of X data assuming a Gaussian Mixture probabilistic Model.
    
    # Parameters:
    * `X`  :      A (n x d) data to clusterise
    * `K`  :      Number of cluster wanted
    * `p0` :      Initial probabilities of the categorical distribution (K x 1) [default: `nothing`]
    * `μ0` :      Initial means (K x d) of the Gaussian [default: `nothing`]
    * `σsq0`:      Initial variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions [default: `nothing`]
    * `tol`:      Initial tolerance to stop the algorithm [default: 0.0001]
    * `msgStep` : Iterations between update messages. Use 0 for no updates [default: 10]
    
    # Returns:
    * A touple of:
      * `pⱼₓ`: Matrix of size (N x K) of the probabilities of each point i to belong to cluster j
      * `pⱼ`  : Probabilities of the categorical distribution (K x 1)
      * `μ`  : Means (K x d) of the Gaussian
      * `σsq` : Variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions
      * `ϵ`  : Vector of the discrepancy (matrix norm) between pⱼₓ and the lagged pⱼₓ at each iteration
    
    # Example:
    ```python
    >>> X =  np.array([[1, 10.5], [1.5, 10.8], [1.8, 8], [1.7, 15], [3.2, 40], [3.6, 32], [3.3, 38], [5.1, -2.3], [5.2, -2.4]])    
    >>> clusters = em(X,3,msgStep=1)
    ```
    """

    # debug:
    #X = np.array([[1, 10.5], [1.5, 10.8], [1.8, 8], [1.7, 15], [3.2, 40], [3.6, 32], [3.3, 38], [5.1, -2.3], [5.2, -2.4]])
    #K = 3
    #p0=None; μ0=None; σsq0=None; tol=0.0001; msgStep=1
    #X     = make_matrix(X)
    (N,D) = np.shape(X)
    
    # Initialisation of the parameters if not provided
    minX = np.min(X,axis=0)
    maxX = np.max(X,axis=0)
    varX_byD = np.mean(np.var(X,axis=0))
    varX = np.mean(np.var(X,axis=0))/K**2
    
    pⱼ =  np.full(K, 1/K) if p0 == None else p0
    
    μ = np.zeros((K,D))
    
    if μ0 != None:
        #μ0  = make_matrix(μ0)
        μ = μ0
    else:
        μ = np.zeros((K,D))
        for d in range(D):
            μ[:,d] = np.linspace(minX[d],maxX[d],K)
    
    
    σsq =  np.full(K, varX) if σsq0 == None else σsq0
    
    pⱼₓ = np.zeros((N,K))
    ϵ = np.empty(0)
       
    while(True):     
        
        # E Step: assigning the posterior prob p(j|xi)
        pⱼₓlagged = np.copy(pⱼₓ)
        
        for n in range(N):
            px = np.sum([pⱼ[j]*normalFixedSd(X[n,:],μ[j,:],σsq[j]) for j in range(K)])
            for k in range(K):
                pⱼₓ[n,k] = pⱼ[k]*normalFixedSd(X[n,:],μ[k,:],σsq[k])/px
        
        
        
        # Compute the log-Likelihood of the parameters given the set of data
        # Just for informaticve purposes, not needed for the algorithm
        lL = 0
        for n in range(N):
            lL += np.log(np.sum([pⱼ[j]*normalFixedSd(X[n,:],μ[j,:],σsq[j]) for j in range(K)]))
        
        if msgStep != 0 and (len(ϵ) % msgStep == 0 or len(ϵ) == 1):
           print("Log likelihood on iter. "+str(len(ϵ)) +"\t: "+str(lL))
        
        # M step: find parameters that maximise the likelihood
        nⱼ  = np.sum(pⱼₓ,axis=0).transpose()
        n   = np.sum(nⱼ)
        pⱼ  = nⱼ / n
        μ   = np.matmul(pⱼₓ.transpose(), X) / nⱼ[:,None]
        σsq = [sum([pⱼₓ[n,j] * np.linalg.norm(X[n,:]-μ[j,:])**2 for n in range(N)]) for j in range(K) ] / (nⱼ * D)
        
        ϵ = np.append(ϵ,np.linalg.norm(pⱼₓlagged - pⱼₓ))
        
        if msgStep != 0 and (len(ϵ) % msgStep == 0 or len(ϵ) == 1):
           print("Iter. "+str(len(ϵ)) +"\t: "+str(ϵ[-1]))
           
        
        if (ϵ[-1] < tol):
            return (pⱼₓ,pⱼ,μ,σsq,ϵ)
        
        
#%timeit cluster = em(np.array([[1, 10.5], [1.5, 10.8], [1.8, 8], [1.7, 15], [3.2, 40], [3.6, 32], [3.3, 38], [5.1, -2.3], [5.2, -2.4]]),3,msgStep=0)
# 3.57 ms 

