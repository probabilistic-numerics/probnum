import numpy as np

def Genz_continuous(x,a=None,u=None):
         
    (n,dim) = x.shape
    
    # Specify default values of parameters a and u     
    if a is None:
        a = np.repeat(5.,dim)
    if u is None:
        u = np.repeat(0.5,dim)
     
    # Check that the parameters have valid values and shape
    assert len(u.shape) == 1 and u.shape[0] == dim
    assert len(a.shape) == 1 and a.shape[0] == dim
    assert np.all(u<=1.) and np.all(u >= 0.)
    
    # Check that the input points have valid values
    assert np.all(x<=1.) and np.all(x >= 0.)
    
    # Reshape u into an (n,dim) array with identical rows   
    u = np.repeat(u.reshape([1,dim]),n,axis=0)
           
    # Compute function values
    f = np.exp(-np.sum(a*np.abs(x-u),axis=1))
    
    return(f.reshape((n,1)))

