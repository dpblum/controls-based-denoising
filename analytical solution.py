def code_closed_form(R,coeff,b):
    # Analytical solution for optimal weights for denoising with patterns of 
    # non-pathological variance (NPV-patterns).
    # 
    # INPUT:
    # R:     n*p matrix with n observations and p dimensions (e.g. healthy controls images) 
    # coeff: patterns of non-pathological variance
    # b:     pathological read-out pattern
    # 
    # OUTPUT: 
    # Optimal weights of patterns of non-pathological variance.

    # centering R allows using norm-function
    R = R - np.mean(R,axis=0,keepdims=True)

    s     = coeff.shape[1]
    nullm = np.zeros((s,s))
    np.fill_diagonal(nullm,b@coeff)
    A     = R@coeff@nullm
    AAinv = la.inv(A.T@A)
    B     = R@b.T
    c     = (b@coeff@np.diag(b@coeff)).T

    w0 = AAinv@A.T@B
    e  = AAinv@c
    t  = la.norm(A@w0-B)**2/(b@b.T-c.T@w0)

    w = w0 - t*e

    return w
