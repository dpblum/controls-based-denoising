def code_closed_form(R,coeff,b):
    # Analytical solution for optimal weights for denoising with patterns of 
    # non-pathological variance (NPV-patterns).
    # 
    # INPUT:
    # R:    n*p matix with n observations and p dimensions (e.g. healthy controls images) 
    # res:  struct output from code_npv_patterns.m that contains NPV-patterns (res.coeff).
    # b:    pathological read-out pattern before CODE.
    # 
    # OUTPUT: 
    # Optimal weights of patterns of non-pathological variance.

    # centering R allows using norm
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
