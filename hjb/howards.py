import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

import time
import ufl
import dune
import dune.vem

# Howard’s algorithm
# (see page 41 in http://d-scholarship.pitt.edu/37334/13/Mohan%20Wu%20Final%20Thesis.pdf)
def Howard(scheme,model,lam,
           solution,oldSol,
           tol=1e-12,startStab=100,
           params=None,
           exact=None,verbose=False):

    order = solution.space.order
    testSpaces = [[0, 0], [order - 4, order - 3], [order - 4]]
    orderTuple = [order,order,order]
    errSpace = dune.vem.vemSpace(solution.space.gridView,
                  order=order,
                  orderTuple=orderTuple, testSpaces=testSpaces)
    dbc = dune.ufl.DirichletBC(scheme.space.as_ufl(),0)
    schemeS = dune.vem.vemScheme(model,space=errSpace, boundary="value",
        hessStabilization=1,
        gradStabilization=0., # 2*lam,
        massStabilization=lam**2,
        solver=("suitesparse","umfpack") )

    errSol = errSpace.function(name="errSol")
    def H(w):
        return ufl.grad(ufl.grad(w))
    errors = []
    # initial guess with right boundary conditions
    scheme.setConstraints(oldSol)
    res_h = solution.copy()
    rhs = solution.copy()
    sd = solution.copy()

    targetStab = scheme.model.stabFactor
    stabFactor = startStab/targetStab
    A = scheme.linear()
    S = schemeS.linear()

    def solve(solution):
        if True:  # A0 = A0 + x ( S1 - A0 ) = (1-x) A0 + x S1
            scheme.model.stabFactor = 0
            scheme.jacobian(errSpace.zero,A,rhs=rhs)
            schemeS.jacobian(errSpace.zero,S)
            S.as_numpy[:] -= A.as_numpy
            A.as_numpy[:] += stabFactor*targetStab*S.as_numpy
            scheme.setConstraints(A)
            Ainv = splu(A.as_numpy.tocsc())
            solution.as_numpy[:] = Ainv.solve(rhs.as_numpy)
            scheme.model.stabFactor = stabFactor*targetStab
        else:
            scheme.solve(target=solution)
    def apply(res_h):
        if True:  # A0 = A0 + x ( S1 - A0 ) = (1-x) A0 + x S1
            scheme.model.stabFactor = 0
            scheme.jacobian(errSpace.zero,A,rhs=rhs)
            schemeS.jacobian(errSpace.zero,S)
            S.as_numpy[:] -= A.as_numpy
            A.as_numpy[:] += stabFactor*targetStab*S.as_numpy
            scheme.setConstraints(A)
            A(solution,res_h)
            res_h -= rhs
            scheme.model.stabFactor = stabFactor*targetStab
        else:
            scheme(solution,res_h)

    start = time.time()
    while True:
        scheme.model.stabFactor = stabFactor*targetStab
        for i in range(1,100):
            sd.assign(oldSol)
            solve(solution)
            oldSol.assign(solution)
            apply(res_h)
            res = np.dot(res_h.as_numpy,res_h.as_numpy)
            if res<tol**2:
                break
            sd.as_numpy[:] -= solution.as_numpy
            test = np.dot(sd.as_numpy,sd.as_numpy)
            print(i,scheme.model.stabFactor,res,tol**2,test)
            if test < 1e-8:
                break
        if verbose and exact:
            errSol.as_numpy[:] = solution.as_numpy
            e_h = errSol - exact
            err = dune.fem.integrate( [ufl.inner(e_h,e_h),
                                       ufl.inner(ufl.grad(e_h),ufl.grad(e_h)),
                                       ufl.inner(H(e_h),H(e_h))] )
            errors += [ [scheme.model.stabFactor,i,np.sqrt(err)] ]
            if verbose:
                print("    Iteration:",i,"\t"
                      "errors:",errors[-1],"\t",
                      "res:",res,"\t",
                      "with stab=",scheme.model.stabFactor,"\t",
                      "with param=",params,"\t",
                      "solving time used:", time.time() - start
                     )
        if stabFactor <= 1.01:
            break
        stabFactor /= 10

    return errors
