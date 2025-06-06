import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

import time
import ufl
import dune

# Howard’s algorithm
# (see page 41 in http://d-scholarship.pitt.edu/37334/13/Mohan%20Wu%20Final%20Thesis.pdf)
def Howard(scheme,lam,
           solution,oldSol,
           tol=1e-12,startStab=100,
           exact=None,verbose=False):
    order = solution.space.order
    testSpaces = [[0, 0], [order - 4, order - 3], [order - 4]]
    orderTuple = [order,order,order]
    errSpace = dune.vem.vemSpace(solution.space.gridView,
                  order=order,
                  orderTuple=orderTuple, testSpaces=testSpaces)
    errSpace = solution.space
    errSol = errSpace.function(name="errSol")
    def H(w):
        return ufl.grad(ufl.grad(w))
    errors = []
    # initial guess with right boundary conditions
    # scheme.setConstraints(oldSol)
    res_h = solution.copy()
    rhs = solution.copy()
    rhs1 = solution.copy()
    sd = solution.copy()

    targetStab = scheme.model.stabFactor
    stabFactor = startStab/targetStab

    scheme.model.stabFactor = 0
    A = scheme.linear()
    A1 = scheme.linear()

    S = dune.vem.stabilization(errSpace,1,2,1) # needs scalling with lambda
    scheme.model.stabFactor = 0
    scheme.jacobian(errSpace.zero,A,rhs=rhs)
    scheme.jacobian(errSpace.zero,A1,rhs=rhs1)
    # A.as_numpy[:] += 100*S.as_numpy[:]
    # scheme.setConstraints(A)
    Ainv = splu(A.as_numpy.tocsc())
    solution.as_numpy[:] = Ainv.solve(rhs.as_numpy)

    # scheme.model.stabFactor = 0
    # scheme.jacobian(errSpace.zero,A1,rhs=rhs1)
    A1inv = splu(A1.as_numpy.tocsc())
    solution.as_numpy[:] -= A1inv.solve(rhs.as_numpy)
    solution.plot()

    start = time.time()
    while True:
        for i in range(1,100):
            sd.assign(oldSol)

            scheme.model.stabFactor = 0
            scheme.jacobian(errSpace.zero,A,rhs=rhs)
            A.as_numpy[:] += stabFactor*targetStab*S.as_numpy
            scheme.setConstraints(A)
            Ainv = splu(A.as_numpy.tocsc())
            solution.as_numpy[:] = Ainv.solve(rhs.as_numpy)

            scheme.model.stabFactor = stabFactor*targetStab
            scheme.jacobian(errSpace.zero,A1,rhs=rhs1)

            scheme.solve(target=solution)

            oldSol.assign(solution)
            scheme(solution,res_h)
            res = np.dot(res_h.as_numpy,res_h.as_numpy)
            if res<tol**2:
                break
            sd.as_numpy[:] -= solution.as_numpy
            test = np.dot(sd.as_numpy,sd.as_numpy)
            print(i,res,tol**2,test)
            if test < 1e-10:
                break
        if verbose and exact:
            errSol.as_numpy[:] = solution.as_numpy
            e_h = errSol - exact
            err = dune.fem.integrate( [ufl.inner(e_h,e_h),
                                       ufl.inner(ufl.grad(e_h),ufl.grad(e_h)),
                                       ufl.inner(H(e_h),H(e_h))] )
            errors += [ [stabFactor,np.sqrt(err)] ]
            if verbose:
                print("    Iteration:",i,res,"\t"
                      "solving time used:", time.time() - start,
                      "with stab=",stabFactor,"\t",
                      "errors:",errors[-1])
        if stabFactor <= 1.01:
            break
        stabFactor /= 10

    return errors
