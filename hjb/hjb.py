#!/usr/bin/env python3

_print = print
def print(*args,**kwargs):
    _print(*args,**kwargs,flush=True)

import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18,
})

import dune.fem
import dune.fem.plotting
import dune.fem.utility
import dune.grid
import dune.ufl
import dune.vem
import ufl
from dune.ufl import Constant

dune.fem.threading.use = dune.fem.threading.max

def H(w):
    return ufl.grad(ufl.grad(w))
def laplace(w):
    return ufl.div(ufl.grad(w))

# This is the HJB functional - params is a function of alpha returning
# the data A,b,c,g for the functional A:H(u)+b.D(u)-cu-g
# In addition this also returns 'gamma(alpha)' which requires the value of 'lambda'
def defL(params,lam):
    def L(alpha,u):
        A,b,c,g = params(alpha)
        return ufl.inner(A,H(u)) + ufl.inner(b,ufl.grad(u)) - c*u - g
    def gamma(alpha):
        A,b,c,g = params(alpha)
        return (   ( ufl.tr(A) + c/lam )
                 / ( ufl.inner(A,A) + ufl.dot(b,b)/(2*lam) + (c/lam)**2 )
               )
    return L,gamma

# argmax functions:
# Discrete version using UFL (will replace by C++ later and non discrete)
def argMaxDiscrete(L,als):
    assert len(als) == 2
    alpha = lambda u: ufl.conditional(L(als[0],u)>=L(als[1],u),als[0],als[1])
    return alpha

# Problem 1 (test 3 on page 41
#            in http://d-scholarship.pitt.edu/37334/13/Mohan%20Wu%20Final%20Thesis.pdf)
# this is a problem with two values for A which is modelled here with
# a double parameter alpha returning A^1 for alpha<1/2 and A^2 otherwise
# For this problem Cordes condition holds with lambda=1.

# We set g so that
# g = sup_{alpha=1,2} [A^alpha:D^2u + b^alpha.Du - c^alpha]
# with u = sin(x)sin(y)
#
# Note that A_1:H(u) = A_2:H(u) with the above 'u' so that alpha is not
# really fixed by this problem.a
# Input
# N: the grid resolution to use
# Return
#     domain: domain for grid construction
#     dimP:   size of alpha (None for discrete - possibly not needed)
#     lambda: lambda for Cordes condition
#     F:      takes two functions ubar and u and returns
#                 gamma_a (A_a:Hu+b_a.Du-cu-g_a)
#             with a = alpha(ubar) with alpha as below
#   Extra return values for debugging
#     alpha:  argmax function given 'u'
#     exact:  exact solution
def problem1(N):
    x = ufl.SpatialCoordinate(dune.ufl.Space(2))

    def params(alpha):
        alpha = ufl.as_ufl(alpha)
        mu = ufl.conditional(x[0]*x[1]>0,1,-1)
        A1 = ufl.as_matrix([ [2+mu,     1/2+mu/2],
                             [1/2+mu/2, 3/2+mu/2] ])
        A2 = ufl.as_matrix([ [3/2+mu/2, 1/2+mu/2],
                             [1/2+mu/2, 2+mu    ] ])
        A = ufl.conditional(alpha<1/2,A1,A2)
        b = ufl.as_vector([1,0])
        c = 1
        g = 0
        return A,b,c,g

    domain = dune.grid.cartesianDomain([-np.pi,-np.pi], [np.pi, np.pi], [N, N])
    exact  = ufl.sin(x[0])*ufl.sin(x[1])

    dimP = None
    lam = 1
    L, gamma = defL(params,lam)
    alpha  = argMaxDiscrete(L,[0,1])
    rhs    = ufl.Max(L(0,exact),L(1,exact))
    F      = lambda ubar,u: gamma(alpha(ubar)) * ( L(alpha(ubar),u) - rhs )
    return domain,dimP,lam,F, alpha,exact

# similar to above but with a more complex exact solution which leads to a
# more complex pattern for alpha
def problem2(N):
    x = ufl.SpatialCoordinate(dune.ufl.Space(2))

    def params(alpha):
        alpha = ufl.as_ufl(alpha)
        mu = ufl.conditional(x[0]*x[1]>0,1,-1)
        A1 = ufl.as_matrix([ [2+mu,     1/2+mu/2],
                             [1/2+mu/2, 3/2+mu/2] ])
        A2 = ufl.as_matrix([ [3/2+mu/2, 1/2+mu/2],
                             [1/2+mu/2, 2+mu    ] ])
        A = ufl.conditional(alpha<1/2,A1,A2)
        b = ufl.as_vector([1,0])
        c = 1
        g = 0
        return A,b,c,g

    domain = dune.grid.cartesianDomain([-np.pi/2,-np.pi/2], [np.pi, np.pi], [N, N])
    exact  = ufl.sin(x[0]*(ufl.pi-x[0]))*ufl.sin(x[1]*x[0])

    dimP = None
    lam = 1
    L, gamma = defL(params,lam)
    alpha  = argMaxDiscrete(L,[0,1])
    rhs    = ufl.Max(L(0,exact),L(1,exact))
    F      = lambda ubar,u: gamma(alpha(ubar)) * ( L(alpha(ubar),u) - rhs )
    return domain,dimP,lam,F, alpha,exact

# similar to above but with a more complex exact solution which leads to a
# more complex pattern for alpha but only with x>0,y>0 so that A_1,A_2 are constant
# (discontinuous previously)
def problem3(N):
    domain = dune.grid.cartesianDomain([np.pi/2,np.pi/2], [np.pi, np.pi], [N, N])
    p = problem2(N)
    return domain,*p[1:]

def problemA1(diagA):
    lam = 1
    x = ufl.SpatialCoordinate(dune.ufl.Space(2))
    # a1, a2 = Constant(0.001), Constant(3.99)
    # a1, a2 = Constant(0.1), Constant(3)
    a1, a2 = Constant(diagA), Constant(diagA)
    fac = 1
    hS,gS,mS = Constant(fac),Constant(fac*2*lam),Constant(fac*lam**2)

    def _problem(N):
        def params(alpha):
            alpha = ufl.as_ufl(alpha)
            # omega = ufl.conditional(alpha<1/2,30/360*2*ufl.pi,140/360*2*ufl.pi)
            omega = ufl.conditional(alpha<1/2,30/360*2*ufl.pi,30/360*2*ufl.pi)
            # omega = ufl.conditional(alpha<1/2,140/360*2*ufl.pi,140/360*2*ufl.pi)
            a = ufl.conditional(alpha<1/2,a1,a2)
            R = ufl.as_matrix([ [ ufl.cos(omega),  -ufl.sin(omega)],
                                [ ufl.sin(omega),   ufl.cos(omega)] ])
            A = ufl.as_matrix([ [1,0],[0,a] ])
            A = R.T*A*R
            b = ufl.as_vector([0,0])
            c = 1
            g = 0
            return A,b,c,g

        # domain = dune.grid.cartesianDomain([-np.pi,-np.pi], [np.pi, np.pi], [N, N])
        domain = dune.grid.cartesianDomain([0, 0], [np.pi, np.pi], [N, N])
        exact  = ufl.sin(x[0])*ufl.sin(x[1])

        dimP = None
        L, gamma = defL(params,lam)
        alpha  = argMaxDiscrete(L,[0,1])
        rhs    = ufl.Max(L(0,exact),L(1,exact))
        F      = lambda ubar,u: gamma(alpha(ubar)) * ( L(alpha(ubar),u) - rhs )
        return domain,dimP,lam,F, alpha,exact,[hS,gS,mS]
    return _problem


################################################################################

def main(order,orderTuple,N,problem): # 0: std, 1:[o,o,o], 2:[o-2,o-2,o-2]
    domain,dimP,lam,F, alpha,exact,stab = problem(N)

    # setup grid
    gridView = dune.vem.polyGrid(domain, cubes=False)

    # setup space
    # orders = 3*(order,)
    testSpaces = [[0, 0], [order - 4, order - 3], [order - 4]]
    space = dune.vem.vemSpace(gridView, order=order, orderTuple=orderTuple, testSpaces=testSpaces)
    solution = space.function(name="solution")
    oldSol = solution.copy()

    exact = dune.fem.function.gridFunction(exact,gridView=gridView,name="exact")

    ##########################

    # Setup model for Howards algorithm:
    #
    # Setup integral of F(ubar,u) (laplace(v) - lam v) = 0
    # where ubar is used to fix alpha and the problem is solved for 'u',  i.e.,
    # F(w,u) = gamma_a ( A_a:D^2u + b_a.Du - c_au - g_a )
    # with a(x) = argmax_a( A_a(x):D^2w(x) + b_a(x).Dw(x) - c_a(x)w(x) - g_a(x) )
    # The algorithm is then a fixedpoint algorithm to obtain solution to
    # integral of F(u,u) (laplace(v) - lam v) = 0
    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)
    testFct = laplace(v) - lam*v
    a = F(oldSol,u) * testFct * ufl.dx

    dbc = [ dune.ufl.DirichletBC(space,exact) ]
    extraArgs = {"boundary": "value"}

    scheme = dune.vem.vemScheme([a == 0, *dbc], **extraArgs,
        hessStabilization=stab[0],
        gradStabilization=stab[1],
        massStabilization=stab[2],
        solver=("suitesparse","umfpack"),
    )

    # some test output
    """
    dune.fem.function.gridFunction(alpha(exact),gridView=gridView,order=1).plot()
    exact.plot()
    # the next three should be (close to) zero
    exact_h = space.interpolate(exact,name="exact_h")
    dune.fem.function.gridFunction(F(exact,exact),gridView=gridView,order=1).plot()
    dune.fem.function.gridFunction(F(exact_h,exact_h),gridView=gridView,order=1).plot()
    oldSol.assign(exact_h)
    scheme(exact_h,oldSol)
    oldSol.plot()
    """

    # Howard’s algorithm
    # (see page 41 in http://d-scholarship.pitt.edu/37334/13/Mohan%20Wu%20Final%20Thesis.pdf)
    diffLast = None
    for i in range(100):
        oldSol.assign(solution)
        info = scheme.solve(target=solution)
        diff = oldSol.as_numpy - solution.as_numpy
        diff = np.dot(diff,diff) / len(diff)
        if diffLast is None:
            diffLast = diff
        else:
            assert diff < diffLast
            diffLast = diff
        # print(f"{i}: {diff} {info} {1e-3**2/N}")
        if diff<1e-3**2/N:
            break
    print("    Iteration:",i,diff)
    # solution.plot()
    # dune.fem.function.gridFunction(alpha(solution),gridView=gridView,order=1).plot()
    e_h = solution - exact
    errors = dune.fem.integrate( [ufl.inner(e_h,e_h),
                                  ufl.inner(ufl.grad(e_h),ufl.grad(e_h)),
                                  ufl.inner(H(e_h),H(e_h))] )
    return np.sqrt(errors)

def rndSc(x,d):
    return [np.format_float_scientific(xx, precision=d,min_digits=d) for xx in x]
def rnd(x,d):
    return [np.round(xx,d) for xx in x]
def simulate(order,problem,proj):

    orderTuple = (max(order,3),max(order-1,2),max(order-2,1))
    if proj == 0:
        orderTuple = (orderTuple[0],orderTuple[0],orderTuple[0])
    if proj == 1:
        orderTuple = (orderTuple[1],orderTuple[1],orderTuple[1])
    if proj == 2:
        orderTuple = (orderTuple[2],orderTuple[2],orderTuple[2])
    print("=============================")
    print(f"{problem} ({order}) {orderTuple}:")
    print("-----------------------------")

    errors = []
    eocs = [[-1,-1,-1]]
    maxLevel = 8-order # 7-order
    for N in [12*(2**i) for i in range(maxLevel)]:
        err2 = main(order=order, N=N+1, problem=problem, orderTuple=orderTuple)
        errors.append( err2 )
        if len(errors)>1:
            eocs.append( np.log(errors[-2]/errors[-1]) / np.log(2) )
        print(N,"\t", *rndSc(errors[-1],5),"\t", *rnd(eocs[-1],2))
    print()
    return np.array( errors )

def plot(fig,ax,errors,order,xname,yname):
    h = np.array([ 0.5**i for i in range(len(errors[:,0])) ])
    ax.loglog(h,errors[:,0],color="c",label="$L^2$",linestyle='--',marker='o')
    ax.loglog(h,errors[:,1],color="r",label="$H^1$",linestyle='--',marker='o')
    ax.loglog(h,errors[:,2],color="b",label="$H^2$",linestyle='--',marker='o')
    ax.loglog(h,0.9*errors[-2,0]*(h/h[-2])**(order+1),color="c",label=f"{order+1}",linestyle='-')
    ax.loglog(h,0.9*errors[-2,1]*(h/h[-2])**(order+0),color="r",label=f"{order+0}",linestyle='-')
    ax.loglog(h,0.9*errors[-2,2]*(h/h[-2])**(order-1),color="b",label=f"{order-1}",linestyle='-')
    ax.legend()
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

#############################################################

problemSet = [problemA1()]
orders = [3,4]

def testProj(proj):
    ret = {}
    fig,axs = plt.subplots(3,3,figsize=(15,15))
    for i,o in enumerate(orders):
        for j,p in enumerate(problemSet):
            errors = simulate(o,p,proj)
            xlabel = f"Problem {j+1}" if i==2 else None
            ylabel = f"order={o}" if j==0 else None
            plot(fig,axs[i][j],errors,o,xlabel,ylabel)
            ret[(i,j)] = errors
    # plt.show()
    plt.savefig(f"equal{proj}.pdf")
    return ret

#############################################################

try:
    with open("hjb.dump", "rb") as f:
        [origError,equalError0, equalError1, equalError2] = pickle.load(f)
except:
    origError,equalError0, equalError1, equalError2 = None,None,None,None
origError,equalError0, equalError1, equalError2 = None,None,None,None

if origError is None:
    origError  = testProj(-1)
if equalError0 is None:
    equalError0 = testProj(0)
"""
if equalError1 is None:
    equalError1 = testProj(1)
if equalError2 is None:
    equalError2 = testProj(2)
"""

with open("hjb.dump", "wb") as f:
    pickle.dump([origError,equalError0,equalError1,equalError2],f)

fig,axs = plt.subplots(9,3,figsize=(15,45))
for i,o in enumerate(orders): # y-axis
    for j,p in enumerate(problemSet): # x-axis
        xlabel = f"Problem {j+1}" if i==2 else None
        origE = origError[(i,j)]
        h = np.array([ 0.5**i for i in range(len(origE[:,0])) ])
        for k,eName in enumerate(["$L^2$","$H^1$","$H^2$"]):
            ylabel = f"order={o}, error={eName}" if j==0 else None         # y-axis
            ax = axs[i*3+k][j]
            ax.loglog(h,origE[:,k], color="c",label="orignal",
                                    linestyle='--',marker='o')
            try:
                equalE0 = equalError0[(i,j)]
                ax.loglog(h,equalE0[:,k],color="r",label="equal (p)",
                                        linestyle='--',marker='o')
            except:
                pass
            try:
                equalE1 = equalError1[(i,j)]
                ax.loglog(h,equalE1[:,k],color="b",label="equal (p-1)",
                                        linestyle='--',marker='o')
            except:
                pass
            try:
                equalE2 = equalError2[(i,j)]
                ax.loglog(h,equalE2[:,k],color="g",label="equal (p-2)",
                                        linestyle='--',marker='o')
            except:
                pass
            ax.legend()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

plt.savefig(f"hjb{diagA}.pdf")
