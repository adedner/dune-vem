#!/usr/bin/env python3

_print = print
def print(*args,**kwargs):
    _print(*args,**kwargs,flush=True)

import argparse
import pickle
import gc
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
def problemA1(diagA,stabFactor,linear):
    lam = 1
    x = ufl.SpatialCoordinate(dune.ufl.Space(2))
    if linear:
        a1, a2 = Constant(diagA), Constant(diagA)
    else:
        a1, a2 = Constant(diagA), Constant(4-diagA)
    hS,gS,mS = Constant(stabFactor),Constant(stabFactor*2*lam),Constant(stabFactor*lam**2)

    def _problem(N):
        def params(alpha):
            alpha = ufl.as_ufl(alpha)
            if linear:
                omega = ufl.conditional(alpha<1/2,30/360*2*ufl.pi,30/360*2*ufl.pi)
            else:
                omega = ufl.conditional(alpha<1/2,30/360*2*ufl.pi,140/360*2*ufl.pi)
            a = ufl.conditional(alpha<1/2,a1,a2)
            R = ufl.as_matrix([ [ ufl.cos(omega),  -ufl.sin(omega)],
                                [ ufl.sin(omega),   ufl.cos(omega)] ])
            A = ufl.as_matrix([ [1,0],[0,a] ])
            A = R.T*A*R
            b = ufl.as_vector([0,0])
            c = 1
            g = 0
            return A,b,c,g

        domain = dune.grid.cartesianDomain([0, 0], [np.pi, np.pi], [N, N])
        exact  = ufl.sin(2*x[0])*ufl.sin(x[1])

        dimP = None
        L, gamma = defL(params,lam)
        alpha  = argMaxDiscrete(L,[0,1])
        rhs    = ufl.Max(L(0,exact),L(1,exact))
        F      = lambda ubar,u: gamma(alpha(ubar)) * ( L(alpha(ubar),u) - rhs )
        return domain,dimP,lam,F, alpha,exact,[hS,gS,mS]
    _problem.params = (diagA,stabFactor,linear)
    return _problem


################################################################################

def main(order,orderTuple,N,problem,ax=None): # 0: std, 1:[o,o,o], 2:[o-2,o-2,o-2]
    domain,dimP,lam,F, alpha,exact,stab = problem(N)

    # setup grid
    gridView = dune.vem.polyGrid(
                 dune.vem.voronoiCells([domain.lower,domain.upper],
                                        domain.division[0]*domain.division[1],
                 lloyd=100, load="seeds") )

    # setup space
    testSpaces = [[0, 0], [order - 4, order - 3], [order - 4]]
    space = dune.vem.vemSpace(gridView, order=order, orderTuple=orderTuple, testSpaces=testSpaces)
    solution = space.function(name="solution")
    oldSol = solution.copy()

    exact = dune.fem.function.gridFunction(exact,gridView=gridView,name="exact")
    solution.interpolate(0)

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
        # parameters={"linear.verbose":True},
    )

    # Howard’s algorithm
    # (see page 41 in http://d-scholarship.pitt.edu/37334/13/Mohan%20Wu%20Final%20Thesis.pdf)
    diffLast = None
    diff0 = 1e5
    tol = 1e-6
    noRed = 0
    for i in range(100):
        oldSol.assign(solution)
        info = scheme.solve(target=solution)
        diff = oldSol.as_numpy - solution.as_numpy
        diff = np.dot(diff,diff) / len(diff)
        diff0 = min(diff,diff0)
        # print(f"{i}: {diff} {info} {tol**2/N}")
        if diffLast is None:
            diffLast = diff
        else:
            if diff > diffLast:
                print("No reduction step ",i,":", diff,diffLast,diff0)
                noRed += 1
                if noRed == 10:
                    print("10 no reduction step - bailing out!")
                    break
            #  assert diff < diffLast
            diffLast = diff

        if diff<tol**2/N:
            break
        if diff > diff0*50:
            print("Difference increased too much",diff,diff0)
            break
    print("    Iteration:",i,diff)
    # solution.plot()

    if ax:
        dune.fem.function.gridFunction(
              alpha(solution),gridView=gridView,order=1).plot(figure=ax,gridLines=None, level=1)
    e_h = solution - exact
    errors = dune.fem.integrate( [ufl.inner(e_h,e_h),
                                  ufl.inner(ufl.grad(e_h),ufl.grad(e_h)),
                                  ufl.inner(H(e_h),H(e_h))] )
    return np.sqrt(errors)

def rndSc(x,d):
    return [np.format_float_scientific(xx, precision=d,min_digits=d) for xx in x]
def rnd(x,d):
    return [np.round(xx,d) for xx in x]

def simulate(order,problem,proj,ax=None):
    orderTuple = (max(order,3),max(order-1,2),max(order-2,1))
    if proj == 0:
        orderTuple = (orderTuple[0],orderTuple[0],orderTuple[0])
    if proj == 1:
        orderTuple = (orderTuple[1],orderTuple[1],orderTuple[1])
    if proj == 2:
        orderTuple = (orderTuple[2],orderTuple[2],orderTuple[2])
    print("=============================")
    print(f"{problem}({problem.params}) order={order} {orderTuple}:")
    print("-----------------------------")

    errors = []
    eocs = [[-1,-1,-1]]
    maxLevel = 8-order # 7-order
    for N in [12*(2**i) for i in range(0,maxLevel)]:
        # if order==5 and proj==0 and N>24: break # issue with solver
        err2 = main(order=order, N=N+1, problem=problem, orderTuple=orderTuple,ax=ax)
        errors.append( err2 )
        if len(errors)>1:
            eocs.append( np.log(errors[-2]/errors[-1]) / np.log(2) )
        print(N,"\t", *rndSc(errors[-1],5),"\t", *rnd(eocs[-1],2), "\t# EOC")
    print()

    collected = gc.collect()
    print("Garbage collector: collected", "%d objects." % collected)

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

def testProj(proj,ret,o,filename):
    for r,pRow in enumerate(problemSet):
        for c,p in enumerate(pRow):
            print(f"Computing {proj,o,p.params}")
            key = (proj,o,p.params)
            if key not in ret:
                errors = simulate(o,p,proj,ax=None)
                ret[key] = errors
                with open(f"{filename}.dump", "wb") as f:
                    pickle.dump(ret,f)
            else:
                print("already computed")
            print("=====================")


#############################################################

parser = argparse.ArgumentParser(prog="kappa")
parser.add_argument("--linear", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--order", type=int, help="order", default=None)
parser.add_argument("--proj", type=int, help="projection", default=None)
args = parser.parse_args()

linear = args.linear
if args.order is not None:
    orders = [args.order]
else:
    orders = [3,4,5]
if args.proj is not None:
    projs = [args.proj]
else:
    projs = [1,2,0,-1]

args = parser.parse_args()

linear = args.linear

problemSet = [ [problemA1(a,s,linear)
                for s in [1] ]
                # for s in [100,10,1,0.1] ]
             for a in [0.0001] ]
             # for a in [2,1,0.01] ]

filename = f"kappa_{'linear' if linear else 'hjb'}"

print("Dumping to file",filename)

#############################################################

try:
    with open(f"{filename}.dump", "rb") as f:
        errors = pickle.load(f)
except:
    errors = {}

for o in orders:
    for proj in projs:
        testProj(proj,errors,o,filename)
