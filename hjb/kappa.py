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

from howards import Howard as solver

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
    stabFactor = Constant(stabFactor,name="stabFactor")  # !!!!
    hS,gS,mS = stabFactor,stabFactor*2*lam,stabFactor*lam**2

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
    _problem.params = (diagA,stabFactor.value,linear)
    return _problem


################################################################################

def main(order,orderTuple,N,problem,ax=None): # 0: std, 1:[o,o,o], 2:[o-2,o-2,o-2]
    domain,dimP,lam,F, alpha,exact,stab = problem(N)

    # setup grid
    if grid == "voronoi":
        gridView = dune.vem.polyGrid(
                     dune.vem.voronoiCells([domain.lower,domain.upper],
                                            domain.division[0]*domain.division[1],
                     lloyd=100, load="seeds") )
    elif grid == "cube":
        gridView = dune.vem.polyGrid(domain, cubes=True)
    elif grid == "simplex":
        gridView = dune.vem.polyGrid(domain, cubes=False)

    # setup space
    testSpaces = [[0, 0], [order - 4, order - 3], [order - 4]]
    space = dune.vem.vemSpace(gridView, order=order,
                              orderTuple=orderTuple,
                              testSpaces=testSpaces)
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

    if True:
        stab[0] = None
        stab[1] = None
    scheme = dune.vem.vemScheme([a == 0, *dbc], boundary="value",
        hessStabilization=stab[0],
        gradStabilization=None,        # stab[1],
        massStabilization=stab[2],
        solver=("suitesparse","umfpack"),
        # parameters={"linear.verbose":True},
    )

    errors = solver(scheme,[a == 0, *dbc],
                    lam,
                    solution,oldSol,
                    tol=1e-8,
                    startStab=startStab,
                    params=problem.params,
                    exact=exact,verbose=True)
    return errors

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
            eocs.append( np.log(errors[-2][-1][2]/errors[-1][-1][2]) / np.log(2) )
        print(N,"\t", *rndSc(errors[-1][-1][2],5),"\t", *rnd(eocs[-1],2), "\t# EOC")
    print()

    collected = gc.collect()
    print("Garbage collector: collected", "%d objects." % collected)

    return errors

def testProj(proj,ret,o,filename):
    for r,p in enumerate(problemSet):
        print(f"Computing {proj,o,p.params}")
        key = (proj,o,p.params)
        if force or key not in ret:
            ret[key] = simulate(o,p,proj,ax=None)
            with open(f"{filename}.dump", "wb") as f:
                pickle.dump(ret,f)
        else:
            print("already computed")
        print("=====================")


#############################################################

grids = ['cube','simplex','voronoi']

parser = argparse.ArgumentParser(prog="kappa")
parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--linear", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--order", type=int, help="order", default=None)
parser.add_argument("--proj", type=int, help="projection", default=None)
parser.add_argument("--stab", type=float, help="stabilization factor")
parser.add_argument("--startStab", type=float, help="iterative stabilization factor")
parser.add_argument("--grid", choices=grids)
args = parser.parse_args()

force = args.force
linear = args.linear
grid = args.grid

if args.order is not None:
    orders = [args.order]
else:
    orders = [3,4,5]
if args.proj is not None:
    projs = [args.proj]
else:
    projs = [1,2,0,-1]

stab = args.stab
startStab = stab if args.startStab<0 else args.startStab

problemSet = [ problemA1(a,stab,linear) for a in [2,1,0.01,0.0001] ]


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
