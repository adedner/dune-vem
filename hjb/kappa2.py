#!/usr/bin/env python3

_print = print
def print(*args,**kwargs):
    _print(*args,**kwargs,flush=True)

import argparse, pickle, time
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
    stabFactor = Constant(stabFactor,name="stabFactor")
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

        # domain = dune.grid.cartesianDomain([-1, -1], [1, 1], [N, N])
        # exact  = x[0]**4 # works
        # exact  = x[1]**4 # works
        # exact  = x[0]*x[1]**3 # works
        # exact  = x[0]**2*x[1]**2 # works
        # exact  = x[0]**2*x[1]**3 # fails
        # exact  = x[0]*x[1]**4 # fails
        # exact  = x[0]**4*x[1] # fails
        # exact  = x[0]**3*x[1]**3
        # exact  = ufl.sin(ufl.pi*2*x[0])*ufl.cos(ufl.pi*x[1])
        # exact = (x[0]-2)*(x[1]-1)**3

        domain = dune.grid.cartesianDomain([0, 0], [np.pi, np.pi], [N, N])
        exact  = ufl.sin(2*x[0])*ufl.sin(x[1])
        # exact = ufl.cos(ufl.pi*0.1*x[0])**2 * ufl.cos(0.1*ufl.pi*x[1])**2

        dimP = None
        L, gamma = defL(params,lam)
        alpha  = argMaxDiscrete(L,[0,1])
        rhs    = ufl.Max(L(0,exact),L(1,exact))
        F      = lambda ubar,u: gamma(alpha(ubar)) * ( L(alpha(ubar),u) - rhs )
        return domain,dimP,lam,F, alpha,exact,[hS,gS,mS]
    _problem.params = (diagA,stabFactor,linear)
    return _problem


################################################################################

def main(order,orderTuple,N,problem,ax): # 0: std, 1:[o,o,o], 2:[o-2,o-2,o-2]
    domain,dimP,lam,F, alpha,exact,stab = problem(N)

    # setup grid
    # gridView = dune.vem.polyGrid(domain, cubes=True)
    print("setting up grid",domain.division[0]*domain.division[1],[domain.lower,domain.upper])
    gridView = dune.vem.polyGrid(
                 dune.vem.voronoiCells([domain.lower,domain.upper],
                                        domain.division[0]*domain.division[1],
                 lloyd=100, load="seeds") )

    # setup space
    # orders = 3*(order,)
    testSpaces = [[0, 0], [order - 4, order - 3], [order - 4]]
    start = time.time()
    print("setting up space")
    space = dune.vem.vemSpace(gridView, order=order,
                     # computeField="Dune::Float128", # 128,
                     orderTuple=orderTuple,
                     testSpaces=testSpaces)
    print("space setup time used:", time.time() - start)

    solution = space.function(name="solution")
    solution.clear()

    exact = dune.fem.function.gridFunction(exact,gridView=gridView,name="exact")
    # solution.interpolate(exact)
    oldSol = solution.copy()

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
    errors = solver(scheme,solution,oldSol,tol=1e-8,startStab=startStab,exact=exact,verbose=True)

    """
    oldSol.interpolate(exact)
    solution.clear()
    scheme.setConstraints(oldSol,solution)
    # solution.plot()
    # Howard’s algorithm
    # (see page 41 in http://d-scholarship.pitt.edu/37334/13/Mohan%20Wu%20Final%20Thesis.pdf)
    tol = 1e-6
    res_h = solution.copy()
    sd = solution.copy()

    print("starting algorithm")
    start = time.time()
    oldSol.assign(solution)
    targetS = scheme.model.stabFactor
    scheme.model.stabFactor = startStab
    while True:
        # print("solving with stability constant:",scheme.model.stabFactor)
        res0 = None
        for i in range(1,100):
            sd.assign(oldSol)
            info = scheme.solve(target=solution)
            oldSol.assign(solution)
            scheme(solution,res_h)
            res = np.dot(res_h.as_numpy,res_h.as_numpy)
            # print(i,res,res0,tol**2/N)
            if res<tol**2/N:
                break
            if res0 is None:
                res0 = res
                continue
            # sd = oldSol - solution
            # solution + sd/2 = solution + oldSol/2 - solution/2 = 1/2 ( oldSol + solution )
            sd.as_numpy[:] -= solution.as_numpy
            test = np.dot(sd.as_numpy,sd.as_numpy)
            while res > res0:
                sd.as_numpy[:] /= 2
                test = np.dot(sd.as_numpy,sd.as_numpy)
                # print("line search:",test,res,res0)
                if test < 1e-10: break
                solution.as_numpy[:] += sd.as_numpy
                # oldSol.assign(solution)
                scheme(solution,res_h)
                res = np.dot(res_h.as_numpy,res_h.as_numpy)
            if test < 1e-10: break
            res0 = res
        e_h = solution - exact
        errors = dune.fem.integrate( [ufl.inner(e_h,e_h),
                                      ufl.inner(ufl.grad(e_h),ufl.grad(e_h)),
                                      ufl.inner(H(e_h),H(e_h))] )
        errors = np.sqrt(errors)
        print("    Iteration:",i,res,"\t"
              "solving time used:", time.time() - start,
              "with stab=",scheme.model.stabFactor,"\t",
              "errors:",errors)
        if scheme.model.stabFactor <= 1.01*targetS:
            break
        scheme.model.stabFactor /= 10
    """

    """
    solution.plot(block=False,level=2)
    fig,axs = plt.subplots(2,2, figsize=(10,10))
    Herr = lambda i,j: ufl.ln(abs(H(solution-exact)[i,j]))
    dune.fem.function.gridFunction(Herr(0,0)).plot(level=1,
              figure=(fig,axs[0][0]))
    dune.fem.function.gridFunction(Herr(0,1)).plot(level=1,
              figure=(fig,axs[0][1]))
    dune.fem.function.gridFunction(Herr(1,0)).plot(level=1,
              figure=(fig,axs[1][0]))
    dune.fem.function.gridFunction(Herr(1,1)).plot(level=1,
              figure=(fig,axs[1][1]))
    plt.show()
    solution.plot(block=False,level=2,gridLines=None)
    dune.fem.function.gridFunction(alpha(solution),gridView=gridView,order=1).plot(
                   gridLines=None, level=1)
    plt.show()
    """
    return errors

def rndSc(x,d):
    return [np.format_float_scientific(xx, precision=d,min_digits=d) for xx in x]
def rnd(x,d):
    return [np.round(xx,d) for xx in x]

def simulate(order,problem,proj,ax):
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
    maxLevel = 8-order # 9-order
    for N in [12*(2**i) for i in range(0,maxLevel)]:
        # if order==5 and proj==0 and N>24: break # issue with solver
        err2 = main(order=order, N=N+1, problem=problem, orderTuple=orderTuple,ax=ax)[-1]
        errors.append( err2 )
        if len(errors)>1:
            eocs.append( np.log(errors[-2]/errors[-1]) / np.log(2) )
        print(N,"\t", *rndSc(errors[-1],5),"\t", *rnd(eocs[-1],2), "\t# EOC")
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

def testProj(proj,order):
    ret = {}
    for j,p in enumerate(problemSet):
        errors = simulate(order,p,proj,None) # ax=(fig,axs[i,j]))
        key = (proj,order,p.params)
        ret[key] = errors
        """
        xlabel = f"Problem {p.params}" if i==2 else None
        ylabel = f"order={o}" if j==0 else None
        axs[i,j].legend()
        axs[i,j].set_xlabel(xlabel)
        axs[i,j].set_ylabel(ylabel)
        plot(fig,axs[i][j],errors,o,xlabel,ylabel)
        """
    # plt.show()
    # plt.savefig(f"{filename}_{proj}_alpha.pdf")
    return ret


#############################################################

parser = argparse.ArgumentParser(prog="kappa")
parser.add_argument("--linear", action=argparse.BooleanOptionalAction)
parser.add_argument("--aParam", type=float, help="a parameter")
parser.add_argument("--stab", type=float, help="stabilization factor")
parser.add_argument("--order", type=int, help="order")
parser.add_argument("--proj", type=int, help="projection")
parser.add_argument("--startStab", type=float, default=-1, help="iterative stabilization factor")

args = parser.parse_args()

linear = args.linear
stab = args.stab
aParam = args.aParam
order = args.order
proj = args.proj
startStab = stab if args.startStab<0 else args.startStab

assert stab>0 and aParam>0
assert order >= 3
assert proj in [-1,0,1,2]

problemSet = [problemA1(aParam,stab,linear)]

filename = f"kappa_{'linear' if linear else 'hjb'}_{stab}"

print("Writing to file",filename)
print("Using stabilization factor=",stab)
origError = testProj(proj,order)
