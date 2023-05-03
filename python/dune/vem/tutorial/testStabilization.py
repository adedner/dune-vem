import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math, argparse
from scipy.sparse.linalg import spsolve
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem.operator import linear as linearOperator
import dune.vem

from ufl import *
import dune.ufl

parser = argparse.ArgumentParser(description='set order and stabilization')
parser.add_argument('-L', type=int, required=True, help="max-level > 0")
parser.add_argument('-l', type=int, required=True, help="polynomial order >= 1")
parser.add_argument('-s', type=int, required=True, help="use stabilization = [-1|0|1]")
parser.add_argument('-p', type=str, required=True, help="problem = [laplace|biharmonic]")

# Space and stabilization is influenced by the '-s' parameter:
#    1: normal space with stabilization
#    0: normal space w.o. stabilization
#   -1: extended space w.o. stabilization
args = parser.parse_args()
useStab       = args.s     # one of 1,0,-1
maxLevel      = args.L     # number of levels to use for EOC computation
problem       = args.p     # problem to solve (laplace or biharmonic)
order         = args.l     # order (>=1 for laplace >=2 for biharmonic)

if useStab == -1: # set range space of value, gradient, and hessian projections to P_l,P_l, and P_{l-1}
    orders = [order,order,order-1]
else:             # default setup
    orders = order # = [order,order-1,order-2]

if problem == "biharmonic": # parameters for H^2-conforming space
    spaceArgs = {"order":orders, "testSpaces":[ [0,0], [order-4,order-3], [order-4] ] }
else: # parameters for H^1-conforming space
    spaceArgs = {"order":orders, "testSpaces":[ [0,-1], [order-2,-1], [order-2] ] }

# parameters for solver
parameters = {"newton.linear.preconditioning.method": "jacobi",
              "newton.linear.tolerance":1e-12,
              "newton.linear.verbose": False, "newton.verbose": False }

# Now we define the model starting with the exact solution and boundary conditions
uflSpace = dune.ufl.Space(2)
x = SpatialCoordinate(uflSpace)
exact = sin(2*pi*x[0])**2*sin(2*pi*x[1])**2
dbc = [dune.ufl.DirichletBC(uflSpace, 0, i+1) for i in range(4)]

# and now the bilinear and linear forms
u = TrialFunction(uflSpace)
v = TestFunction(uflSpace)
H = lambda w: grad(grad(w))

kappa = 10/(1e-2+dot(x,x))

if problem == "biharmonic":
    a = kappa*inner(H(u),H(v)) * dx
    b = sum([ H(kappa*H(exact)[i,j])[i,j] for i in range(2) for j in range(2) ]) * v * dx
else:
    a = kappa*inner(grad(u),grad(v)) * dx
    b = - div(kappa*grad(exact)) * v * dx

# Finally the stabilization depending on provided runtime arguments
if useStab == 1:
    print("# original space with standard stabilization!")
    stabCoeff = kappa
elif useStab == 0:
    print("# no stabilization with original space!")
    stabCoeff = None
else: # useStab=-1
    print("# using extended projections with no stabilization!")
    stabCoeff = None

# We now define a function to compute the solution and the $L^2,H^1$ error
# given a grid and a space
def compute(grid, space):
    df = discreteFunction(space, name="solution") # space.interpolate([0],name="solution")
    scheme = dune.vem.vemScheme([a==b, *dbc], space,
                        solver="cg",
                        # solver=("suitesparse","umfpack"),
                        hessStabilization=stabCoeff if problem=="biharmonic" else None,
                        gradStabilization=stabCoeff if problem=="laplace" else None,
                        massStabilization=None,
                        parameters=parameters)
    info = scheme.solve(target=df)
    edf = exact-df
    energy  = replace(a,{u:edf,v:edf}).integrals()[0].integrand()
    err = [energy,
           inner(edf,edf),
           inner(grad(edf),grad(edf))]
    if problem == "biharmonic":
        err += [inner(grad(grad(edf)),grad(grad(edf)))]
    errors  = [ math.sqrt(e) for e in integrate(grid, err, order=8) ]
    return df, errors, info

# Finally we iterate over the requested methods and solve the problems
results = []
for level in range(maxLevel+1):
    constructor = cartesianDomain([0,0],[1,1],[4*2**level,4*2**level])
    polyGrid = dune.vem.polyGrid(constructor, cubes=True )
    # constructor = dune.vem.voronoiCells([[0,0],[1,1]],16*2**level*2**level, load="voronoiseeds",show=False,lloyd=50)
    # polyGrid = dune.vem.polyGrid(constructor)
    res = []
    space = dune.vem.vemSpace(polyGrid, **spaceArgs)
    dfs,errors,info = compute(polyGrid, space)
    print("[",level,"]", "Size: ",polyGrid.size(0), space.size,
          "Energy: ", errors[0], "semi-norms: ", errors[1:],
          info["linear_iterations"], flush=True)
    results += [ [polyGrid.hierarchicalGrid.agglomerate.size,space.size,info["linear_iterations"],*errors] ]

# compute EOCs and output results
h = lambda size: 1/sqrt(size)
eoc = (len(results[0])-3)*[-1]
for level in range(len(results)):
    print(h(results[level][0]),*results[level],*eoc,"# result-output")
    if level<len(results)-1:
        eoc = [math.log(results[level+1][3+j]/results[level][3+j])/
               math.log(h(results[level+1][0])/h(results[level][0]))
               for j in range(len(eoc))]
print("\n")
