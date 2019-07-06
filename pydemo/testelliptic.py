# <markdowncell>
# # Laplace problem
#
# We first consider a simple Laplace problem with Dirichlet boundary conditions
# \begin{align*}
#   -\Delta u &= f, && \text{in } \Omega, \\
#           u &= g, && \text{on } \partial\Omega,
# \end{align*}
# with $\Omega=[-\frac{1}{2},1]^2$ and choosing the forcing and the boundary conditions
# so that the exact solution is equal to
# \begin{align*}
#   u(x,y) &= xy\cos(\pi xy)
# \end{align*}

# First some setup code:
# <codecell>
try:
    import dune.vem
except:
    import sys
    sys.exit(0)
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math, pickle, gc
from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate
from dune.fem import parameter
from dune.vem import voronoiCells
from dune.vem import space as generalSpace

from ufl import *
import dune.ufl

dune.fem.parameter.append({"fem.verboserank": 0})

# <markdowncell>
# we can compare different method, e.g., a lagrange space (on the # subtriangulation),
# a bounding box dg space and a conforming/non conforming VEM space

# <codecell>
orders = [5] # range(1,6)
resolutions = [2*2**n for n in range(8)]
methods = lambda order:\
          [ ### "[spaceVersion,scheme]"
            # ["continuousSimplex","h1"],
            ["continuous","vem"],
            # ["non-conforming","vem"],
            # ["dg","bbdg"],
            # [[0,order-2,order-3],"vem"],
            # [[0,order-2,order-1],"vem"],  # bubble fails with N=160, order=5
   ]
# <markdowncell>
# Now we define the model starting with the exact solution:

# <codecell>
uflSpace = dune.ufl.Space(2, dimRange=1)
x = SpatialCoordinate(uflSpace)
exact = as_vector( [x[0]*x[1] * cos(pi*x[0]*x[1])] )
exact = exact*10*((x[0]-1)*(x[0]+0.)*(x[1]-1)*(x[1]+0.))

# next the bilinear form
u = TrialFunction(uflSpace)
v = TestFunction(uflSpace)
massCoeff = 0   # 1+sin(dot(x,x))
diffCoeff = 1   # -0.9*cos(dot(x,x))
a   = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx
b   = (-diffCoeff*div(grad(exact[0])) + massCoeff*exact[0] ) * v[0] * dx
dbc = [dune.ufl.DirichletBC(uflSpace, exact, i+1) for i in range(4)]

# <markdowncell>
# Now we define a grid build up of voronoi cells around $50$ random points
# We now define a function to compute the solution and the $L^2,H^1$ error
# given a grid and a space

# <codecell>
def compute(grid, order, spaceArgs, schemeName):
    print("setting up space...",flush=True)
    space = generalSpace(polyGrid, order=order,
                         version=spaceArgs, dimRange=1, storage="istl")
    print("...done",flush=True)
    # solve the pde
    parameters = {"newton.linear.tolerance": 1e-12,
                  "newton.linear.preconditioning.method": "ilu",
                  "penalty": 8*space.order**2,  # for the bbdg scheme
                  "newton.linear.maxiterations": 20000,
                  "newton.linear.verbose": True,
                  "newton.verbose": False
                  }
    if schemeName == "vem":
        scheme = create.scheme(schemeName, [a==b, *dbc], space,
                       solver="cg", # ("suitesparse","umfpack"),
                       gradStabilization=diffCoeff,
                       massStabilization=massCoeff,
                       parameters=parameters)
    else:
        scheme = create.scheme(schemeName, [a==b, *dbc], space,
                       solver="cg", # ("suitesparse","umfpack"),
                       parameters=parameters)
    df = space.interpolate([0],name="solution")

    if False:
        from dune.fem.operator import linear as linearOperator
        A = linearOperator(scheme).as_numpy.todense()
        import numpy as np
        print( "symmetry:", np.allclose(A, A.T, rtol=1e-7, atol=1e-9),flush=True )
        l = np.linalg.eigvals(A)
        l.sort()
        print( "eigenvalues:", l,flush=True )

    # info = {"linear_iterations":0}
    # df.interpolate(exact)
    info = scheme.solve(target=df)

    # compute the error
    edf = exact-df
    errors = [ math.sqrt(e) for e in
               integrate(grid, [inner(edf,edf),inner(grad(edf),grad(edf))], order=8) ]

    return df, errors, info

# <markdowncell>
# Finally we iterate over the requested methods and solve the problems

# <codecell>
results = []
for N in resolutions:
    constructor = cartesianDomain([0,0],[1,1],[N,N])
    constructor = voronoiCells(constructor,N*N,"voronoiseeds",True)
    polyGrid = create.grid("polygrid", constructor, cubes=False )
    for order in orders:
        # fig = pyplot.figure(figsize=(10*len(methods(order)),10))
        # figPos = 100+10*len(methods(order))+1
        for i,m in enumerate(methods(order)):
            print("Calculating: (",N,order,"):",i,m,flush=True)
            dfs,errors,info = compute(polyGrid, order,m[0],m[1])
            print("method:",m[0]," order: ",order," cells: ",N*N," simplex: ", polyGrid.size(0),
                  "size: ",dfs.size, "L^2: ", errors[0], "H^1: ", errors[1],
                  info["linear_iterations"], flush=True)
            print("Done")
            results += [N,order,m,polyGrid.size(0),dfs.size,errors,info]
            # dfs.plot(figure=(fig,figPos+i),gridLines="black", colorbar="horizontal")
            del dfs
            print("collected: ",gc.collect())
        # pyplot.show()
        pickle.dump(results,open('testelliptic.dump','wb'))
