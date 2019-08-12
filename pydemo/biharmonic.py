try:
    import dune.vem
except:
    import sys
    sys.exit(0)
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math
from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate
from dune.fem import parameter
from dune.vem import voronoiCells
from dune.fem.operator import linear as linearOperator
from scipy.sparse.linalg import spsolve

from ufl import *
import dune.ufl

order = 3

dune.fem.parameter.append({"fem.verboserank": 0})

# <markdowncell>
# we can compare different method, e.g., a lagrange space (on the # subtriangulation),
# a bounding box dg space and a conforming/non conforming VEM space

# <codecell>
# Note: suboptimal laplace error for bubble (space is reduced to polorder=3 but could be 4 = ts+2
methods = [ ### "[space,scheme,spaceKwrags]"
            ["vem","vem",{"testSpaces":[ [0],  [order-3,order-2], [order-4] ] }, "C1-non-conforming"],
            # ["vem","vem",{"testSpaces":[ [0],  [order-2,order-2], [order-2] ] }, "C1C0-conforming"],
   ]
parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "newton.linear.verbose": True,
              "newton.verbose": True
              }

# <markdowncell>
# Now we define the model starting with the exact solution:

# <codecell>
uflSpace = dune.ufl.Space(2, dimRange=1)
x = SpatialCoordinate(uflSpace)
exact = as_vector( [sin(2*pi*x[0])**2*sin(2*pi*x[1])**2] )

# next the bilinear form
laplace = lambda w: div(grad(w))
u = TrialFunction(uflSpace)
v = TestFunction(uflSpace)
a = ( inner(laplace(u[0]),laplace(v[0])) ) * dx + inner(grad(u),grad(v)) * dx + inner(u,v) * dx

# finally the right hand side and the boundary conditions
b = laplace(laplace(exact[0]))*v[0] * dx - laplace(exact[0])*v[0] * dx + exact[0]*v[0] * dx
# dbc = [dune.ufl.DirichletBC(uflSpace, [0], i+1) for i in range(4)]
dbc = []
biLaplaceCoeff = 1
diffCoeff = 1
massCoeff = 1

# <markdowncell>
# Now we define a grid build up of voronoi cells around $50$ random points

# We now define a function to compute the solution and the $L^2,H^1$ error
# given a grid and a space

# <codecell>
def compute(grid, space, schemeName):
    # solve the pde
    df = space.interpolate([0],name="solution")
    # df.plot(level=3)
    info = {"linear_iterations":1}
    scheme = create.scheme(schemeName, [a==b, *dbc], space, solver="cg",
                        hessStabilization=biLaplaceCoeff,
                        gradStabilization=diffCoeff,
                        massStabilization=massCoeff,
                      parameters=parameters)
    # info = scheme.solve(target=df)
    jacobian = linearOperator(scheme)
    rhs = space.interpolate([0],name="rhs")
    scheme(df,rhs)
    rhs.as_numpy[:] *= -1
    df.as_numpy[:] = spsolve(jacobian.as_numpy, rhs.as_numpy[:])
    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf)),
           inner(grad(grad(edf)),grad(grad(edf)))]
    errors = [ math.sqrt(e) for e in integrate(grid, err, order=8) ]
    return df, errors, info

# <markdowncell>
# Finally we iterate over the requested methods and solve the problems

# <codecell>
maxLevel = 2
fig = pyplot.figure(figsize=(10*maxLevel*len(methods),10))
figPos = 100+10*2*len(methods)+1
for level in range(maxLevel):
    constructor = cartesianDomain([-0.5,-0.5],[1,1],[40*2**level,40*2**level])
    polyGrid = create.grid("polygrid", constructor, cubes=False )
    for i,m in enumerate(methods):
        space = create.space(m[0], polyGrid, order=order, dimRange=1, storage="fem", **m[2])
        dfs,errors,info = compute(polyGrid, space, m[1])
        print("method:(",m[0],m[2],")",
              "Size: ",space.size, "L^2: ", errors[0], "H^1: ", errors[1],
              "H^2: ", errors[2],
              info["linear_iterations"], flush=True)
        # dfs.plot(figure=(fig,figPos+level*len(methods)+i),gridLines=None, colorbar="horizontal",level=3)
        plot(grad(grad(dfs))[0,0,0],grid=polyGrid,level=3,
             figure=(fig,figPos+level*len(methods)+i),gridLines=None, colorbar="horizontal")
        # interpol = space.interpolate(exact,name="interpolation")
        # plot(grad(grad(interpol))[0,0,0],grid=polyGrid,level=3,
        #      figure=(fig,figPos+level*len(methods)+i+1),gridLines=None, colorbar="horizontal")
pyplot.show()
