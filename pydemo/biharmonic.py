import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot
import math
from dune import create
from dune.grid import cartesianDomain
from dune.fem.function import gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.space import lagrange
from dune.fem import integrate
from dune.fem import parameter
from dune.vem import voronoiCells
from scipy.sparse.linalg import spsolve

from ufl import *
import dune.ufl

maxLevel     = 4
order        = 3
epsilon      = 1
laplaceCoeff = 1
mu           = 1

dune.fem.parameter.append({"fem.verboserank": 0})

# <markdowncell>
# we can compare different method, e.g., a lagrange space (on the # subtriangulation),
# a bounding box dg space and a conforming/non conforming VEM space

# <codecell>
# Note: suboptimal laplace error for bubble (space is reduced to polorder=3 but could be 4 = ts+2
methods = [ ### "[space,scheme,spaceKwrags]"
        ["vem","vem",{"order":order, "testSpaces":[ [0,0],[order-4,order-3], [order-4] ] }, "C1-conforming"],
        # ["vem","vem",{"order":order, "testSpaces":[ [0],  [order-3,order-2], [order-4] ] }, "C1-non-conforming"],
        # ["vem","vem",{"order":order, "testSpaces":[ [0],  [order-2,order-2], [order-2] ] }, "C1C0-conforming"],
        # ["vem","vem",{"order":order, "testSpaces":[ [0],  [order-3,order-2], [order-3] ] }, "C1mod-conforming"],
          ]
if epsilon == 0:
    methods += [
            ["vem","vem",{"order":order, "testSpaces":[ [0],  [order-2,-1], [order-2] ] },      "C0-conforming"],
            # ["vem","vem",{"order":order-1, "testSpaces":[ [0],  [order-3,-1], [order-3] ] },      "C0pm1-conforming"],
               ]
parameters = {"linear.tolerance": 1e-12,
              "linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "linear.verbose": False,
              "nonlinear.verbose": False
              }

# <markdowncell>
# Now we define the model starting with the exact solution:

# <codecell>
uflSpace = dune.ufl.Space(2, dimRange=1)
x = SpatialCoordinate(uflSpace)
exact = as_vector( [sin(2.3*pi*x[0])*cos(2*pi*x[1])] )
# exact = as_vector( [x[0]**3] )

# next the bilinear form
# Note: for function which continuous derivatives we have
#       laplace(u)*laplace(v)*dx = inner(u,v)*dx
#       as can be seen by using integration by parts on the mixed terms on the right
#       and using continuety of u,v
#       For the non conforming spaces we don't have continuety of the
#       derivatives so the equivalence does not hold and one should use the
#       right hand side directly to obtain a coercive bilinear form w.r.t.
#       the norm on H^2 (the left is not a norm in this case).
#       For computing the forcing term 'b' both formula are fine since
#       'exact' is smooth enough.
laplace = lambda w: div(grad(w))
H = lambda w: grad(grad(w))
u = TrialFunction(uflSpace)
v = TestFunction(uflSpace)
if False:
    a = ( epsilon*inner(H(u[0]),H(v[0])) +\
          laplaceCoeff*inner(grad(u),grad(v)) +\
          mu*inner(u,v)
        ) * dx
    # finally the right hand side and the boundary conditions
    b = ( epsilon*laplace(laplace(exact[0])) -\
          laplaceCoeff*laplace(exact[0]) +\
          mu*exact[0] ) * v[0] * dx
    biLaplaceCoeff = epsilon
    diffCoeff      = laplaceCoeff
    massCoeff      = mu
else:
    kappa = 1./(1+dot(x,x))
    beta  = exp(-x[0]*x[1])
    gamma = sin(dot(x,x))**2
    a = ( kappa*inner(H(u[0]),H(v[0])) +\
          beta*inner(grad(u),grad(v)) +\
          gamma*inner(u,v)
        ) * dx
    q = sum([ H(kappa*H(exact[0])[i,j])[i,j] for i in range(2) for j in range(2) ])
    b = ( q -\
          laplaceCoeff*div(beta*grad(exact[0])) +\
          mu*gamma*exact[0] ) * v[0] * dx
    biLaplaceCoeff = kappa
    diffCoeff      = laplaceCoeff*beta
    massCoeff      = mu*gamma

dbc = [dune.ufl.DirichletBC(uflSpace, exact, i+1) for i in range(4)]

# <markdowncell>
# Now we define a grid build up of voronoi cells around $50$ random points

# We now define a function to compute the solution and the $L^2,H^1$ error
# given a grid and a space

# <codecell>
def compute(grid, space, schemeName):
    # solve the pde
    df = space.function(name="solution") # space.interpolate([0],name="solution")
    # df.plot(level=3)
    info = {"linear_iterations":1}
    if False:
        df.interpolate(exact)
        # df.plot(level=3)
        # uflFunction(grid,name="grad",order=1,ufl=grad(df[0])[0]).plot(level=3)
        # uflFunction(grid,name="grad",order=1,ufl=grad(df[0])[1]).plot(level=3)
    else:
        scheme = create.scheme(schemeName, [a==b, *dbc], space,
                            # solver="cg",
                            boundary="derivative",
                            solver=("suitesparse","umfpack"),
                            hessStabilization=biLaplaceCoeff,
                            gradStabilization=diffCoeff,
                            massStabilization=massCoeff,
                            parameters=parameters)
        scheme.setConstraints(df)
        for d in df.dofVector:
            print(d)

        spcLag = lagrange(space.gridView,space.order)
        fig,axs = pyplot.subplots(nrows=1,ncols=3)
        spcLag.interpolate(df,name="v").plot(level=3,figure=(fig,axs[0]))
        spcLag.interpolate(grad(df[0])[0],name="dx").plot(level=3,figure=(fig,axs[1]))
        spcLag.interpolate(grad(df[0])[1],name="dy").plot(level=3,figure=(fig,axs[2]))
        pyplot.show()
        # info = scheme.solve(target=df)
        jacobian = scheme.linear()
        rhs = space.function(name="rhs")
        scheme(df,rhs)
        rhs.as_numpy[:] *= -1
        df.as_numpy[:] = spsolve(jacobian.as_numpy, rhs.as_numpy[:])
    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf)),
           inner(grad(grad(edf)),grad(grad(edf)))]
    errors = [ math.sqrt(e) for e in integrate(err, order=8) ]
    return df, errors, info

# <markdowncell>
# Finally we iterate over the requested methods and solve the problems

# <codecell>
# fig = pyplot.figure(figsize=(10*maxLevel,10*len(methods)))
# figPos = 100*len(methods)+10*maxLevel+1
results = []
for level in range(maxLevel):
    # constructor = cartesianDomain([0,0],[1,1],[2*2**level,2*2**level])
    constructor = cartesianDomain([0,0],[1,1],[2,2])
    polyGrid = create.grid("agglomerate", constructor, cubes=False )
    # polyGrid = create.grid("agglomerate",
         # voronoiCells(constructor,4*2**level*2**level,load="voronoiseeds",
         #        show=False,lloyd=5), convex=True )

    # N = 10*3*2**level+1 # needs to be of the form 6*i+1
    # polyGrid = hexaGrid(N, 1, 1)

    res = []
    for i,m in enumerate(methods):
        space = create.space(m[0], polyGrid, dimRange=1, storage="numpy", **m[2])
        dfs,errors,info = compute(polyGrid, space, m[1])
        # dfs.plot(level=3)
        print("[",level,"]","method:(",m[0],m[2],")",
              "Sizes (polys,simplex,dofs): ",polyGrid.hierarchicalGrid.agglomerate.size, polyGrid.size(0), space.size, "L^2: ", errors[0], "H^1: ", errors[1],
              "H^2: ", errors[2],
              info["linear_iterations"], flush=True)
        res += [ [polyGrid.hierarchicalGrid.agglomerate.size,space.size,errors[0],errors[1],errors[2]] ]
        # dfs.plot(level=4)
        # plot(grad(grad(dfs))[0,0,0],grid=polyGrid,level=3,
        # plot(dfs,grid=polyGrid,level=3,
        #      figure=(fig,figPos+i*maxLevel+level),gridLines=None, colorbar="horizontal")
        # interpol = space.interpolate(exact,name="interpolation")
        # plot(grad(grad(interpol))[0,0,0],grid=polyGrid,level=3,
        #      figure=(fig,figPos+level*len(methods)+i+1),gridLines=None, colorbar="horizontal")
    results += [res]
h = lambda size: 1/sqrt(size)
print("# p="+str(order)+", eps="+str(epsilon)+", laplace="+str(laplaceCoeff))
for i,m in enumerate(methods):
    eoc = [-1,-1,-1]
    for level in range(0,maxLevel):
        print(h(results[level][i][0]),*results[level][i],*eoc)
        if level<maxLevel-1:
            eoc = [math.log(results[level+1][i][2+j]/results[level][i][2+j])/
                   math.log(h(results[level+1][i][0])/h(results[level][i][0]))
                   for j in range(3)]
    print("\n")
# pyplot.show()
