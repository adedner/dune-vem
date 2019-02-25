from __future__ import print_function

import math, sys, pickle

from ufl import *
import dune.ufl

from dune import create
from dune.grid import cartesianDomain
from dune.fem import parameter
from dune.vem import voronoiCells

dimRange = 1
# polOrder, endEoc = 1,8
polOrder, endEoc = 1,8
# polOrder, endEoc = 3,6
# polOrder, endEoc = 4,6

dune.fem.parameter.append({"fem.verboserank": 0})

def plot(grid, solution):
    try:
        from matplotlib import pyplot
        from numpy import amin, amax, linspace

        triangulation = grid.triangulation(4)
        data = solution.pointData(4)

        levels = linspace(amin(data[:,0]), amax(data[:,0]), 256)

        pyplot.gca().set_aspect('equal')
        pyplot.triplot(grid.triangulation(), antialiased=True, linewidth=0.2, color='black')
        pyplot.tricontourf(triangulation, data[:,0], cmap=pyplot.cm.rainbow, levels=levels)
        pyplot.show()
    except ImportError:
        pass

def error(grid, df, interp, exact):
    edf = exact-df
    ein = exact-interp
    errors = create.function("ufl", grid, "error", 5,
            [ dot(edf,edf), dot(ein,ein),
              inner(grad(edf),grad(edf)), inner(grad(ein),grad(ein))
            ] ).integrate()
    return [ math.sqrt(e) for e in errors ]

parameters = {"newton.linear.tolerance": 1e-9,
              "newton.linear.verbose": "true",
              "newton.tolerance": 1e-7,          # can't be smaller for high order
              "newton.maxiterations": 20,
              "newton.maxlinesearchiterations":50,
              "newton.verbose": "true",
              "newton.linear.preconditioning.method": "lu",
              "penalty": 8*polOrder*polOrder
              }

methods = [ # "[space,scheme]"
            #["lagrange","h1"],
            ["vem","vem"],
            #["bbdg","bbdg"],
            #["dgonb","dg"]
   ]

h1errors  = []
l2errors  = []
spaceSize = []

def solve(polyGrid,model,exact,space,scheme,order=1,penalty=None):
    try:
        grid = polyGrid.grid
    except AttributeError:
        grid = polyGrid
    print("SOLVING: ",space,scheme,penalty,flush=True)
    try:
        spc = create.space(space, polyGrid, dimrange=dimRange, order=order,
                storage="petsc")
    except AttributeError:
        spc = create.space(space, grid, dimrange=dimRange, order=order,
                storage="petsc")
    interpol = spc.interpolate( exact, "interpol_"+space )
    scheme = create.scheme(scheme, model, spc,
                solver="cg",
                #         ("suitesparse","umfpack"),
                parameters=parameters)
    df = spc.interpolate([0],name=space)
    info = scheme.solve(target=df)
    errors = error(grid,df,interpol,exact)
    print("Computed",space," size:",spc.size,
          "L^2 (s,i):", [errors[0],errors[1]],
          "H^1 (s,i):", [errors[2],errors[3]],
          "linear and Newton iterations:",
          info["linear_iterations"], info["iterations"],flush=True)
    global h1errors, l2errors, spaceSize
    l2errors  += [errors[0],errors[1]]
    h1errors  += [errors[2],errors[3]]
    spaceSize += [spc.size]
    return interpol, df

def compute(polyGrid):
    grid = polyGrid.grid
    uflSpace = dune.ufl.Space((grid.dimGrid, grid.dimWorld), dimRange, field="double")
    u = TrialFunction(uflSpace)
    v = TestFunction(uflSpace)
    x = SpatialCoordinate(uflSpace.cell())
    # problem 1
    ### trivial Neuman bc
    #factor = 2 # change to 2 for higher order approx
    #exact  = as_vector( [cos(factor*pi*x[0])*cos(factor*pi*x[1])] )
    #### zero boundary conditions
    #exact *= x[0]*x[1]*(2-x[0])*(2-x[1])
    #### non zero and non trivial Neuman boundary conditions
    #exact += as_vector( [sin(factor*x[0]*factor*x[1])] )
    #H = lambda w: grad(grad(w))
    #a = (inner(grad(u), grad(v)) + inner(u,v)) * dx
    #b = ( -(H(exact[0])[0,0]+H(exact[0])[1,1]) + exact[0] ) * v[0] * dx
    ## problem 2: dummy quasilinear problem:
    exact = as_vector ( [  (x[0] - x[0]*x[0] ) * (x[1] - x[1]*x[1] ) ] )
    Dcoeff = lambda u: 1.0 + u[0]**2
    a = (Dcoeff(u)* inner(grad(u), grad(v)) ) * dx
    b = -div( Dcoeff(u) * grad(exact[0]) ) * v[0] * dx
    model = create.model("elliptic", grid, a==b
            , *[dune.ufl.DirichletBC(uflSpace, exact, i+1) for i in range(4)]
            )
    dfs = []
    for m in methods:
        dfs += solve(polyGrid,model,exact,*m,order=polOrder)

    grid.writeVTK(polyGrid.agglomerate.suffix,
        pointdata=dfs,
        celldata =[ create.function("local",grid,"cells",1,lambda en,x: [polyGrid.agglomerate(en)]) ])
    grid.writeVTK("s"+polyGrid.agglomerate.suffix,subsampling=polOrder-1,
        pointdata=dfs,
        celldata =[ create.function("local",grid,"cells",1,lambda en,x: [polyGrid.agglomerate(en)]) ])


start = 4
for i in range(endEoc-start):
    print("*******************************************************")
    n = 2**(i+start)
    N = 2*n
    print("Test: ",n,N)
    constructor = cartesianDomain([0,0],[1,1],[N,N])
    # polyGrid = create.grid("polygrid",constructor,[n,n])
    # polyGrid = create.grid("polygrid",constructor,n*n)
    polyGrid = create.grid("polygrid", voronoiCells(constructor,n*n,"voronoiseeds",True) )
    # polyGrid = create.grid("polygrid",constructor)
    compute(polyGrid)
    if i>0:
        l = len(methods)
        for j in range(2*l):
            l2eoc = math.log( l2errors[2*l*i+j]/l2errors[2*l*(i-1)+j] ) / math.log(0.5)
            h1eoc = math.log( h1errors[2*l*i+j]/h1errors[2*l*(i-1)+j] ) / math.log(0.5)
            print("EOC",methods[int(j/2)][0],j,l2eoc,h1eoc)
    with open("errors_p"+str(polOrder)+".dump", 'wb') as f:
        pickle.dump([methods,spaceSize,l2errors,h1errors], f)
    print("*******************************************************")
