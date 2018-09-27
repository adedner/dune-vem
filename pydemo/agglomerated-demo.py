"""Solve the Laplace equation
"""
from __future__ import print_function

import math
import numpy
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree

from ufl import *

import dune.ufl
import dune.fem
import dune.fem.function as gf

import dune.create as create

from voronoi import triangulated_voronoi

dimRange = 1

dune.fem.parameter.append("parameter")

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

# http://zderadicka.eu/voronoi-diagrams/
class Agglomerate:
    def __init__(self,NX,NY=None):
        if NY is not None:
            self.NX = NX
            self.NY = NY
            self.N  = NX*NY
            self.suffix = str(NX)+"_"+str(NY)
            self.cartesian = True
        else:
            self.N = NX
            self.suffix = str(self.N)
            self.cartesian = False

        numpy.random.seed(1234)
        self.voronoi_points = numpy.random.rand(self.N, 2)
        self.voronoi_kdtree = cKDTree(self.voronoi_points)
        self.ind = set()

        vor = Voronoi(self.voronoi_points)
        voronoi_plot_2d(vor).savefig("agglomerate_voronoi"+self.suffix+".pdf", bbox_inches='tight')

    def __call__(self,en):
        p = en.geometry.center
        if self.cartesian:
            index = int(p[0] * self.NX)*self.NY + int(p[1] * self.NY)
        else:  # Voronoi
            test_point_dist, test_point_regions = self.voronoi_kdtree.query([p], k=1)
            index = test_point_regions[0]
        self.ind.add(index)
        return index
    def check(self):
        return len(self.ind)==self.N

parameters = {"linabstol": 1e-8, "reduction": 1e-8,
        "tolerance": 3e-7,
        "krylovmethod": "gmres",
        "istl.preconditioning.method": "amg-ilu",
        "maxiterations": 50,
        "maxlineariterations": 2500,
        "maxlinesearchiterations":50,
        "verbose": "true", "linear.verbose": "false"}

def solve(grid,agglomerate,model,exact,name,space,scheme,penalty=None):
    print("SOLVING: ",name,space,scheme,penalty,flush=True)
    gf_exact = create.function("ufl",grid,"exact",4,exact)
    if agglomerate:
        spc = create.space(space, grid, agglomerate, dimrange=dimRange, order=1, storage="istl")
        assert agglomerate.check(), "missing or too many indices provided by agglomoration object. Should be "+str(NX*NY)+" was "+str(len(ind))
    else:
        spc = create.space(space, grid, dimrange=dimRange, order=1, storage="istl")
    interpol = spc.interpolate( gf_exact, "exact_"+name )
    if penalty:
        df,info = create.scheme(scheme, model, spc, penalty=penalty, solver="cg",
                     parameters={"fem.solver.newton." + k: v for k, v in parameters.items()})\
        .solve(name=name)
    else:
        # scheme = create.scheme(scheme, model, spc, solver="cg",
        #              parameters={"fem.solver.newton." + k: v for k, v in parameters.items()})
        # df = spc.interpolate([-0.5],name=name)
        # scheme.setConstraints(df)
        # info = {}
        # info["linear_iterations"] = 0
        # info["iterations"] = 0
        df,info = create.scheme(scheme, model, spc, solver="cg",
                     parameters={"fem.solver.newton." + k: v for k, v in parameters.items()})\
                .solve(name=name)
    errors = error(grid,df,interpol,exact)
    print("Computed",name+" size:",spc.size,
          "L^2 (s,i):", [errors[0],errors[1]],
          "H^1 (s,i):", [errors[2],errors[3]],
          "linear and Newton iterations:",
          info["linear_iterations"], info["iterations"],flush=True)
    return interpol, df

def compute(agglomerate):
    NX = NY = 60*4
    if agglomerate.cartesian:
        print("Cartesian",flush=True)
        grid       = create.view("ALUSimplex", constructor=dune.grid.cartesianDomain([0, 0], [1, 1], [NX, NY]))
        coarsegrid = create.view("ALUSimplex", constructor=dune.grid.cartesianDomain([0, 0], [1, 1], [agglomerate.NX, agglomerate.NY]))
    else:
        print("Voronoi",flush=True)
        bounding_box = numpy.array([0., 1., 0., 1.]) # [x_min, x_max, y_min, y_max]
        points, triangles = triangulated_voronoi(agglomerate.voronoi_points, bounding_box)
        grid = create.grid("ALUSimplex", {'vertices':points, 'simplices':triangles}, dimgrid=2)

    uflSpace = dune.ufl.Space((grid.dimGrid, grid.dimWorld), dimRange, field="double")
    u = TrialFunction(uflSpace)
    v = TestFunction(uflSpace)
    x = SpatialCoordinate(uflSpace.cell())
    multBnd = 1 # x[0]*(1-x[0])*x[1]*(1-x[1])
    exact = as_vector( [cos(2.*pi*x[0])*cos(2.*pi*x[1])*multBnd,]*dimRange )
    H = lambda w: grad(grad(w))
    a = (inner(grad(u), grad(v)) + inner(u,v)) * dx
    b = ( -(H(exact[0])[0,0]+H(exact[0])[1,1]) + exact[0] ) * v[0] * dx
    if dimRange == 2:
        def nonLinear(w): return 20./(w*w+1.)
        a = a + nonLinear(u[1]) * v[1] * dx
        b = b + ( -(H(exact[1])[0,0]+H(exact[1])[1,1]) + exact[1] + nonLinear(exact[1])) * v[1] * dx
    model = create.model("elliptic", grid, a==b, dune.ufl.DirichletBC(uflSpace,exact,1))

    # print(df_adg.grid) # <- causes error
    if dimRange == 1:
        interpol_vem, df_vem = solve(grid,agglomerate,model,exact,"vem","AgglomeratedVEM","vem")
    else:
        interpol_vem, df_vem = interpol_adg.copy("tmp_interpol"), df_adg.copy("tmp")

    if agglomerate.cartesian:
        print("NX,NY,NX*NY,...",NX,NY,NX*NY,agglomerate.NX,agglomerate.NY,agglomerate.N,flush=True)
        interpol_lag, df_lag = solve(coarsegrid,None, model,exact,
                "h1","Lagrange","h1")
        interpol_dg,  df_dg  = solve(coarsegrid,None, model,exact,
                "dgonb","DGONB","dg",penalty=10)
        penalty = max(agglomerate.NX,agglomerate.NY) / min(NX,NY)
        interpol_adg, df_adg = solve(grid,agglomerate, model,exact,
                "adg","AgglomeratedDG","dg",penalty=10*penalty)
    else:
        interpol_lag, df_lag = solve(grid,None, model,exact,
                "h1","Lagrange","h1")
        interpol_dg,  df_dg  = solve(grid,None, model,exact,
                "dgonb","DGONB","dg",penalty=10)
        interpol_adg, df_adg = solve(grid,agglomerate, model,exact,
                "adg","AgglomeratedDG","dg",penalty=1)

    if agglomerate.cartesian:
        coarsegrid.writeVTK("agglomerate_coarse"+agglomerate.suffix,
            pointdata=[ df_dg, df_lag ] )
        grid.writeVTK("agglomerate"+agglomerate.suffix,
            pointdata=[ df_adg, interpol_adg,
                        df_vem, interpol_vem ],
            celldata =[ create.function("local",grid,"cells",1,lambda en,x: [agglomerate(en)]) ])
    else:
        grid.writeVTK("agglomerate"+agglomerate.suffix,
            pointdata=[ df_adg, interpol_adg,
                        df_vem, interpol_vem,
                        df_dg, df_lag ],
            celldata =[ create.function("local",grid,"cells",1,lambda en,x: [agglomerate(en)]) ])

print("*******************************************************")
print("Test 1: Agglomerate(251)")
compute(Agglomerate(251))
print("*****************")
print("Test 1: Agglomerate(999)")
compute(Agglomerate(999))
print("*****************")
print("Test 1: Agglomerate(4013)")
compute(Agglomerate(4013))
if False:
    # these test don't make a lot of sense since 'hanging' nodes are not removed
    print("*******************************************************")
    print("Test 2: Agglomerate(15,15)")
    compute(Agglomerate(15,15))
    print("*****************")
    print("Test 2: Agglomerate(30,30)")
    compute(Agglomerate(30,30))
    print("*****************")
    print("Test 2: Agglomerate(60,60)")
    compute(Agglomerate(60,60))
    print("*******************************************************")
