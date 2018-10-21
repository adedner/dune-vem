from __future__ import print_function

import math
import numpy
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import numpy.linalg
import scipy.sparse.linalg

from ufl import *

from dune.grid import cartesianDomain
import dune.ufl
import dune.fem
import dune.fem.function as gf

import dune.create as create

from voronoi import triangulated_voronoi
import graph
import holes

testVoronoi = False
testMetis = False

dimRange = 1

# dune.fem.parameter.append("parameter")
dune.fem.parameter.append({"fem.verboserank": 0})
dune.fem.parameter.append({"fem.solver.verbose":False})

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
    def __init__(self,N,constructor=None,version="voronoi"):
        self.N = N
        self.suffix = str(self.N)
        self.version = version
        if version == "metis" or version == "metisVoronoi":
            self.suffix = "m" + self.suffix
            self.grid = create.grid("ALUSimplex", constructor=constructor, dimgrid=2)
            self.parts = graph.partition(self.grid, nparts=self.N,
                    contig=True, iptype="node", objtype="vol")
            if version == "metisVoronoi":
                voronoi_points = numpy.zeros((self.N,2))
                weights = numpy.zeros(self.N)
                for en in self.grid.elements:
                    index = self.parts[ self.grid.indexSet.index(en) ]
                    voronoi_points[index] += en.geometry.center
                    weights[index] += 1
                for p,w in zip(voronoi_points,weights):
                    p /= w
                voronoi_kdtree = cKDTree(voronoi_points)
                for en in self.grid.elements:
                    p = en.geometry.center
                    test_point_dist, test_point_regions = voronoi_kdtree.query([p], k=1)
                    index = test_point_regions[0]
                    self.parts[ self.grid.indexSet.index(en) ] = index
                vor = Voronoi(voronoi_points)
                voronoi_plot_2d(vor).savefig("metis_voronoi"+self.suffix+".pdf", bbox_inches='tight')
        elif version == "voronoi":
            self.suffix = "v" + self.suffix
            numpy.random.seed(1234)
            self.voronoi_points = numpy.random.rand(self.N, 2)
            self.voronoi_kdtree = cKDTree(self.voronoi_points)
            vor = Voronoi(self.voronoi_points)
            voronoi_plot_2d(vor).savefig("agglomerate_voronoi"+self.suffix+".pdf", bbox_inches='tight')
            if constructor == None:
                bounding_box = numpy.array([0., 1., 0., 1.]) # [x_min, x_max, y_min, y_max]
                points, triangles = triangulated_voronoi(self.voronoi_points, bounding_box)
                self.grid = create.grid("ALUSimplex", {'vertices':points, 'simplices':triangles}, dimgrid=2)
            else:
                self.grid = create.grid("ALUSimplex", constructor=constructor, dimgrid=2)
        else:
            self.suffix = "c" + self.suffix
            self.grid = create.grid("yasp", constructor=constructor)
            self.division = constructor.division
        self.ind = set()

    def __call__(self,en):
        if self.version == "metis" or self.version == "metisVoronoi":
            index = self.parts[ self.grid.indexSet.index(en) ]
            self.ind.add(index)
        elif self.version == "voronoi":
            p = en.geometry.center
            test_point_dist, test_point_regions = self.voronoi_kdtree.query([p], k=1)
            index = test_point_regions[0]
            self.ind.add(index)
        else:
            idx = self.grid.indexSet.index(en)
            nx = int(idx / self.division[1])
            ny = int(idx % self.division[1])
            print("nx,ny",nx,ny)
            nx = int(nx/self.division[0]*self.N[0])
            ny = int(ny/self.division[1]*self.N[1])
            index = nx*self.N[1]+ny
            print("index",index)
        return index
    def check(self):
        return len(self.ind)==self.N

newtonParameters = {"linabstol": 1e-8, "reduction": 1e-8, "tolerance": 3e-5,
              "maxiterations": 50,
              "maxlineariterations": 2500,
              "maxlinesearchiterations":50,
              "verbose": "true", "linear.verbose": "true",
              }
parameters = {"fem.solver.newton." + k: v for k, v in newtonParameters.items()}
parameters["istl.preconditioning.method"] = "ilu"

def solve(grid,agglomerate,model,exact,name,space,scheme,order=1,penalty=None):
    print("SOLVING: ",name,space,scheme,penalty,flush=True)
    gf_exact = create.function("ufl",grid,"exact",4,exact)
    if agglomerate:
        print("agglomerate",agglomerate)
        spc = create.space(space, grid, agglomerate, dimrange=dimRange,
                order=order, storage="fem")
        assert agglomerate.check(), "missing or too many indices provided by agglomoration object. Should be "+str(agglomerate.N)+" was "+str(len(agglomerate.ind))
    else:
        print("no agglomerate")
        spc = create.space(space, grid, dimrange=dimRange, order=order,
                storage="fem")
    print("interpolate")
    interpol = spc.interpolate( gf_exact, "exact_"+name )
    if penalty:
        print("penalty",penalty)
        scheme = create.scheme(scheme, model, spc, penalty=penalty,
                solver=("suitesparse","umfpack"),
                parameters=parameters)
        df,info = scheme.solve(name=name)
    else:
        print("no penalty")
        scheme = create.scheme(scheme, model, spc,
                solver=("suitesparse","umfpack"),
                parameters=parameters)
        df,info = scheme.solve(name=name)
    print("computing errors")
    errors = error(grid,df,interpol,exact)
    if spc.size < 0:
        A = scheme.assemble(df).as_numpy.transpose()
        try:
            c1 = numpy.linalg.cond(A.todense())
        except:
            c1 = -1
        norm_A    = scipy.sparse.linalg.norm(A)
        norm_invA = scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(A))
        c2        = norm_A*norm_invA
    else:
        c1 = -1
        c2 = -1
    print("Computed",name+" size:",spc.size,
          "L^2 (s,i):", [errors[0],errors[1]],
          "H^1 (s,i):", [errors[2],errors[3]],
          "cond:",[c1,c2],
          "linear and Newton iterations:",
          info["linear_iterations"], info["iterations"],flush=True)
    return interpol, df

def compute(agglomerate,filename):
    grid = agglomerate.grid
    uflSpace = dune.ufl.Space((grid.dimGrid, grid.dimWorld), dimRange, field="double")
    u = TrialFunction(uflSpace)
    v = TestFunction(uflSpace)
    x = SpatialCoordinate(uflSpace.cell())
    multBnd = 1 # x[0]*(1-x[0])*x[1]*(1-x[1]) # to get zero bcs
    exact = as_vector( [cos(2.*pi*x[0])*cos(2.*pi*x[1])*multBnd,]*dimRange )
    exact = as_vector( [cos(x[0])*cos(x[1])*multBnd,]*dimRange )
    # exact = as_vector( [0] )
    H = lambda w: grad(grad(w))
    a = (inner(grad(u), grad(v)) + inner(u,v)) * dx
    b = ( -(H(exact[0])[0,0]+H(exact[0])[1,1]) + exact[0] ) * v[0] * dx
    # b = v[0]*dx
    if dimRange == 2:
        def nonLinear(w): return 20./(w*w+1.)
        a = a + nonLinear(u[1]) * v[1] * dx
        b = b + ( -(H(exact[1])[0,0]+H(exact[1])[1,1]) + exact[1] + nonLinear(exact[1])) * v[1] * dx
    model = create.model("elliptic", grid, a==b, dune.ufl.DirichletBC(uflSpace,exact,1))

    # print(df_adg.grid) # <- causes error
    interpol_lag, df_lag = solve(grid,None, model,exact,
            "h1","Lagrange","h1",order=1)
    interpol_dg,  df_dg  = solve(grid,None, model,exact,
            "dgonb","DGONB","dg",penalty=10)
    interpol_adg, df_adg = solve(grid,agglomerate, model,exact,
            "adg","AgglomeratedDG","dg",order=1,penalty=0.1)

    if dimRange == 1:
        interpol_vem, df_vem = solve(grid,agglomerate,model,exact,"vem","AgglomeratedVEM","vem")
    else:
        interpol_vem, df_vem = None, None # interpol_adg.copy("tmp_interpol"), df_adg.copy("tmp")
        print("Skippting vem: not yet implemented for dimRange>1")

    grid.writeVTK(filename+agglomerate.suffix,
        pointdata=[ df_adg, interpol_adg,
                    df_vem, interpol_vem,
                    df_dg, df_lag ],
        celldata =[ create.function("local",grid,"cells",1,lambda en,x: [agglomerate(en)]) ])

# test of higher order vem spaces
if True:
    constructor = cartesianDomain([0,0],[1,1],[12,12])
    agglomerate = Agglomerate([4,4],version="cartesian",constructor=constructor)
    space = create.space("agglomeratedvem", agglomerate.grid, agglomerate,
            dimrange=1, order=1, storage="fem")
    dfs = [space.interpolate(lambda x:(x[0]*x[1])**2,name="one")]
    print("order=1, size=",space.size)
    space = create.space("agglomeratedvem", agglomerate.grid, agglomerate,
            dimrange=1, order=2, storage="fem")
    dfs += [space.interpolate(lambda x:(x[0]*x[1])**2,name="two")]
    print("order=2, size=",space.size)
    space = create.space("agglomeratedvem", agglomerate.grid, agglomerate,
            dimrange=1, order=3, storage="fem")
    dfs += [space.interpolate(lambda x:(x[0]*x[1])**2,name="three")]
    print("order=3",space.size)
    space.grid.writeVTK("test", pointdata=dfs)
    sys.exit(0)

if testVoronoi:
    print("*******************************************************")
    constructor = None # cartesianDomain([0,0],[3,2],[200,200])
    print("Test 1: Voronoi(251)")
    # compute(Agglomerate(251,version="voronoi",constructor=constructor),
    #         "voronoi")
    print("*****************")
    print("Test 2: Voronoi(999)")
    # compute(Agglomerate(999,version="voronoi",constructor=constructor),
    #         "voronoi")
    print("*****************")
    print("Test 3: Voronoi(4013)")
    # compute(Agglomerate(4013,version="voronoi",constructor=constructor),
    #         "voronoi")

if testMetis:
    print("*******************************************************")
    # constructor = cartesianDomain([0,0],[1,1],[200,200])
    print("Test 1: Metis(251)")
    constructor = holes.get(area=0.05,plotDomain=False)
    compute(Agglomerate(251,version="metis",constructor=constructor),
            "metis")
    print("*****************")
    print("Test 2: Metis(999)")
    constructor = holes.get(area=0.01,plotDomain=False)
    compute(Agglomerate(999,version="metisVoronoi",constructor=constructor),
            "metisVoronoi")
    compute(Agglomerate(999,version="metisVoronoi",constructor=constructor),
            "metisVoronoi")
    print("*****************")
    print("Test 3: Metis(4013)")
    # compute(Agglomerate(4013,version="metis",constructor=constructor),
    #         "metis")
