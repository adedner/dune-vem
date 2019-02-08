from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
logger = logging.getLogger(__name__)

from dune.generator import Constructor, Method
import dune.common.checkconfiguration as checkconfiguration
import dune

def bbdgSpace(viewagglomerate, order=1, dimrange=1, field="double", storage="adaptive", **unused):
    """create a discontinous galerkin space over an agglomerated grid

    Args:
        view: the underlying grid view
        agglomerated: grouping of elements into polygons
        order: polynomial order of the finite element functions
        dimrange: dimension of the range space
        field: field of the range space
        storage: underlying linear algebra backend

    Returns:
        Space: the constructed Space
    """

    from dune.fem.space import module, addStorage
    if dimrange < 1:
        raise KeyError(\
            "Parameter error in DiscontinuosGalerkinSpace with "+
            "dimrange=" + str(dimrange) + ": " +\
            "dimrange has to be greater or equal to 1")
    if order < 0:
        raise KeyError(\
            "Parameter error in DiscontinuousGalerkinSpace with "+
            "order=" + str(order) + ": " +\
            "order has to be greater or equal to 0")
    if field == "complex":
        field = "std::complex<double>"

    try:
        view = viewagglomerate[0]
        agglomerate = viewagglomerate[1]
    except:
        agglomerate = viewagglomerate.agglomerate
        view = viewagglomerate.grid

    includes = [ "dune/vem/agglomeration/dgspace.hh" ] + view._includes
    dimw = view.dimWorld
    viewType = view._typeName

    gridPartName = "Dune::FemPy::GridPart< " + view._typeName + " >"
    typeName = "Dune::Vem::AgglomerationDGSpace< " +\
      "Dune::Fem::FunctionSpace< double, " + field + ", " + str(dimw) + ", " + str(dimrange) + " >, " +\
      gridPartName + ", " + str(order) + " >"

    constructor = Constructor(
                   ['pybind11::object gridView',
                    'const pybind11::function agglomerate'],
                   ['auto agglo = new Dune::Vem::Agglomeration<' + gridPartName + '>',
                    '         (Dune::FemPy::gridPart<' + viewType + '>(gridView), [agglomerate](const auto& e) { return agglomerate(e).template cast<unsigned int>(); } ); ',
                    'auto obj = new DuneType( *agglo );',
                    'pybind11::cpp_function remove_agglo( [ agglo ] ( pybind11::handle weakref ) {',
                    '  delete agglo;',
                    '  weakref.dec_ref();',
                    '} );',
                    '// pybind11::handle nurse = pybind11::detail::get_object_handle( &obj, pybind11::detail::get_type_info( typeid( ' + typeName + ' ) ) );',
                    '// assert(nurse);',
                    'pybind11::weakref( agglomerate, remove_agglo ).release();',
                    'return obj;'],
                   ['"gridView"_a', '"agglomerate"_a',
                    'pybind11::keep_alive< 1, 2 >()'] )

    spc = module(field, includes, typeName, constructor, storage=storage).Space(view, agglomerate)
    addStorage(spc, storage)
    return spc.as_ufl()

from dune.fem.scheme import dg
def bbdgScheme(model, space, penalty=0, solver=None, parameters={}):
    spaceType = space._typeName
    penaltyClass = "Dune::Vem::BBDGPenalty<"+spaceType+">"
    return dg(model,space,penalty,solver,parameters,penaltyClass)

def vemSpace(viewagglomerate, order=1, dimrange=1, field="double", storage="adaptive", **unused):
    """create a virtual element space over an agglomerated grid

    Args:
        view: the underlying grid view
        agglomerated: grouping of elements into polygons
        order: polynomial order of the finite element functions
        dimrange: dimension of the range space
        field: field of the range space
        storage: underlying linear algebra backend

    Returns:
        Space: the constructed Space
    """

    from dune.fem.space import module, addStorage
    if dimrange < 1:
        raise KeyError(\
            "Parameter error in DiscontinuosGalerkinSpace with "+
            "dimrange=" + str(dimrange) + ": " +\
            "dimrange has to be greater or equal to 1")
    if order < 0:
        raise KeyError(\
            "Parameter error in DiscontinuousGalerkinSpace with "+
            "order=" + str(order) + ": " +\
            "order has to be greater or equal to 0")
    if field == "complex":
        field = "std::complex<double>"

    try:
        view = viewagglomerate[0]
        agglomerate = viewagglomerate[1]
    except:
        agglomerate = viewagglomerate.agglomerate
        view = viewagglomerate.grid

    includes = [ "dune/vem/space/agglomeration.hh" ] + view._includes
    dimw = view.dimWorld
    viewType = view._typeName

    gridPartName = "Dune::FemPy::GridPart< " + view._typeName + " >"
    typeName = "Dune::Vem::AgglomerationVEMSpace< " +\
      "Dune::Fem::FunctionSpace< double, " + field + ", " + str(dimw) + ", " + str(dimrange) + " >, " +\
      gridPartName + ", " + str(order) + " >"

    constructor = Constructor(
                   ['pybind11::object gridView',
                    'const pybind11::function agglomerate'],
                   ['auto agglo = new Dune::Vem::Agglomeration<' + gridPartName + '>',
                    '         (Dune::FemPy::gridPart<' + viewType + '>(gridView), [agglomerate](const auto& e) { return agglomerate(e).template cast<unsigned int>(); } ); ',
                    'auto obj = new DuneType( *agglo );',
                    'pybind11::cpp_function remove_agglo( [ agglo ] ( pybind11::handle weakref ) {',
                    '  delete agglo;',
                    '  weakref.dec_ref();',
                    '} );',
                    '// pybind11::handle nurse = pybind11::detail::get_object_handle( &obj, pybind11::detail::get_type_info( typeid( ' + typeName + ' ) ) );',
                    '// assert(nurse);',
                    'pybind11::weakref( agglomerate, remove_agglo ).release();',
                    'return obj;'],
                   ['"gridView"_a', '"agglomerate"_a',
                    'pybind11::keep_alive< 1, 2 >()'] )

    spc = module(field, includes, typeName, constructor, storage=storage).Space(view, agglomerate)
    addStorage(spc, storage)
    return spc.as_ufl()

def vemScheme(model, space, solver=None, parameters={}):
    """create a scheme for solving second order pdes with the virtual element method

    Args:

    Returns:
        Scheme: the constructed scheme
    """
    """create a scheme for solving second order pdes with continuous finite element

    Args:

    Returns:
        Scheme: the constructed scheme
    """
    # from dune.fem.space import module
    from dune.fem.scheme import module
    from dune.fem.scheme import femschemeModule
    # from . import module
    includes = [ "dune/vem/operator/vemelliptic.hh" ]

    op = lambda linOp,model: "DifferentiableVEMEllipticOperator< " +\
                             ",".join([linOp,model]) + ">"

    if model.hasDirichletBoundary:
        includes += [ "dune/fem/schemes/dirichletwrapper.hh",
                      "dune/vem/operator/vemdirichletconstraints.hh"]
        constraints = lambda model: "Dune::VemDirichletConstraints< " +\
                ",".join([model,space._typeName]) + " > "
        operator = lambda linOp,model: "DirichletWrapperOperator< " +\
                ",".join([op(linOp,model),constraints(model)]) + " >"
    else:
        operator = op

    return femschemeModule(space, model,includes,solver,operator,parameters=parameters)

class CartesianAgglomerate:
    def __init__(self,N,constructor):
        self.N = N
        self.suffix = "cartesian"+str(N)
        self.grid = dune.create.grid("yasp", constructor=constructor)
        self.division = constructor.division
        self.ind = set()
    def __call__(self,en):
        idx = self.grid.indexSet.index(en)
        nx = int(idx / self.division[1])
        ny = int(idx % self.division[1])
        nx = int(nx/self.division[0]*self.N[0])
        ny = int(ny/self.division[1]*self.N[1])
        index = nx*self.N[1]+ny
        self.ind.add(index)
        return index
    def check(self):
        return len(self.ind)==self.N[0]*self.N[1]
class TrivialAgglomerate:
    def __init__(self,constructor):
        self.grid = dune.create.grid("ALUConform", constructor=constructor)
        self.suffix = "simple"+str(self.grid.size(0))
    def __call__(self,en):
        return self.grid.indexSet.index(en)
    def check(self):
        return True
# http://zderadicka.eu/voronoi-diagrams/
from dune.vem.voronoi import triangulated_voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import numpy
class VoronoiAgglomerate:
    def __init__(self,N,constructor):
        self.N = N
        self.suffix = "voronoi"+str(self.N)
        lowerleft  = numpy.array(constructor.lower)
        upperright = numpy.array(constructor.upper)
        numpy.random.seed(1234)
        self.voronoi_points = numpy.random.rand(self.N, 2)
        self.voronoi_points = numpy.array(
                [ p*(upperright-lowerleft) + lowerleft
                    for p in self.voronoi_points ])
        self.voronoi_kdtree = cKDTree(self.voronoi_points)
        vor = Voronoi(self.voronoi_points)
        points, triangles = triangulated_voronoi(constructor, self.voronoi_points)
        self.grid = dune.create.grid("ALUSimplex", {'vertices':points, 'simplices':triangles}, dimgrid=2)
        self.ind = set()
    def __call__(self,en):
        p = en.geometry.center
        test_point_dist, test_point_regions = self.voronoi_kdtree.query([p], k=1)
        index = test_point_regions[0]
        self.ind.add(index)
        return index
    def check(self):
        return len(self.ind)==self.N

from sortedcontainers import SortedDict
import triangle
import matplotlib.pyplot as plt
class PolyAgglomerate:
    def __init__(self,constructor):
        self.suffix = "poly"+str(len(constructor["polygons"]))
        self.domain, self.index = self.construct(constructor["vertices"],
                                                 constructor["polygons"])
        self.grid = dune.create.grid("ALUSimplex", self.domain)
        self.ind = set()
    def __call__(self,en):
        bary = en.geometry.center
        return self.index[self.roundBary(bary)]
    def check(self):
        return len(self.ind)==self.N
    def roundBary(self,a):
        return tuple(round(aa,5) for aa in a)
    def construct(self,vertices,polygons):
        vertices = numpy.array(vertices)
        tr = []
        index = SortedDict()
        for nr, p in enumerate(polygons):
            N = len(p)
            e = [ [p[i],p[(i+1)%N]] for i in range(N) ]
            domain = { "vertices":vertices,
                       "segments":numpy.array(e) }
            tr += [triangle.triangulate(domain,opts="p")]
            bary = [ (vertices[p0]+vertices[p1]+vertices[p2])/3.
                     for p0,p1,p2 in tr[-1]["triangles"] ]
            for b in bary:
                index[self.roundBary(b)] = nr
            # triangle.plot.plot(plt.axes(), **tr[-1])
        # plt.show()
        domain = {"vertices":  numpy.array(vertices),
                  "simplices": numpy.vstack([ t["triangles"] for t in tr ])}
        return domain, index

def polyGrid(constructor,N=None):
    class PolyGrid:
        def __init__(self,constructor,N):
            if isinstance(N,int):
                self.agglomerate = VoronoiAgglomerate(N,constructor)
            elif N is None:
                if isinstance(constructor,dict) and \
                   constructor.get("polygons",None) is not None:
                    self.agglomerate = PolyAgglomerate(constructor)
                else:
                    self.agglomerate = TrivialAgglomerate(constructor)
            else:
                self.agglomerate = CartesianAgglomerate(N,constructor)
            self.grid = self.agglomerate.grid
            self._includes = self.grid._includes
    return PolyGrid(constructor,N)
