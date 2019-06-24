from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
logger = logging.getLogger(__name__)

from ufl.equation import Equation
from ufl import Form
from dune.generator import Constructor, Method
import dune.common.checkconfiguration as checkconfiguration
import dune

def bbdgSpace(view, order=1, dimRange=1, field="double", storage="adaptive"):
    """create a discontinous galerkin space over an agglomerated grid

    Args:
        view: the underlying grid view
        order: polynomial order of the finite element functions
        dimRange: dimension of the range space
        field: field of the range space
        storage: underlying linear algebra backend

    Returns:
        Space: the constructed Space
    """

    from dune.fem.space import module, addStorage
    if dimRange < 1:
        raise KeyError(\
            "Parameter error in DiscontinuosGalerkinSpace with "+
            "dimRange=" + str(dimRange) + ": " +\
            "dimRange has to be greater or equal to 1")
    if order < 0:
        raise KeyError(\
            "Parameter error in DiscontinuousGalerkinSpace with "+
            "order=" + str(order) + ": " +\
            "order has to be greater or equal to 0")
    if field == "complex":
        field = "std::complex<double>"

    agglomerate = view.hierarchicalGrid.agglomerate

    includes = [ "dune/vem/agglomeration/dgspace.hh" ] + view._includes
    dimw = view.dimWorld
    viewType = view._typeName

    gridPartName = "Dune::FemPy::GridPart< " + view._typeName + " >"
    typeName = "Dune::Vem::AgglomerationDGSpace< " +\
      "Dune::Fem::FunctionSpace< double, " + field + ", " + str(dimw) + ", " + str(dimRange) + " >, " +\
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

    spc = module(field, includes, typeName, constructor, storage=storage,ctorArgs=[view, agglomerate])
    addStorage(spc, storage)
    return spc.as_ufl()

from dune.fem.scheme import dg
def bbdgScheme(model, space, penalty=0, solver=None, parameters={}):
    spaceType = space._typeName
    penaltyClass = "Dune::Vem::BBDGPenalty<"+spaceType+">"
    return dg(model,space,penalty,solver,parameters,penaltyClass)

def vemSpace(view, order=1, dimRange=1, conforming=True, field="double", storage="adaptive"):
    """create a virtual element space over an agglomerated grid

    Args:
        view: the underlying grid view
        order: polynomial order of the finite element functions
        dimRrange: dimension of the range space
        field: field of the range space
        storage: underlying linear algebra backend

    Returns:
        Space: the constructed Space
    """

    from dune.fem.space import module, addStorage
    if dimRange < 1:
        raise KeyError(\
            "Parameter error in DiscontinuosGalerkinSpace with "+
            "dimRange=" + str(dimRange) + ": " +\
            "dimRange has to be greater or equal to 1")
    if order < 0:
        raise KeyError(\
            "Parameter error in DiscontinuousGalerkinSpace with "+
            "order=" + str(order) + ": " +\
            "order has to be greater or equal to 0")
    if field == "complex":
        field = "std::complex<double>"

    agglomerate = view.hierarchicalGrid.agglomerate

    includes = [ "dune/vem/space/agglomeration.hh" ] + view._includes
    dimw = view.dimWorld
    viewType = view._typeName

    gridPartName = "Dune::FemPy::GridPart< " + view._typeName + " >"
    typeName = "Dune::Vem::AgglomerationVEMSpace< " +\
      "Dune::Fem::FunctionSpace< double, " + field + ", " + str(dimw) + ", " + str(dimRange) + " >, " +\
      gridPartName + ", " + str(order) + " >"

    constructor = Constructor(
                   ['pybind11::object gridView',
                    'const pybind11::function agglomerate',
                    'bool conforming'],
                   ['auto agglo = new Dune::Vem::Agglomeration<' + gridPartName + '>',
                    '         (Dune::FemPy::gridPart<' + viewType + '>(gridView), [agglomerate](const auto& e) { return agglomerate(e).template cast<unsigned int>(); } ); ',
                    'auto obj = new DuneType( *agglo, conforming );',
                    'pybind11::cpp_function remove_agglo( [ agglo ] ( pybind11::handle weakref ) {',
                    '  delete agglo;',
                    '  weakref.dec_ref();',
                    '} );',
                    '// pybind11::handle nurse = pybind11::detail::get_object_handle( &obj, pybind11::detail::get_type_info( typeid( ' + typeName + ' ) ) );',
                    '// assert(nurse);',
                    'pybind11::weakref( agglomerate, remove_agglo ).release();',
                    'return obj;'],
                   ['"gridView"_a', '"agglomerate"_a', '"conforming"_a',
                    'pybind11::keep_alive< 1, 2 >()'] )

    spc = module(field, includes, typeName, constructor, storage=storage, ctorArgs=[view, agglomerate, conforming])
    addStorage(spc, storage)
    return spc.as_ufl()

def vemScheme(model, space=None, solver=None, parameters={}):
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

    if isinstance(model, (list, tuple)):
        modelParam = model[1:]
        model = model[0]
    if isinstance(model,Equation):
        if space == None:
            try:
                space = model.lhs.arguments()[0].ufl_function_space()
            except AttributeError:
                raise ValueError("no space provided and could not deduce from form provided")
        from dune.fem.model._models import elliptic
        if modelParam:
            model = elliptic(space.grid,model,*modelParam)
        else:
            model = elliptic(space.grid,model)

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

def vemOperator(model, domainSpace=None, rangeSpace=None):
    from dune.fem.operator import load
    if rangeSpace is None:
        rangeSpace = domainSpace

    modelParam = None
    if isinstance(model, (list, tuple)):
        modelParam = model[1:]
        model = model[0]
    if isinstance(model,Form):
        model = model == 0
    if isinstance(model,Equation):
        from dune.fem.model._models import elliptic as makeElliptic
        if rangeSpace == None:
            try:
                rangeSpace = model.lhs.arguments()[0].ufl_function_space()
            except AttributeError:
                raise ValueError("no range space provided and could not deduce from form provided")
        if domainSpace == None:
            try:
                domainSpace = model.lhs.arguments()[1].ufl_function_space()
            except AttributeError:
                raise ValueError("no domain space provided and could not deduce from form provided")
        if modelParam:
            model = makeElliptic(domainSpace.grid,model,*modelParam)
        else:
            model = makeElliptic(domainSpace.grid,model)

    if not hasattr(rangeSpace,"interpolate"):
        raise ValueError("wrong range space")
    if not hasattr(domainSpace,"interpolate"):
        raise ValueError("wrong domain space")

    domainSpaceType = domainSpace._typeName
    rangeSpaceType = rangeSpace._typeName

    storage,  domainFunctionIncludes, domainFunctionType, _, _, dbackend = domainSpace.storage
    rstorage, rangeFunctionIncludes,  rangeFunctionType,  _, _, rbackend = rangeSpace.storage
    if not rstorage == storage:
        raise ValueError("storage for both spaces must be identical to construct operator")

    includes = ["dune/vem/operator/elliptic.hh",
                "dune/fempy/py/grid/gridpart.hh",
                "dune/fem/schemes/dirichletwrapper.hh",
                "dune/vem/operator/vemdirichletconstraints.hh"]
    includes += domainSpace._includes + domainFunctionIncludes
    includes += rangeSpace._includes + rangeFunctionIncludes
    includes += ["dune/fem/schemes/diffusionmodel.hh", "dune/fempy/parameter.hh"]

    import dune.create as create
    linearOperator = create.discretefunction(storage)(domainSpace,rangeSpace)[3]

    modelType = "DiffusionModel< " +\
          "typename " + domainSpaceType + "::GridPartType, " +\
          domainSpaceType + "::dimRange, " +\
          rangeSpaceType + "::dimRange, " +\
          "typename " + domainSpaceType + "::RangeFieldType >"
    typeName = "Dune::Vem::DifferentiableEllipticOperator< " + linearOperator + ", " + modelType + ">"
    if model.hasDirichletBoundary:
        constraints = "Dune::VemDirichletConstraints< " +\
                ",".join([modelType,domainSpace._typeName]) + " > "
        typeName = "DirichletWrapperOperator< " +\
                ",".join([typeName,constraints(model)]) + " >"

    constructor = Constructor(['const '+domainSpaceType+'& dSpace','const '+rangeSpaceType+' &rSpace', modelType + ' &model'],
                              ['return new ' + typeName + '( dSpace, rSpace, model );'],
                              ['pybind11::keep_alive< 1, 2 >()', 'pybind11::keep_alive< 1, 3 >()', 'pybind11::keep_alive< 1, 4 >()'])

    scheme = load(includes, typeName, constructor).Operator(domainSpace,rangeSpace, model)
    scheme.model = model
    return scheme

#################################################################

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

#################################################################

def aluGrid(constructor, dimgrid=None, dimworld=None, elementType=None, **parameters):
    if not dimgrid:
        dimgrid = getDimgrid(constructor)
    if dimworld is None:
        dimworld = dimgrid
    if elementType is None:
        elementType = parameters.pop("type")
    refinement = parameters["refinement"]
    if refinement == "conforming":
        refinement="Dune::conforming"
    elif refinement == "nonconforming":
        refinement="Dune::nonconforming"
    if not (2 <= dimworld and dimworld <= 3):
        raise KeyError("Parameter error in ALUGrid with dimworld=" + str(dimworld) + ": dimworld has to be either 2 or 3")
    if not (2 <= dimgrid and dimgrid <= dimworld):
        raise KeyError("Parameter error in ALUGrid with dimgrid=" + str(dimgrid) + ": dimgrid has to be either 2 or 3")
    if refinement=="Dune::conforming" and elementType=="Dune::cube":
        raise KeyError("Parameter error in ALUGrid with refinement=" + refinement + " and type=" + elementType + ": conforming refinement is only available with simplex element type")
    typeName = "Dune::ALUGrid< " + str(dimgrid) + ", " + str(dimworld) + ", " + elementType + ", " + refinement + " >"
    includes = ["dune/alugrid/grid.hh", "dune/alugrid/dgf.hh"]
    gridModule = module(includes, typeName, dynamicAttr=True)
    return gridModule.LeafGrid(gridModule.reader(constructor))
def aluSimplexGrid(constructor, dimgrid=None, dimworld=None):
    from dune.grid.grid_generator import module, getDimgrid
    typeName = "Dune::Vem::Grid"
    includes = ["dune/vem/misc/grid.hh"]
    gridModule = module(includes, typeName, dynamicAttr=True)
    return gridModule.LeafGrid(gridModule.reader(constructor))

#################################################################

class TrivialAgglomerate:
    def __init__(self,constructor):
        self.grid = aluSimplexGrid(constructor)
        self.suffix = "simple"+str(self.grid.size(0))
    def __call__(self,en):
        return self.grid.indexSet.index(en)
    def check(self):
        return True
# http://zderadicka.eu/voronoi-diagrams/
from dune.vem.voronoi import triangulated_voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree, Delaunay
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
        domain = {'vertices':points, 'simplices':triangles}
        self.grid = aluSimplexGrid(self.domain, dimgrid=2)
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
        self.grid = aluSimplexGrid(self.domain)
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
            if False: # use triangle
                e = [ [p[i],p[(i+1)%N]] for i in range(N) ]
                domain = { "vertices":vertices,
                           "segments":numpy.array(e) }
                tr += [triangle.triangulate(domain,opts="p")]
            else: # use scipy
                poly = numpy.append(p,[p[0]])
                vert = vertices[p, :]
                tri = Delaunay(vert).simplices
                tr += [{"triangles":p[tri]}]
            bary = [ (vertices[p0]+vertices[p1]+vertices[p2])/3.
                     for p0,p1,p2 in tr[-1]["triangles"] ]
            for b in bary:
                index[self.roundBary(b)] = nr

        domain = {"vertices":  numpy.array(vertices),
                  "simplices": numpy.vstack([ t["triangles"] for t in tr ])}
        return domain, index

def polyGrid(constructor,N=None):
    if isinstance(N,int):
        agglomerate = VoronoiAgglomerate(N,constructor)
    elif N is None:
        if isinstance(constructor,dict) and \
           constructor.get("polygons",None) is not None:
            agglomerate = PolyAgglomerate(constructor)
        else:
            agglomerate = TrivialAgglomerate(constructor)
    else:
        agglomerate = CartesianAgglomerate(N,constructor)
    grid = agglomerate.grid
    grid.hierarchicalGrid.agglomerate = agglomerate
    return grid
