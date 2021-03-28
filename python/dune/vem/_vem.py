from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
logger = logging.getLogger(__name__)

from ufl.equation import Equation
from ufl import Form
from dune.generator import Constructor, Method
import dune.common.checkconfiguration as checkconfiguration
import dune

def bbdgSpace(view, order=1, scalar=False, dimRange=None, field="double", storage="adaptive"):
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
    if dimRange is None:
        dimRange = 1
        scalar = True

    if dimRange > 1 and scalar:
        raise KeyError(\
                "trying to set up a scalar space with dimRange = " +\
                str(dimRange) + ">1")
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

    spc = module(field, includes, typeName, constructor, scalar=scalar, storage=storage,ctorArgs=[view, agglomerate])
    addStorage(spc, storage)
    return spc.as_ufl()

from dune.fem.scheme import dg,galerkin
def bbdgScheme(model, space=None, penalty=1, solver=None, parameters={}):
    if space == None:
        try:
            space = model.lhs.arguments()[0].ufl_function_space()
        except AttributeError:
            raise ValueError("no space provided and could not deduce from form provided")
    spaceType = space._typeName
    penaltyClass = "Dune::Vem::BBDGPenalty<"+spaceType+">"
    return dg(model,space,penalty,solver,parameters,penaltyClass)
    # return galerkin(model,space,solver,parameters)

def vemSpace(view, order=1, testSpaces=None, scalar=False,
             dimRange=None, conforming=True, field="double",
             storage="adaptive", basisChoice=2,
             edgeInterpolation=False):
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
    if dimRange is None:
        dimRange = 1
        scalar = True

    if dimRange > 1 and scalar:
        raise KeyError(\
                "trying to set up a scalar space with dimRange = " +\
                str(dimRange) + ">1")
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

    if testSpaces is None:
        if conforming:
            testSpaces = [  [0], [order-2], [order-2] ]
        else:
            testSpaces = [ [-1], [order-1], [order-2] ]
    if testSpaces == "serendipity":
        etaMin = view.hierarchicalGrid.agglomerate.minEdgeNumber
        testSpaces = [  [0], [order-2], [max(-1,order-etaMin)] ]
        print("serendipity:",testSpaces,etaMin)

    for i,ts in enumerate(testSpaces):
        if type(ts) == int:
            testSpaces[i] = [ts]

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
                    'const std::array<std::vector<int>,'+str(view.dimension+1)+'> &testSpaces',
                    'int basisChoice','bool edgeInterpolation'],
                   ['auto agglo = new Dune::Vem::Agglomeration<' + gridPartName + '>',
                    '         (Dune::FemPy::gridPart<' + viewType + '>(gridView), [agglomerate](const auto& e) { return agglomerate(e).template cast<unsigned int>(); } ); ',
                    'auto obj = new DuneType( *agglo, testSpaces, basisChoice, edgeInterpolation );',
                    'pybind11::cpp_function remove_agglo( [ agglo ] ( pybind11::handle weakref ) {',
                    '  delete agglo;',
                    '  weakref.dec_ref();',
                    '} );',
                    '// pybind11::handle nurse = pybind11::detail::get_object_handle( &obj, pybind11::detail::get_type_info( typeid( ' + typeName + ' ) ) );',
                    '// assert(nurse);',
                    'pybind11::weakref( agglomerate, remove_agglo ).release();',
                    'return obj;'],
                   ['"gridView"_a', '"agglomerate"_a', '"testSpaces"_a',
                    '"basisChoice"_a', '"edgeInterpolation"_a',
                    'pybind11::keep_alive< 1, 2 >()'] )
    diameterMethod = Method('diameters',
       '''[]( DuneType &self ) { return self.blockMapper().indexSet().diameters(); }''' )

    spc = module(field, includes, typeName, constructor, diameterMethod,
                scalar=scalar, storage=storage,
                ctorArgs=[view, agglomerate, testSpaces, basisChoice, edgeInterpolation])
    addStorage(spc, storage)
    return spc.as_ufl()

def space(view, order=1, dimRange=None,
          testSpaces=None,
          conforming=True,
          version=None,
          scalar=False,
          basisChoice=2,
          field="double", storage="adaptive"):
    '''
        version is tuple,list for a general vem space (=tests[aces)
        version == "dg" (bbdg)
        version == "dgSimplex" (dg on triangulation)
        version == "continuous" (vem)
        version == "continuousSimplex" (lagrange)
        version == "non-conformin" (vem)
    '''
    if version is None and testSpaces is None:
        raise AttributeError("version and testSpaces parameters can not both be None (default)")
    if version is None: version = testSpaces
    if isinstance(version,tuple) or isinstance(version,list):
        return vemSpace(view,order=order,scalar=scalar,dimRange=dimRange,field=field,storage=storage,
                        testSpaces=version, basisChoice=basisChoice)
    elif version == "dg":
        return bbdgSpace(view,order=order,scalar=scalar,dimRange=dimRange,field=field,storage=storage)
    elif version == "dgSimplex":
        from dune.fem.space import onb
        return onb(view,order=order,scalar=scalar,dimRange=dimRange,field=field,storage=storage)
    elif version == "continuous":
        return vemSpace(view,order=order,scalar=scalar,dimRange=dimRange,field=field,storage=storage,
                        conforming=True, basisChoice=basisChoice)
    elif version == "continuousSimplex":
        from dune.fem.space import lagrange
        return lagrange(view,order=order,scalar=scalar,dimRange=dimRange,field=field,storage=storage)
    elif version == "non-conforming":
        return vemSpace(view,order=order,scalar=scalar,dimRange=dimRange,field=field,storage=storage,
                        conforming=False, basisChoice=basisChoice)
    raise AttributeError("wrong version string provided:",version)

#########################################################

# from dune.fem.model import elliptic
from dune.fem.model import integrands
from dune.vem.patch import transform

def vemModel(view, equation, space,
        hessStabilization=None,gradStabilization=None, massStabilization=None,
        *args, **kwargs):
    return integrands(view, equation,
                      modelPatch=transform(space,hessStabilization,gradStabilization,massStabilization),
                      includes=["dune/vem/py/integrands.hh"],
                      *args, **kwargs)

def vemScheme(model, space=None, solver=None, parameters={},
              hessStabilization=None, gradStabilization=None, massStabilization=None,
              boundary="full"):
    """create a scheme for solving second order pdes with the virtual element method

    Args:

    Returns:
        Scheme: the constructed scheme
    """

    from dune.fem.scheme import module
    from dune.fem.scheme import femschemeModule

    assert boundary=="full" or boundary=="value" or boundary=="derivative" or boundary is None

    modelParam = None
    if isinstance(model, (list, tuple)):
        modelParam = model[1:]
        model = model[0]
    if isinstance(model,Equation):
        if space == None:
            try:
                space = model.lhs.arguments()[0].ufl_function_space()
            except AttributeError:
                raise ValueError("no space provided and could not deduce from form provided")
        else:
            try:
                eqSpace = model.lhs.arguments()[0].ufl_function_space()
                if not eqSpace.dimRange == space.dimRange:
                    raise ValueError("""range of space used for arguments in equation
                    and the range of space passed to the scheme have to match -
                    space argument to scheme can be 'None'""")
            except AttributeError:
                pass
        if modelParam:
            model = vemModel(space.grid,model,space,hessStabilization,gradStabilization,massStabilization,*modelParam)
        else:
            model = vemModel(space.grid,model,space,hessStabilization,gradStabilization,massStabilization)

    includes = [ "dune/vem/operator/vemelliptic.hh", "dune/vem/operator/diffusionmodel.hh" ]

    op = lambda linOp,model: "DifferentiableVEMEllipticOperator< " +\
                             ",".join([linOp,model]) + ">"

    if model.hasDirichletBoundary:
        assert boundary is not None
        includes += [ "dune/fem/schemes/dirichletwrapper.hh",
                      "dune/vem/operator/vemdirichletconstraints.hh"]
        if boundary   == "full":       boundary = 3
        elif boundary == "value":      boundary = 1
        elif boundary == "derivative": boundary = 2
        constraints = lambda model: "Dune::VemDirichletConstraints< " +\
                    ",".join([model,space._typeName,str(boundary)]) + " > "
        operator = lambda linOp,model: "DirichletWrapperOperator< " +\
                ",".join([op(linOp,model),constraints(model)]) + " >"
    else:
        assert boundary is None
        operator = op

    spaceType = space._typeName
    # modelType = "VEMDiffusionModel< " +\
    #       "typename " + spaceType + "::GridPartType, " +\
    #       spaceType + "::dimRange, " +\
    #       spaceType + "::dimRange, " +\
    #       "typename " + spaceType + "::RangeFieldType >"

    includes += ["dune/vem/operator/diffusionmodel.hh"]
    valueType = 'std::tuple< typename ' + spaceType + '::RangeType, typename ' + spaceType + '::JacobianRangeType >'
    modelType = 'Dune::Fem::VirtualizedVemIntegrands< typename ' + spaceType + '::GridPartType, ' + model._domainValueType + ", " + model._rangeValueType+ ' >'

    return femschemeModule(space,model,includes,solver,operator,
            parameters=parameters,
            modelType=modelType
            )


def vemOperator(model, domainSpace=None, rangeSpace=None):
    # from dune.fem.model._models import elliptic as makeElliptic
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
            model = vemModel(domainSpace.grid,model,domainSpace,None,None,*modelParam)
        else:
            model = vemModel(domainSpace.grid,model,domainSpace,None,None)

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

    includes = ["dune/vem/operator/vemelliptic.hh",
                "dune/fempy/py/grid/gridpart.hh",
                "dune/fem/schemes/dirichletwrapper.hh",
                "dune/vem/operator/vemdirichletconstraints.hh"]
    includes += domainSpace._includes + domainFunctionIncludes
    includes += rangeSpace._includes + rangeFunctionIncludes
    includes += ["dune/vem/operator/diffusionmodel.hh", "dune/fempy/parameter.hh"]

    import dune.create as create
    linearOperator = create.discretefunction(storage)(domainSpace,rangeSpace)[3]

    modelType = "VEMDiffusionModel< " +\
          "typename " + domainSpaceType + "::GridPartType, " +\
          domainSpaceType + "::dimRange, " +\
          rangeSpaceType + "::dimRange, " +\
          "typename " + domainSpaceType + "::RangeFieldType >"
    typeName = "DifferentiableVEMEllipticOperator< " + linearOperator + ", " + modelType + ">"
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
def aluSimplexGrid(constructor, dimgrid=2, dimworld=2):
    from dune.grid.grid_generator import module, getDimgrid
    typeName = "Dune::Vem::Grid<"+str(dimgrid)+","+str(dimworld)+">"
    includes = ["dune/vem/misc/grid.hh"]
    gridModule = module(includes, typeName, dynamicAttr=True)
    return gridModule.LeafGrid(gridModule.reader(constructor))
def aluCubeGrid(constructor, dimgrid=2, dimworld=2):
    from dune.grid.grid_generator import module, getDimgrid
    typeName = "Dune::Vem::CubeGrid<"+str(dimgrid)+","+str(dimworld)+">"
    includes = ["dune/vem/misc/grid.hh"]
    gridModule = module(includes, typeName, dynamicAttr=True)
    return gridModule.LeafGrid(gridModule.reader(constructor))
def yaspGrid(constructor, dimgrid=None, coordinates="equidistant", ctype="double"):
    from dune.generator import Constructor
    from dune.grid.grid_generator import module, getDimgrid
    from dune.typeregistry import generateTypeName

    if not dimgrid:
        dimgrid = getDimgrid(constructor)
    if coordinates == "equidistant":
        coordinates = "Dune::EquidistantOffsetCoordinates"
    elif coordinates == "tensorproduct":
        coordinates = "Dune::TensorProductCoordinates"

    if not dimgrid:
        dimgrid = getDimgrid(constructor)
    coordinates, _ = generateTypeName(coordinates, ctype, dimgrid)
    typeName, includes = generateTypeName("Dune::Vem::YGrid", dimgrid, coordinates)
    includes += ["dune/vem/misc/grid.hh", "dune/grid/io/file/dgfparser/dgfyasp.hh"]

    ctor = Constructor([
              'Dune::FieldVector<'+ctype+', '+str(dimgrid)+'> lowerleft',
              'Dune::FieldVector<'+ctype+', '+str(dimgrid)+'> upperright',
              'std::array<int, '+str(dimgrid)+'> elements',
              'std::array<bool, '+str(dimgrid)+'> periodic',
              'int overlap'],
              ['std::bitset<'+str(dimgrid)+'> periodic_;',
               'for (int i=0;i<'+str(dimgrid)+';++i) periodic_.set(i,periodic[i]);',
               'return new DuneType(lowerleft,upperright,elements,periodic_,overlap);'],
              ['"lowerleft"_a', '"upperright"_a', '"elements"_a', '"periodic"_a', '"overlap"_a'])

    gridModule = module(includes, typeName, ctor, dynamicAttr=True)

    try:
        lowerleft  = constructor.lower
        upperright = constructor.upper
        elements   = constructor.division
        periodic   = constructor.param.get("periodic", [False,]*dimgrid)
        overlap    = constructor.param.get("overlap", 0)
        return gridModule.HierarchicalGrid(lowerleft,upperright,elements,periodic,overlap).leafView
    except AttributeError:
        return gridModule.reader(constructor).leafView

#################################################################
class TrivialAgglomerate:
    def __init__(self,constructor, cubes=False, **kwargs):
        if cubes:
            self.grid = aluCubeGrid(constructor, **kwargs)
        else:
            self.grid = aluSimplexGrid(constructor, **kwargs)
        self.suffix = "simple"+str(self.grid.size(0))
        self.size = self.grid.size(0)
    def __call__(self,en):
        return self.grid.indexSet.index(en)
    def check(self):
        return True
# http://zderadicka.eu/voronoi-diagrams/
from dune.vem.voronoi import triangulated_voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree, Delaunay
import numpy

import triangle
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
class PolyAgglomerate:
    def __init__(self,constructor,cubes=False,convex=False):
        self.convex = convex
        self.cubes = cubes
        self.suffix = "poly"+str(len(constructor["polygons"]))
        self.domain, self.index = self.construct(constructor["vertices"],
                                                 constructor["polygons"])
        if cubes:
            self.grid = aluCubeGrid(self.domain)
        else:
            self.grid = aluSimplexGrid(self.domain)
        self.ind = set()
        self.size = len( constructor["polygons"] )
    def __call__(self,en):
        bary = en.geometry.center
        return self.index[self.roundBary(bary)]
    def check(self):
        return len(self.ind)==self.N
    def roundBary(self,a):
        return tuple(round(aa,8) for aa in a)
    def construct(self,vertices,polygons):
        index = SortedDict()
        if self.cubes:
            for i,poly in enumerate(polygons):
                assert len(poly)==4
                bary = [sum([vertices[p][i] for p in poly]) / 4 for i in range(2)]
                index[self.roundBary(bary)] = i
                polygons[i][2], polygons[i][3] = poly[3], poly[2]
            domain = {"vertices": vertices, "cubes": polygons}
        else:
            vertices = numpy.array(vertices)
            tr = []
            for nr, p in enumerate(polygons):
                p = numpy.array(p)
                N = len(p)
                if not self.convex: # use triangle
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

def polyGrid(constructor,cubes=False, convex=False,**kwargs):
    if isinstance(constructor,dict) and \
        constructor.get("polygons",None) is not None:
        agglomerate = PolyAgglomerate(constructor,cubes,convex)
        agglomerate.minEdgeNumber = min([len(p) for p in constructor["polygons"]])
    else:
        agglomerate = TrivialAgglomerate(constructor, cubes, **kwargs)
        if cubes:
            agglomerate.minEdgeNumber = 4
        else:
            agglomerate.minEdgeNumber = 3
    grid = agglomerate.grid
    grid.hierarchicalGrid.agglomerate = agglomerate
    return grid
