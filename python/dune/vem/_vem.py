from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
logger = logging.getLogger(__name__)

import dune.common.checkconfiguration as checkconfiguration

def agglomerateddg(view, agglomerate, order=1, dimrange=1, field="double", storage="adaptive", **unused):
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

    from dune.fem.space import module
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

    includes = [ "dune/vem/agglomeration/dgspace.hh" ] + view._includes
    dimw = view.dimWorld
    viewType = view._typeName

    gridPartName = "Dune::FemPy::GridPart< " + view._typeName + " >"
    typeName = "Dune::Vem::AgglomerationDGSpace< " +\
      "Dune::Fem::FunctionSpace< double, " + field + ", " + str(dimw) + ", " + str(dimrange) + " >, " +\
      gridPartName + ", " + str(order) + " >"

    constructor = ['[] ( ' + typeName + ' &self, pybind11::object gridView, ' +\
                   'const pybind11::function agglomerate) {',
                   '    auto agglo = new Dune::Vem::Agglomeration<' + gridPartName + '>',
                   '         (Dune::FemPy::gridPart<' + viewType + '>(gridView), [agglomerate](const auto& e) { return agglomerate(e).template cast<unsigned int>(); } ); ',
                   '    new (&self) ' + typeName + '( *agglo );',
                   '  }, "gridView"_a, "agglomerate"_a, pybind11::keep_alive< 1, 2 >()']

    return module(field, storage, includes, typeName, [constructor]).Space(view, agglomerate)

def agglomeratedvem(view, agglomerate, order=1, dimrange=1, field="double", storage="adaptive", **unused):
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

    from dune.fem.space import module
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

    includes = [ "dune/vem/space/agglomeration.hh" ] + view._includes
    dimw = view.dimWorld
    viewType = view._typeName

    gridPartName = "Dune::FemPy::GridPart< " + view._typeName + " >"
    typeName = "Dune::Vem::AgglomerationVEMSpace< " +\
      "Dune::Fem::FunctionSpace< double, " + field + ", " + str(dimw) + ", " + str(dimrange) + " >, " +\
      gridPartName + ", " + str(order) + " >"

    constructor = ['[] ( ' + typeName + ' &self, pybind11::object gridView, ' +\
                   'const pybind11::function agglomerate) {',
                   '    auto agglo = new Dune::Vem::Agglomeration<' + gridPartName + '>',
                   '         (Dune::FemPy::gridPart<' + viewType + '>(gridView), [agglomerate](const auto& e) { return agglomerate(e).template cast<unsigned int>(); } ); ',
                   '    new (&self) ' + typeName + '( *agglo );',
                   '  }, "gridView"_a, "agglomerate"_a, pybind11::keep_alive< 1, 2 >()']

    return module(field, storage, includes, typeName, [constructor]).Space(view, agglomerate)
