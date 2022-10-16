from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.fem.operator import linear as linearOperator

import ufl.algorithms
from ufl import *
import dune.ufl

def interpolate_secondorder(space, exact):
    df = space.interpolate(exact,name="solution")

    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]

    return err

def interpolate_fourthorder(space, exact):
    df = space.interpolate(exact,name="solution")

    edf = exact-df
    err = [inner(edf,edf),
            inner(grad(edf),grad(edf)),
            inner(grad(grad(edf)),grad(grad(edf)))]

    return err