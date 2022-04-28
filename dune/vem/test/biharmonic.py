from dune import create
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction
from dune.fem import parameter
from dune.fem.operator import linear as linearOperator

import ufl.algorithms
from ufl import *
import dune.ufl

parameters = {"newton.linear.tolerance": 1e-12,
              "newton.linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

def biharmonic(space, exact):
    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)

    epsilon      = 1
    laplaceCoeff = 1
    mu           = 1

    a = ( epsilon*inner(H(u[0]),H(v[0])) +\
          laplaceCoeff*inner(grad(u),grad(v)) +\
          mu*inner(u,v)
        ) * dx
    b = ( epsilon*laplace(laplace(exact[0])) -\
          laplaceCoeff*laplace(exact[0]) +\
          mu*exact[0] ) * v[0] * dx
    dbc = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]

    biLaplaceCoeff = epsilon
    diffCoeff      = laplaceCoeff
    massCoeff      = mu

    scheme = dune.vem.vemScheme([a==b, *dbc], space,
                            solver="cg",
                            hessStabilization=biLaplaceCoeff,
                            gradStabilization=diffCoeff,
                            massStabilization=massCoeff,
                            parameters=parameters)

    df = discreteFunction(space, name="solution")
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
            inner(grad(edf),grad(edf)),
            inner(grad(grad(edf)),grad(grad(edf)))]

    return err