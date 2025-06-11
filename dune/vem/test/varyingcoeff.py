from dune import create
from dune.grid import cartesianDomain
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import integrate, discreteFunction, gridFunction
from dune.fem import parameter
from dune.fem.operator import linear as linearOperator

import ufl.algorithms
from ufl import *
import dune.ufl

from script import runTest, checkEOC, interpolate
from interpolate import interpolate_fourthorder

dimR = 1

parameters = {"linear.tolerance": 1e-18,
              "linear.preconditioning.method": "jacobi",
              "penalty": 40,  # for the bbdg scheme
              "linear.verbose": False,
              "nonlinear.verbose": True
              }

def varyingcoeff(space, exact):
    laplace = lambda w: div(grad(w))
    H = lambda w: grad(grad(w))
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space)

    laplaceCoeff = 1
    mu           = 1

    kappa = 0 # 1./(1+dot(x,x))
    beta  = 0 # exp(-x[0]*x[1])
    gamma = 1 # sin(dot(x,x))**2

    a = ( kappa*inner(H(u[0]),H(v[0])) +\
          laplaceCoeff*beta*inner(grad(u),grad(v)) +\
          mu*gamma*inner(u,v)
        ) * dx

    # right hand side and the boundary conditions
    if kappa == 0:
        q = 0
    else:
        q = sum([ H(kappa*H(exact[0])[i,j])[i,j] for i in range(2) for j in range(2) ])
    b = ( q -\
          laplaceCoeff*div(beta*grad(exact[0])) +\
          mu*gamma*exact[0] ) * v[0] * dx
    dbc = [] # dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

    biLaplaceCoeff = kappa
    diffCoeff      = laplaceCoeff*beta
    massCoeff      = mu*gamma

    if kappa == 0:
        extraArgs = {"boundary": "value"}
    else:
        extraArgs = {}
    scheme = dune.vem.vemScheme(inner(u-exact,v)*dx==0,
                            # [a==b, *dbc], space,
                            solver=("suitesparse","umfpack"), # "cg",
                            # **extraArgs,
                            hessStabilization=None, # biLaplaceCoeff,
                            gradStabilization=None, # diffCoeff,
                            massStabilization=1, # massCoeff,
                            parameters=parameters)

    df = discreteFunction(space, name="solution")
    df.clear()
    # df.interpolate(exact)
    info = scheme.solve(target=df)
    # gridFunction(grad(df[0]))[0].plot()
    # gridFunction(grad(df[0]))[1].plot()
    # df.plot()

    edf = exact-df
    err = [inner(edf,edf),
            inner(grad(edf),grad(edf)),
            inner(grad(grad(edf)),grad(grad(edf)))]

    return err

def runTestVaryingcoeff(testSpaces, order):
      x = SpatialCoordinate(triangle)
      # exact  = as_vector( [x[0]**3*x[1]**2] )
      # exact = as_vector( dimR*[cos(pi*x[0])**2*sin(pi*x[1])**2] )
      exact = as_vector( dimR*[2*ufl.sin(x[0])*ufl.sin(x[1]) ] )

      spaceConstructor = lambda grid, r:(
                    dune.vem.vemSpace( grid,
                                       computeField=256,
                                       basisField="long double",
                                       order=order,
                                       dimRange=r,
                                       testSpaces=testSpaces )
               )

      if (order==2):
        if len(testSpaces[0]) == 2: # conforming space seems to have worse L^2 eoc?
            expected_eoc = [order+1, order, order-1]
        else:
            expected_eoc = [order, order, order-1]
      else:
        expected_eoc = [order+1, order, order-1]

      if (interpolate()):
            eoc = runTest(exact, spaceConstructor, interpolate_fourthorder, N0=13)
      else:
            eoc = runTest(exact, spaceConstructor, varyingcoeff,N0=13)

      return eoc, expected_eoc

def main():
      ret = 0
      orders = [3,4]
      for order in orders:
            """
            print("order: ", order)
            C1NCtestSpaces = [ [0], [order-3,order-2], [order-4] ]
            print("C1 non conforming test spaces: ", C1NCtestSpaces)
            eoc, expected_eoc = runTestVaryingcoeff( C1NCtestSpaces, order )
            ret += checkEOC(eoc, expected_eoc)
            """

            print("order: ", order)
            C1ConftestSpaces = [ [0,0], [order-4,order-3], [order-4] ]
            print("C1 conforming test spaces: ", C1ConftestSpaces)
            eoc, expected_eoc = runTestVaryingcoeff( C1ConftestSpaces, order )
            ret += checkEOC(eoc, expected_eoc)
            print("->",ret,":",eoc,expected_eoc)
      assert ret==0, "some test went wrong"

main()
