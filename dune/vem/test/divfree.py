from ufl import *
import dune.ufl

from script import runTest, checkEOC, interpolate
from interpolate import interpolate_secondorder

parameters = {"newton.linear.tolerance": 1e-8,
              "newton.tolerance": 5e-6,
              "newton.lineSearch": "simple",
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

mass = dune.ufl.Constant(1, "mu")

def divfree(space, exact):
    u = TrialFunction(space)
    v = TestFunction(space)

    a = (inner(grad(u),grad(v)) + mass*dot(u,v) ) * dx

    b = dot( -div(grad(exact))  + mass*exact, v) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

    df = space.interpolate(exact,name="solution")
    scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space,
                #   solver=("suitesparse","umfpack"),
                  parameters=parameters,
                  gradStabilization=1,
                  massStabilization=mass)
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]

    return err

def runTestDivFree(order):
    x = SpatialCoordinate(triangle)

    p = (order+1)//2
    exact = as_vector([-x[1]+x[0]**(p+1)*x[1]**p,
                        x[0]-x[1]**(p+1)*x[0]**p])

    spaceConstructor = lambda grid, r: dune.vem.divFreeSpace( grid,
                                                              order=order )
    expected_eoc = [order+1]
    if (interpolate()):
      eoc = runTest(exact, spaceConstructor, interpolate_secondorder)
    else:
      eoc = runTest(exact, spaceConstructor, divfree)

    return eoc, expected_eoc

def main():
    orders = [2,4]
    for order in orders:
        print("order: ", order)

        eoc, expected_eoc = runTestDivFree( order )
        checkEOC(eoc, expected_eoc)

main()
