from ufl import *
import dune.ufl

from scipy.sparse.linalg import spsolve

from script import runTest, checkEOC, interpolate, getParameters

parameters = {"newton.linear.tolerance": 1e-8,
              "newton.tolerance": 5e-6,
              "newton.lineSearch": "simple",
              "newton.linear.verbose": False,
              "newton.verbose": True
              }

mass = dune.ufl.Constant(1, "mu")

def curlfree(space, exact):
    u = TrialFunction(space)
    v = TestFunction(space)

    a = (div(u)*div(v) + dot(u,v) ) * dx
    b = dot( -grad(div(exact)) + exact, v) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

    df = space.interpolate(exact,name="solution")
    scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space,
                  solver="cg",
                #   solver=("suitesparse","umfpack"),
                  parameters=parameters,
                  gradStabilization=0,
                  massStabilization=mass)
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
           inner(div(edf),div(edf))]

    return err

def runTestCurlfree(order):
    x = SpatialCoordinate(triangle)
    ln, lm, Lx, Ly = getParameters()
    # print(ln, lm, Lx, Ly)
    exact = -grad( (cos(ln/Lx*pi*x[0])*cos(lm/Ly*pi*x[1])) )
    spaceConstructor = lambda grid, r: dune.vem.curlFreeSpace( grid,
                                                               order=order )

    expected_eoc = [order+1,order+1]
    if (interpolate()):
      eoc = runTest(exact, spaceConstructor, interpolate_secondorder)
    else:
      eoc = runTest(exact, spaceConstructor, curlfree)

    return eoc, expected_eoc

def main():
    orders = [1,3]
    for order in orders:
        print("order: ", order)

        eoc, expected_eoc = runTestCurlfree( order )
        checkEOC(eoc, expected_eoc)

main()
