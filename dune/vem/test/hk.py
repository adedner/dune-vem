from ufl import *
import dune.ufl

from script import runTest, checkEOC, interpolate
from interpolate import interpolate_secondorder

dimR = 1

def hk(space, exact):
    massCoeff = 1
    diffCoeff = 1
    assert(dimR == exact.ufl_shape[0]), "dim range not compatible"
#     dimR = exact.ufl_shape[0]
    u = TrialFunction(space)
    v = TestFunction(space)

    a = (inner(grad(u),grad(v)) + dot(u,v) ) * dx
    b = sum( [ ( -div(diffCoeff*grad(exact[i])) + massCoeff*exact[i] ) * v[i]
             for i in range(dimR) ] ) * dx
    dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]

    df = space.interpolate(dimR*[0],name="solution")
    scheme = dune.vem.vemScheme(
                  [a==b, *dbc], space, solver="cg",
                  gradStabilization=diffCoeff,
                  massStabilization=massCoeff)
    info = scheme.solve(target=df)

    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]

    return err

def runTesthk(testSpaces, order, vectorSpace, reduced):
    x = SpatialCoordinate(triangle)

    exact = as_vector( dimR*[x[0]*x[1] * cos(pi*x[0]*x[1])] )
    spaceConstructor = lambda grid, r: dune.vem.vemSpace( grid,
                                                          order=order,
                                                          dimRange=r,
                                                          testSpaces=testSpaces,
                                                          vectorSpace=vectorSpace,
                                                          reduced=reduced)
    expected_eoc = [order+1,order]
    if (interpolate()):
       eoc = runTest(exact, spaceConstructor, interpolate_secondorder)
    else:
       eoc = runTest(exact, spaceConstructor, hk)

    return eoc, expected_eoc

def main():
    # test elliptic with conforming and nonconforming second order VEM space
    orderslist_secondorder = [2]
    for order in orderslist_secondorder:
       print("order: ", order)
       C0NCtestSpaces = [-1,order-1,order-2]
       print("C0 non conforming test spaces: ", C0NCtestSpaces)
       eoc_solve, expected_eoc = runTesthk( C0NCtestSpaces, order, False, False )

       checkEOC(eoc_solve, expected_eoc)

main()