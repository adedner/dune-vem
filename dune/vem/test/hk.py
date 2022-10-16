from ufl import *
import dune.ufl

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from script import runTest, checkEOC, interpolate
from interpolate import interpolate_secondorder

dimR = 3

def hk(space, exact):
    x = SpatialCoordinate(space)
    massCoeff = 1+sin(dot(x,x))
    diffCoeff = 1-0.9*cos(dot(x,x))
    assert(dimR == exact.ufl_shape[0]), "dim range not compatible"
#    dimR = exact.ufl_shape[0]
    u = TrialFunction(space)
    v = TestFunction(space)

    a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx
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
    orders = [1,3]
    for order in orders:
       print("order: ", order)

       with ProcessPoolExecutor(max_workers=4) as executor:
           # C0 non conforming VEM
           C0NCtestSpaces = [-1,order-1,order-2]
           print("C0 non conforming test spaces: ", C0NCtestSpaces)
           C0NCFuture = executor.submit( runTesthk, C0NCtestSpaces, order, False, False )

           # C0 conforming VEM
           C0testSpaces = [0,order-2,order-2]
           print("C0 conforming test spaces: ", C0testSpaces)
           C0Future = executor.submit( runTesthk, C0testSpaces, order, False, False )

           # C0 conforming VEM with vectorSpace=true
           print("C0 conforming test spaces with vectorSpace=True")
           C0vecFuture = executor.submit( runTesthk, C0testSpaces, order, True, False )

           # C0 serendipity VEM
           C0serendipitytestSpaces = [0,order-2,order-3]
           print("C0 serendipity test spaces: ", C0serendipitytestSpaces)
           C0serFuture = executor.submit( runTesthk, C0serendipitytestSpaces, order, False, False )


       # C0 non conforming VEM
       print("EOCs for C0 non conforming test spaces: ")
       eoc, expected_eoc = C0NCFuture.result()
       checkEOC(eoc, expected_eoc)

       # C0 conforming VEM
       print("EOCs for C0 conforming test spaces: ")
       eoc, expected_eoc = C0Future.result()
       checkEOC(eoc, expected_eoc)

       # C0 conforming VEM with vectorSpace=true
       print("EOCs for C0 conforming test spaces with vectorSpace=True")
       eoc, expected_eoc = C0vecFuture.result()
       checkEOC(eoc, expected_eoc)

       # C0 serendipity VEM
       print("EOCs for C0 serendipity test spaces: ")
       eoc, expected_eoc = C0serFuture.result()
       checkEOC(eoc, expected_eoc)

main()
