import math
import argparse

import dune.fem
import dune.fem.plotting
import dune.grid
import dune.ufl
import dune.vem
import matplotlib.pyplot as plt
import numpy as np
import ufl

dune.fem.threading.use = dune.fem.threading.max

# ************************ #
# Parameters of the script #
# ************************ #
parser = argparse.ArgumentParser(description='2nd order pde using C^1 spaces')
parser.add_argument('-m', type=str, required=True,
                          choices=["C1c","C1nc","C1C0"],
                          help="method (C1c,C1nc,C1C0)")
parser.add_argument('-M', type=str, required=True,
                          choices=["cube","simplex","voronoi"],
                          help="mesh (cube,simplex,voronoi)")
parser.add_argument('-p', type=str, required=True,
                          choices=["interpolation","poisson","nonvariational","bilaplace"],
                          help="pde (interpolation,poisson,nonvariational,bilaplace)")
parser.add_argument('-u', type=str, required=True,
                          choices=["u1","u2","u3"],
                          help="solution (u1,u2,u3)")
parser.add_argument('-A', type=str, required=True,
                          default="id",
                          choices=["id","coercive","notH2"],
                          help="solution (id,coercive,notH2)")
parser.add_argument('-l', type=int, required=True, help="polynomial degree (>=2)")
args = parser.parse_args()

method = args.m
gridtype = args.M
pde = args.p
solution_name = args.u
problemA = args.A
order = args.l

useCrissCross = False

# **************** #
# Helper functions #
# **************** #


def H(w):
    return ufl.grad(ufl.grad(w))


def laplace(w):
    return ufl.div(ufl.grad(w))


def bilaplace(w):
    return laplace(laplace(w))


# **************** #
# Main Python code #
# **************** #


def errors(n):
    if gridtype == "voronoi":
        domain = dune.vem.voronoiCells([[0,0],[1,1]], 9*n*n, lloyd=100, load="voronoi")
    else:
        domain = dune.grid.cartesianDomain([0, 0], [1, 1], [12*n, 12*n])
    grid = dune.vem.polyGrid(domain, cubes=(gridtype=="cube") )
    if gridtype == "simplex" and useCrissCross:
        grid.hierarchicalGrid.globalRefine(1)
    """
    indexSet = grid.indexSet
    @dune.grid.gridFunction(grid, name="cells", order=1)
    def polygons(en,x):
        return grid.hierarchicalGrid.agglomerate(indexSet.index(en))
    polygons.plot()
    """

    if method == "C1c":
        testSpaces = [[0, 0], [order - 4, order - 3], [order - 4]]
    elif method == "C1nc":
        testSpaces = [[0], [order - 3, order - 2], [order - 4]]
    elif method == "C1C0":
        testSpaces = [[0], [order - 2, order - 2], [order - 4]]
        # testSpaces = [[0, 0], [order - 4, order - 3], [order - 2]]
        # testSpaces = [0, order - 2, order - 2]
        # testSpaces = [[0], [order - 3, order - 2], [order - 4]]
        # testSpaces = [-1, order - 1, order - 2]

    space = dune.vem.vemSpace(grid, order=order, testSpaces=testSpaces)
    # space = dune.fem.space.lagrange(grid, order=order)

    x = ufl.SpatialCoordinate(space)

    if solution_name == "u1":
        exact = ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(ufl.pi * x[1]) ** 2
    elif solution_name == "u2":
        exact = ufl.sin(ufl.pi * x[0]) ** 2 + ufl.sin(ufl.pi * x[1]) ** 2
    elif solution_name == "u3":
        exact = ufl.sin(2.7*ufl.pi*x[0])*ufl.cos(2.2*ufl.pi*x[1])
    exact = dune.fem.function.uflFunction(grid, name="exact", order=space.order, ufl=exact )

    u      = ufl.TrialFunction(space)
    v      = ufl.TestFunction(space)
    normal = ufl.FacetNormal(space)
    tau    = ufl.as_vector([ -normal[1], normal[0] ])
    hbnd   = ufl.CellVolume(space) / ufl.FacetArea(space)
    hint   = ufl.avg(ufl.CellVolume(space)) / ufl.FacetArea(space)
    beta   = dune.ufl.Constant(10 * space.order**2,"beta")
    dbc    = dune.ufl.DirichletBC(space, exact)

    solution = dune.fem.function.discreteFunction(space, name="solution")
    solution.interpolate(exact)
    interMax = max(solution.as_numpy)
    # print("interpolation:", interMax)

    if pde == "bilaplace":
        a = ufl.inner(H(u), H(v)) * ufl.dx
        b = ufl.inner(H(exact), H(v)) * ufl.dx
        # b = bilaplace(exact) * v * ufl.dx

        scheme = dune.vem.vemScheme(
            [a == b, dbc],
            space,
            solver=("suitesparse","umfpack"),
            parameters={"newton.linear.verbose":False,
                        "newton.verbose":False,
                        "newton.linear.preconditioning.method":"jacobi",
                        "newton.linear.tolerance":1e-12,
                        "newton.maxiterations":10},
            hessStabilization=1,
            boundary="full",
        )
        scheme.solve(target=solution)
    elif pde == "poisson":
        a  = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        b  = -laplace(exact) * v * ufl.dx # +\
        a -= ufl.dot(ufl.grad(u),normal) * v * ufl.ds +\
             ufl.dot(ufl.grad(v),normal) * (u-exact) * ufl.ds
        a += beta/hbnd * (u-exact) * v * ufl.ds
        # a += beta * ufl.dot(ufl.grad(u-exact),tau) * ufl.dot(ufl.grad(v),tau) * ufl.ds

        scheme = dune.vem.vemScheme(
            [a == b],
            solver=("suitesparse","umfpack"),
            gradStabilization=1,
            massStabilization=0,
            parameters={"newton.linear.verbose":False,
                        "newton.verbose":False,
                        "newton.maxiterations":10},
            # boundary=None,
            # boundary="value",
        )
        """
        scheme = dune.fem.scheme.galerkin( a == b, solver=("suitesparse","umfpack") )
        """
        scheme.solve(target=solution)
    elif pde == "nonvariational":
        if problemA == "id":
            d11=d22 = 1
            d12 = 0
        if problemA == "coercive":
            d11 = 1
            d22 = 1-ufl.ln((x[0]-ufl.sqrt(3)/3)**2+10**(-4))
            d12 = 0
        if problemA == "notH2":
             d11 = 1
             d22 = 2
             d12 = (x[0]*x[1])**(2./3.)
        A = ufl.as_matrix([ [d11,d12], [d12,d22] ])
        lam = (1+d22)/2 + ufl.sqrt( (1+d22)**2/4 - d22+d12**2 )
        a  = -ufl.inner(A,H(u-exact)) * v * ufl.dx
        a += beta/hbnd*lam * (u-exact) * v * ufl.ds
        a += beta*hbnd*lam * ufl.dot(ufl.grad(u-exact),tau) * ufl.dot(ufl.grad(v),tau) * ufl.ds

        # possibly of interest for the non-conforming spaces but does not work at the moment
        # a += beta/hint * ufl.jump(u) * ufl.jump(v) * ufl.dS
        # a += beta/hint * inner( ufl.jump(grad(u)), ufl.jump(grad(v)) ) * ufl.dS

        scheme = dune.vem.vemScheme(
            [a == 0],
            solver=("suitesparse","umfpack"),
            hessStabilization=0,
            gradStabilization=lam,
            massStabilization=0,
            parameters={"newton.linear.verbose":False,
                        "newton.verbose":False,
                        "newton.maxiterations":10},
        )
        """
        scheme = dune.fem.scheme.galerkin( a == b, solver=("suitesparse","umfpack") )
        """
        scheme.solve(target=solution)

    # solution.as_numpy[:] *= interMax / max(solution.as_numpy)
    # for d in solution.as_numpy: print(d,end=" ")
    # print("solution:",max(solution.as_numpy))
    grid.writeVTK(f"{gridtype}{method}{solution_name}{pde}{problemA}{order}_{n}",
                                  celldata={"exact":exact,
                                            "solution":solution,
                                            "dxExact":ufl.grad(exact)[0],
                                            "dxSol":ufl.grad(solution)[0]},
                  subsampling=2)
    # solution.plot()

    error = solution - exact

    error_l2 = math.sqrt(
        dune.fem.function.integrate(grid, ufl.inner(error, error), order=5)
    )
    error_h1 = math.sqrt(
        dune.fem.function.integrate(
            grid, ufl.inner(ufl.grad(error), ufl.grad(error)), order=5
        ))
    error_h2 = math.sqrt(
        dune.fem.function.integrate(grid, ufl.inner(H(error), H(error)), order=5)
    )
    print(n,space.size,"   ",error_l2, error_h1, error_h2)

    return error_l2, error_h1, error_h2, space.size < 100000


def main():
    # N = np.stack([2 * 2 ** np.arange(4), 3 * 2 ** np.arange(4)], axis=-1).flatten()
    N = 2 ** np.arange(7)
    errors_l2 = np.zeros(N.shape)
    errors_h1 = np.zeros(N.shape)
    errors_h2 = np.zeros(N.shape)

    for i, n in enumerate(N):
        errors_l2[i], errors_h1[i], errors_h2[i], ctn = errors(n)
        if not ctn: break
    N = N[:i+1]
    errors_l2 = errors_l2[:i+1]
    errors_h1 = errors_h1[:i+1]
    errors_h2 = errors_h2[:i+1]
    print("# L^2-EOCs:", np.log( errors_l2[1:] / errors_l2[:-1] ) / np.log(0.5) )
    print("# H^1-EOCs:", np.log( errors_h1[1:] / errors_h1[:-1] ) / np.log(0.5) )
    print("# H^2-EOCs:", np.log( errors_h2[1:] / errors_h2[:-1] ) / np.log(0.5) )

    start = lambda p: np.sqrt(errors_h1[0]*errors_l2[0])*N[0]**p
    plt.loglog(N, start(1) / N, "k:", label="order = 1")
    plt.loglog(N, start(2) / N**2, "k--", label="order = 2")
    plt.loglog(N, start(3) / N**3, "k-.", label="order = 3")
    plt.loglog(N, start(4) / N**4, "k-", label="order = 4")

    plt.loglog(N, errors_l2, ".-", label="$L^2$ error")
    plt.loglog(N, errors_h1, ".-", label="$H^1$ error")
    plt.loglog(N, errors_h2, ".-", label="$H^2$ error")

    plt.xlabel("N")
    plt.legend()
    plt.savefig(f"{gridtype}{method}{solution_name}{pde}{problemA}{order}.png",dpi=300)

main()
