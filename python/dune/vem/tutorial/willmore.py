from ufl import *
from dune.ufl import Constant, DirichletBC
from dune.vem import polyGrid, voronoiCells, vemSpace, vemScheme
import dune.fem

dune.fem.threading.use = 8   # enable multithreading
conforming = True            # either conforming and non-conforming
order = 4                    # ... space of order 4
N     = 50                   # number of Voronoi cells
tau   = Constant(5e-6,"dt")  # time step size
saveTimes  = [1e-4] # add other times for longer simulations, e.g.,
# saveTimes += [4e-4,7e-4]

# moments for conforming and non-conforming C^1 spaces
if conforming:
    testSpaces=[[0,0],[order-4,order-3],order-4]
    name="willmore"+str(order)+"C"
else:
    testSpaces=[0,[order-3,order-2],order-4]
    name="willmore"+str(order)+"NC"

# setup grid and space
gridView = polyGrid( voronoiCells([[0,0],[1,1]],N,lloyd=200) )
space    = vemSpace(gridView, order=order, testSpaces=testSpaces)

# time, space coordinate, initial conditions and solution function
x   = SpatialCoordinate(space)
initial = sin(2*pi*x[0])**2*sin(2*pi*x[1])**2
df  = space.interpolate(initial, name="solution") # main solution

# Wilmore functional W(psi) and variational form
psi = Coefficient(space)
Q = lambda p: 1+inner(p,p)
E = lambda p: 1/Q(p)**0.25 * ( Identity(2) - outer(p,p)/Q(p) )
W = 1/2*inner( E(grad(psi)),grad(grad(psi)) )**2 * dx
phi = TestFunction(space)
a   = derivative(W,psi,phi)

# third order two stage time stepping (Gauss radau collocation)
# .. space with dimension 2 (df=U^n, U[0]=intermediate, U[1]=U^{n+1})
rkSpace = vemSpace(gridView, order=order,
                   dimRange=2,testSpaces=testSpaces)
stages  = rkSpace.interpolate([initial,initial], name="rk")
U       = TrialFunction(rkSpace)
V       = TestFunction(rkSpace)
dfs     = [  df,   U[0],  U[1]  ]
factors = [ [-2,  3./2., 1./2.] ,
            [ 2, -9./2., 5./2.] ]
dtForm  = lambda w,u,v: inner(w/Q(u),v)*dx
rkForm  = sum(f*dtForm(u,U[i],V[i])
              for i in range(2) for u,f in zip(dfs,factors[i]))
rkForm += tau*sum(replace(a,{psi:U[i],phi:V[i]}) for i in range(2))

# set up actual Newton solver using a direct solver for the linear part
dbc = DirichletBC(rkSpace, rkSpace.dimRange*[0])
factor = 40 # used for stabilization, determined experimentally
scheme = vemScheme( [rkForm == 0, dbc],
            solver=("suitesparse","umfpack"),
            hessStabilization=[factor*tau,factor*tau],
            gradStabilization=[factor*tau,factor*tau],
            massStabilization=[factor,factor],
            parameters={"newton.tolerance": 1e-6,
                        "newton.lineSearch": "simple"})

# time loop
t = 0

vtk = gridView.sequencedVTK(name, pointdata=[df], subsampling=2)
vtk()
saveNext = saveTimes.pop()
while True:
    info = scheme.solve(target=stages)
    # the second stage of the RK method is the approximation at t^{n+1}
    df.interpolate(stages[1])
    t += tau.value
    print("time",t,tau.value,info["iterations"],flush=True)
    # increase time step if Newton iterations drop to 5 or lower
    if info["iterations"]<=5: tau.value *= 1.25
    if t > saveNext:
        print("    saving....",flush=True)
        vtk()
        if len(saveTimes) == 0: break
        saveNext = saveTimes.pop()
