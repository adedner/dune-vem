# %% [markdown]
# # 2D navier stokes around a cylinder
#
# https://hplgit.github.io/fenics-tutorial/pub/sphinx1/._ftut1004.html#flow-past-a-cylinder

# %%
import matplotlib

matplotlib.rc("image", cmap="jet")
from math import sqrt
import matplotlib.pyplot as plt
import pygmsh
from dune.ufl import Constant, DirichletBC
import dune.fem.space
import dune.fem.scheme
import dune.vem
from uzawa import Uzawa, plot
from ufl import dot, nabla_grad, SpatialCoordinate, as_vector

explicit = False
useVem = True
order = 2
T   = 5.0  # final time
# tau = T / 10000
# tau = T / 5000
tau = T / 4000


with pygmsh.occ.Geometry() as geom:
    # geom.characteristic_length_max = 0.02
    geom.set_mesh_size_callback( lambda dim, tag, x, y, z, lc:
          min(0.04+(x-0.2)**2+(y-0.2)**2 - 0.1**2, 0.1) ) # p=4
          # min(0.01+(x-0.2)**2+(y-0.2)**2 - 0.1**2, 0.1) ) # p=2
          #  min(0.005+(x-0.2)**2+(y-0.2)**2 - 0.05**2, 0.02) )
    rectangle = geom.add_rectangle([0, 0, 0], 2.2, 0.41)
    cylinder = geom.add_disk([0.2, 0.2, 0.0], 0.05)
    geom.boolean_difference(rectangle, cylinder)

    mesh = geom.generate_mesh()
    points, cells = mesh.points, mesh.cells_dict
    domain = {
        "vertices": points[:, :2].astype(float),
        "simplices": cells["triangle"].astype(int),
    }
    print(len(domain["simplices"]))

gridView = dune.vem.polyGrid(domain)
# gridView.plot()
# plt.show()
if useVem:
    if explicit:
        path="uzawaeVem_p" + str(order)
    else:
        path="uzawaVem_p" + str(order)
    space_u = dune.vem.divFreeSpace( gridView, order=order)
    space_p = dune.fem.space.finiteVolume( gridView )
else:
    if explicit:
        path="uzawaeTH_p" + str(order)
    else:
        path="uzawaTH_p" + str(order)
    space_u = dune.fem.space.lagrange( gridView, order=order, dimRange=2)
    space_p = dune.fem.space.lagrange( gridView, order=order-1 )

mu  = 0.001

u_h = space_u.interpolate([0, 0], name="u_h")
u_h_n = space_u.interpolate([0, 0], name="u_h")
p_h = space_p.interpolate(0, name="p_h")

x = SpatialCoordinate(space_u)
inflow = [6 * x[1] * (0.41 - x[1]) / 0.41**2, 0] # 6->18
dbc_u_in = DirichletBC(space_u, inflow, x[0] <= 1e-8)
dbc_u_noflow = DirichletBC(space_u, [0, 0], None)
dbc_u_out = DirichletBC(space_u, [None,None], 2.2 - x[0] <= 1e-8)
dbc_p_out = DirichletBC(space_p, 0, 2.2 - x[0] <= 1e-8)

# %%
from pathlib import Path
p = Path(path)
p.mkdir(exist_ok=True)
vtk = gridView.sequencedVTK(path+f"/2DCylinder",
              subsampling=3,
              pointdata=[p_h], pointvector=[u_h])
vtk()

time = 0
saveStep = 0.05
nextSave = saveStep

uzawa = Uzawa(gridView, space_u, space_p,
              [dbc_u_in,dbc_u_out,dbc_u_noflow], as_vector([0,0]),
              mu, tau, u_h_n,
              tolerance=1e-6, precondition=True, verbose=False,
              lagrange=not useVem, explicit=explicit)
while time < T:
    u_h_n.assign(u_h)

    info = uzawa.solve([u_h,p_h])

    time += tau
    print(time, info, flush=True)
    if time > nextSave:
        vtk()
        nextSave += saveStep
