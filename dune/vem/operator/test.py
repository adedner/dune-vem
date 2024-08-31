from dune.grid import cartesianDomain
from dune.vem import polyGrid
from dune.vem import vemSpace

grid = polyGrid( cartesianDomain([0,0],[1,1],[10,10]) )
spc = vemSpace(grid)

S = spc.stabilization()
print(S.as_numpy)
