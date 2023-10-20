import math, numpy, scipy
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, bmat
import dune.fem

# (  M   G  ) ( u     )   ( g ) # g: Dirichlet BC - Neuman BC are natural
# ( -D   0  ) ( sigma ) = ( f )
# sigma = -M^{-1}Gu + M^{-1}g
# f = -Dsigma = DM^{-1}Gu - DM^{-1}g
# DM^{-1}Gu = f + DM^{-1}g
# Au = b with A = DM^{-1}G and b = f + DM^{-1}g
class MixedSolver(scipy.sparse.linalg.LinearOperator):
    def __init__(self, schemeM, schemeG, schemeD):
        self.schemeD = schemeD # needed for rhs
        self.schemeM = schemeM
        self.M = schemeM.linear().as_numpy
        self.invM = scipy.sparse.linalg.splu(self.M.tocsc())
        self.G = schemeG.linear().as_numpy
        self.D = schemeD.linear().as_numpy
        self.tmp = schemeD.domainSpace.interpolate([0,0],name="rhs")
        self.rhs = schemeD.rangeSpace.interpolate(0,name="rhs")
        self.s1 = self.tmp.copy()
        # schemeD.domainSpace.interpolate([0,0],name="tmp").as_numpy[:]
        self.s0 = self.tmp.copy()
        # schemeD.domainSpace.interpolate([0,0],name="tmp").as_numpy[:]
        self.y  = self.rhs.as_numpy.copy()
        # schemeD.rangeSpace.interpolate(0,name="tmp").as_numpy[:]
        self.shape = (schemeD.rangeSpace.size, schemeD.rangeSpace.size)
        self.dtype = self.y.dtype
        self.count = 0

    def update(self, x_coeff, f):
        pass
    def _matvec(self, x_coeff):
        self.s0.as_numpy[:] = self.G@x_coeff[:]
        # self.s1[:] = scipy.sparse.linalg.cg(self.M,self.s0[:])[0]
        # spsolve(self.M,self.s0[:])
        # self.schemeM.solve(rhs=self.s0,target=self.s1)
        self.s1.as_numpy[:] = self.invM.solve(rhs=self.s0.as_numpy[:])
        self.y[:] = self.D@self.s1.as_numpy[:]
        return self.y
    def callback(self,xk):
        self.count += 1
        self.y[:] = self._matvec(xk)
        self.y[:] -= self.rhs.as_numpy[:]
        print(self.count, numpy.dot(self.y,self.y))

    def solve(self,target):
        self.tmp.clear()
        self.schemeD(self.tmp,self.rhs)
        target[0].as_numpy[:] = scipy.sparse.linalg.cg(
          self,self.rhs.as_numpy, atol=0
          # ,callback=lambda xk:self.callback(xk)
          )[0]
        self.tmp.as_numpy[:]  = -self.G@target[0].as_numpy[:]
        self.schemeM.solve(rhs=self.tmp,target=target[1])
        # target[1].as_numpy[:] = spsolve(self.M,self.tmp.as_numpy[:])
