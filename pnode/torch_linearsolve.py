import torch
import torch.utils.dlpack as dlpack
import petsc4py
from petsc4py import PETSc

class PCShell:
    def __init__(self, A, m, n, use_cuda, always_update_jacobian):
        self._m = m # batch dimension
        self._n = n # data size
        self._A = A
        self._always_update_jacobian = always_update_jacobian
        self._LU = None
        self._pivots = None

    def get_factor(self):
        if self._LU is None or self._always_update_jacobian:
            A_tensor = dlpack.from_dlpack(self._A)
            self._LU, self._pivots = torch.linalg.lu_factor(A_tensor)
        return self._LU, self._pivots
    
    def reset_factor(self):
        self._LU = None
        self._pivots = None

    def apply(self, pc, x, y):
        X = dlpack.from_dlpack(x.toDLPack(mode="r")).reshape(self._m, self._n)
        Y = dlpack.from_dlpack(y).reshape(self._m, self._n)
        LU, pivots = self.get_factor()
        torch.linalg.lu_solve(LU, pivots, X, out=Y, left=False)

    def applyTranspose(self, pc, x, y):
        X = dlpack.from_dlpack(x.toDLPack(mode="r")).reshape(self._m, self._n)
        Y = dlpack.from_dlpack(y).reshape(self._m, self._n)
        LU, pivots = self.get_factor()
        torch.linalg.lu_solve(LU, pivots, X, out=Y, adjoint=True, left=False)
