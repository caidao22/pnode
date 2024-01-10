import petsc4py
from petsc4py import PETSc

_dump_data_when_ksp_fails = False # for debugging only

class PCShell:
    def __init__(self, A, m, n, use_cuda, always_update_jacobian):
        self._m = m
        self._n = n
        self._ksp = PETSc.KSP()
        self._ksp.create(PETSc.COMM_WORLD)
        self._ksp.setType("hpddm")
        self._ksp.setOperators(A)
        self._ksp.setOptionsPrefix("pnode_inner_")
        self._ksp.getPC().setType("none")
        self._ksp.setHPDDMType(PETSc.KSP.HPDDMType.BGMRES)
        self._ksp.setErrorIfNotConverged(True)
        self._ksp.setFromOptions()
        self._use_cuda = use_cuda
        self._random = PETSc.Random()
        self._random.create(PETSc.COMM_WORLD)
        self._random.setInterval([-0.1, 0.1])

    def apply(self, pc, x, y):
        # SNES uses a zero initial guess for KSP by default
        # We force a nonzero initial guess to circumvent the difficulty in dealing with
        # zero columns in the RHS mat when using KSPMatSolve()
        if self._ksp.getInitialGuessNonzero():
            y.setRandom(self._random)
        if self._use_cuda:
            xhdl = x.getCUDAHandle("r")
            yhdl = y.getCUDAHandle("rw")
            X = PETSc.Mat().createDenseCUDA([self._n, self._m], cudahandle=xhdl)
            Y = PETSc.Mat().createDenseCUDA([self._n, self._m], cudahandle=yhdl)
            x.restoreCUDAHandle(xhdl, "r")
            y.restoreCUDAHandle(yhdl, "rw")
        else:
            X = PETSc.Mat().createDense([self._n, self._m], array=x.array_r)
            Y = PETSc.Mat().createDense([self._n, self._m], array=y.array)
        self._ksp.matSolve(X, Y)
        if _dump_data_when_ksp_fails and not self._ksp.is_converged:
           A, _ = self._ksp.getOperators()
           viewer = PETSc.Viewer().createBinary('A_mat', 'w')
           A.view(viewer)
           viewer = PETSc.Viewer().createBinary('rhs_mat', 'w')
           X.view(viewer)
        X.destroy()
        Y.destroy()

    def applyTranspose(self, pc, x, y):
        if self._ksp.getInitialGuessNonzero():
            y.setRandom(self._random)
        if self._use_cuda:
            xhdl = x.getCUDAHandle("r")
            yhdl = y.getCUDAHandle("rw")
            X = PETSc.Mat().createDenseCUDA([self._n, self._m], cudahandle=xhdl)
            Y = PETSc.Mat().createDenseCUDA([self._n, self._m], cudahandle=yhdl)
            x.restoreCUDAHandle(xhdl, "r")
            y.restoreCUDAHandle(yhdl, "rw")
        else:
            X = PETSc.Mat().createDense([self._n, self._m], array=x.array_r)
            Y = PETSc.Mat().createDense([self._n, self._m], array=y.array)
        self._ksp.matSolveTranspose(X, Y)
        X.destroy()
        Y.destroy()
