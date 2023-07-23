import torch
import torch.nn as nn
import torch.utils.dlpack as dlpack
from .misc import _flatten, _flatten_convert_none_to_zeros
import petsc4py
from petsc4py import PETSc

# def _mat_shift_and_scale(A, X, Y):
#     Y.scale(A.vscale)
#     if A.vshift != 0.0: Y.axpy(A.vshift, X)


class RHSJacShell:
    def __init__(self, ode):
        self.ode_ = ode
        # self.vshift = 0.0
        # self.vscale = 1.0

    def mult(self, A, X, Y):
        """The Jacobian is A = shift*I - dFdU"""
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            x_tensor = dlpack.from_dlpack(X.toDLPack(mode="r"))
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y)
        else:
            x_tensor = torch.from_numpy(X.array_r.reshape(self.ode_.tensor_size)).to(
                device=self.ode_.device, dtype=self.ode_.tensor_dtype
            )
            y = Y.array
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.funcEX(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_u = torch.autograd.grad(
                func_eval,
                self.ode_.cached_u_tensor,
                x_tensor,
                allow_unused=True,
                create_graph=True,
            )
            x_tensor.requires_grad_(True)
            jvp_u = torch.autograd.grad(vjp_u[0], x_tensor, x_tensor, allow_unused=True)
            self.ode_.cached_u_tensor.requires_grad_(False)
        if jvp_u[0] is None:
            jvp_u[0] = torch.zeros_like(y)
        if self.ode_.use_dlpack:
            y.copy_(jvp_u[0])
        else:
            y[:] = jvp_u[0].cpu().numpy().flatten()
        # _mat_shift_and_scale(self, X, Y)

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            x_tensor = dlpack.from_dlpack(X.toDLPack(mode="r"))
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y)
        else:
            x_tensor = torch.from_numpy(X.array_r.reshape(self.ode_.tensor_size)).to(
                device=self.ode_.device, dtype=self.ode_.tensor_dtype
            )
            y = Y.array
        f_params = tuple(
            filter(lambda p: p.requires_grad, self.ode_.funcEX.parameters())
        )
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.funcEX(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_u, *self.ode_.vjp_params = torch.autograd.grad(
                func_eval,
                (self.ode_.cached_u_tensor,) + f_params,
                x_tensor,
                allow_unused=True,
                retain_graph=True,
            )
        # autograd.grad returns None if no gradient, set to zero.
        # vjp_u = tuple(torch.zeros_like(y_) if vjp_u_ is None else vjp_u_ for vjp_u_, y_ in zip(vjp_u, y))
        if vjp_u is None:
            vjp_u = torch.zeros_like(y)
        if self.ode_.use_dlpack:
            y.copy_(vjp_u)
        else:
            y[:] = vjp_u.cpu().numpy().flatten()

    # def scale(self, A, a):
    #    self.vscale = self.vscale * a
    #    self.vshift = self.vshift * a

    # def shift(self, A, a):
    #    self.vshift = self.vshift + a


class IJacShell:
    def __init__(self, ode):
        self.ode_ = ode
        # self.vshift = 0.0
        # self.vscale = 1.0

    def mult(self, A, X, Y):
        """The Jacobian is A = shift*I - dFdU"""
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            self.x_tensor = dlpack.from_dlpack(X.toDLPack(mode="r"))
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y)
        else:
            self.x_tensor = torch.from_numpy(
                X.array_r.reshape(self.ode_.tensor_size)
            ).to(device=self.ode_.device, dtype=self.ode_.tensor_dtype)
            y = Y.array
        jvp_u = self._jvp(self.x_tensor, y)
        if self.ode_.use_dlpack:
            if self.ode_.mass is None:
                y.copy_(self.x_tensor.mul(self.ode_.shift) - jvp_u[0])
            else:
                y.copy_(
                    torch.matmul(self.ode_.mass, self.x_tensor.mul(self.ode_.shift))
                    - jvp_u[0]
                )
        else:
            if self.ode_.mass is None:
                y[:] = self.ode_.shift * X.array_r - jvp_u[0].cpu().numpy().flatten()
            else:
                y[:] = (
                    self.ode_.shift * self.ode_.mass.cpu().numpy().dot(X.array_r)
                    - jvp_u[0].cpu().numpy().flatten()
                )
        # _mat_shift_and_scale(self, X, Y)

    def _jvp(self, x, y):
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.funcIM(self.ode_.t, self.ode_.cached_u_tensor)
            x = x.detach().requires_grad_(True)
            vjp_u = torch.autograd.grad(
                func_eval,
                self.ode_.cached_u_tensor,
                x,
                allow_unused=True,
                create_graph=True,
            )
            jvp_u = torch.autograd.grad(vjp_u[0], x, x, allow_unused=True)
        if jvp_u[0] is None:
            jvp_u[0] = torch.zeros_like(y)
        return jvp_u

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            self.x_tensor = dlpack.from_dlpack(X.toDLPack(mode="r"))
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y)
        else:
            self.x_tensor = torch.from_numpy(
                X.array_r.reshape(self.ode_.tensor_size)
            ).to(device=self.ode_.device, dtype=self.ode_.tensor_dtype)
            y = Y.array
        vjp_u = self._vjp(self.x_tensor, y)
        if self.ode_.use_dlpack:
            if self.ode_.mass is None:
                y.copy_(torch.mul(self.x_tensor, self.ode_.shift) - vjp_u)
            else:
                y.copy_(
                    torch.matmul(
                        self.ode_.mass.transpose(-2, -1),
                        torch.mul(self.x_tensor, self.ode_.shift),
                    )
                    - vjp_u
                )
        else:
            if self.ode_.mass is None:
                y[:] = self.ode_.shift * X.array_r - vjp_u.cpu().numpy().flatten()
            else:
                y[:] = (
                    self.ode_.shift
                    * self.ode_.mass.transpose(-2, -1).cpu().numpy().dot(X.array_r)
                    - vjp_u.cpu().numpy().flatten()
                )

    def _vjp(self, x, y):
        f_params = tuple(
            filter(lambda p: p.requires_grad, self.ode_.funcIM.parameters())
        )
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.funcIM(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_u, *self.ode_.vjp_params = torch.autograd.grad(
                func_eval,
                (self.ode_.cached_u_tensor,) + f_params,
                x,
                allow_unused=True,
                retain_graph=True,
            )
        # autograd.grad returns None if no gradient, set to zero.
        # vjp_u = tuple(torch.zeros_like(y_) if vjp_u_ is None else vjp_u_ for vjp_u_, y_ in zip(vjp_u, y))
        if vjp_u is None:
            vjp_u = torch.zeros_like(y)
        return vjp_u

    # MatProduct for KSPMatSolve
    def productSetFromOptions(self, mat, producttype, A, B, C):
        return True

    def productSymbolic(self, mat, product, producttype, A, B, C):
        product.setType(B.getType())
        product.setSizes(B.getSizes())
        product.setUp()
        product.assemble()

    def productNumeric(self, mat, product, producttype, A, B, C):
        if producttype == "AB" or producttype == "AtB":
            if self.ode_.use_dlpack:
                self.x_tensor = dlpack.from_dlpack(B.toDLPack(mode="r")).T.view(
                    self.ode_.tensor_size
                )
                P_tensor = dlpack.from_dlpack(product).T.view(self.ode_.tensor_size)
            else:
                self.x_tensor = (
                    torch.from_numpy(B.getDenseArray())
                    .view(self.ode_.tensor_size)
                    .to(device=self.ode_.device, dtype=self.ode_.tensor_dtype)
                )
                P_tensor = (
                    torch.from_numpy(product.getDenseArray())
                    .view(self.ode_.tensor_size)
                    .to(device=self.ode_.device, dtype=self.ode_.tensor_dtype)
                )
        if producttype == "AB":
            jvp_u = self._jvp(self.x_tensor, P_tensor)
            if self.ode_.use_dlpack:
                if self.ode_.mass is None:
                    P_tensor.copy_(self.x_tensor.mul(self.ode_.shift) - jvp_u[0])
                else:
                    P_tensor.copy_(
                        torch.matmul(self.ode_.mass, self.x_tensor.mul(self.ode_.shift))
                        - jvp_u[0]
                    )
            else:
                if self.ode_.mass is None:
                    P_tensor[:] = (
                        self.ode_.shift * B.getDenseArray().flattern()
                        - jvp_u[0].cpu().numpy().flatten()
                    )
                else:
                    P_tensor[:] = (
                        self.ode_.shift
                        * self.ode_.mass.cpu().numpy().dot(B.getDenseArray().flattern())
                        - jvp_u[0].cpu().numpy().flatten()
                    )

        if producttype == "AtB":
            vjp_u = self._vjp(self.x_tensor, P_tensor)
            if self.ode_.use_dlpack:
                if self.ode_.mass is None:
                    P_tensor.copy_(torch.mul(self.x_tensor, self.ode_.shift) - vjp_u)
                else:
                    P_tensor.copy_(
                        torch.matmul(
                            self.ode_.mass.transpose(-2, -1),
                            torch.mul(self.x_tensor, self.ode_.shift),
                        )
                        - vjp_u
                    )
            else:
                if self.ode_.mass is None:
                    P_tensor[:] = (
                        self.ode_.shift * B.getDenseArray().flatten()
                        - vjp_u.cpu().numpy().flatten()
                    )
                else:
                    P_tensor[:] = (
                        self.ode_.shift
                        * self.ode_.mass.transpose(-2, -1)
                        .cpu()
                        .numpy()
                        .dot(B.getDenseArray().flattern())
                        - vjp_u.cpu().numpy().flatten()
                    )


class IJacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def _vjp(self, x, y):
        """
        When the IJacobian is provided explicitly (use_hpddm=True and matrixfree_hpddm=False), vjp_params cannot
        be obtained from IJac's multTranspose(). So we need to compute it here.
        """
        f_params = tuple(
            filter(lambda p: p.requires_grad, self.ode_.funcIM.parameters())
        )
        with torch.set_grad_enabled(True):
            func_eval = self.ode_.funcIM(self.ode_.t, self.ode_.cached_u_tensor)
            self.ode_.vjp_params = torch.autograd.grad(
                func_eval,
                f_params,
                x,
                allow_unused=True,
                retain_graph=True,
            )

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            Y.attachDLPackInfo(self.ode_.adj_p[0])
            y = dlpack.from_dlpack(Y)
            if self.ode_.use_hpddm and not self.ode_.matrixfree_hpddm:
                X.attachDLPackInfo(self.ode_.cached_U)
                self.x_tensor = dlpack.from_dlpack(X.toDLPack(mode="r"))
        else:
            y = Y.array
            if self.ode_.use_hpddm and not self.ode_.matrixfree_hpddm:
                self.x_tensor = torch.from_numpy(
                    X.array_r.reshape(self.ode_.tensor_size)
                ).to(device=self.ode_.device, dtype=self.ode_.tensor_dtype)
        f_params = tuple(
            filter(lambda p: p.requires_grad, self.ode_.funcIM.parameters())
        )
        if self.ode_.use_hpddm and not self.ode_.matrixfree_hpddm:
            self._vjp(self.x_tensor, y)
        vjp_params = _flatten_convert_none_to_zeros(self.ode_.vjp_params, f_params)
        if self.ode_.imex:
            vjp_params = torch.cat(
                (
                    vjp_params,
                    torch.zeros(self.ode_.npEX).to(
                        device=self.ode_.device, dtype=self.ode_.tensor_dtype
                    ),
                )
            )
        if self.ode_.use_dlpack:
            y.copy_(torch.mul(vjp_params, -1.0))
        else:
            y[:] = -vjp_params.cpu().numpy().flatten()


class RHSJacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            Y.attachDLPackInfo(self.ode_.adj_p[0])
            y = dlpack.from_dlpack(Y)
        else:
            y = Y.array
        f_params = tuple(
            filter(lambda p: p.requires_grad, self.ode_.funcEX.parameters())
        )
        vjp_params = _flatten_convert_none_to_zeros(self.ode_.vjp_params, f_params)
        if self.ode_.imex:
            vjp_params = torch.cat(
                (
                    torch.zeros(self.ode_.npIM).to(
                        device=self.ode_.device, dtype=self.ode_.tensor_dtype
                    ),
                    vjp_params,
                )
            )
        if self.ode_.use_dlpack:
            y.copy_(vjp_params)
        else:
            y[:] = vjp_params.cpu().numpy().flatten()


class ODEPetsc(object):
    comm = PETSc.COMM_SELF

    def __init__(self):
        self.ts = PETSc.TS().create(comm=self.comm)
        self.n = 0
        self.tensor_size = None
        self.tensor_dtype = None
        self.adj_u = []
        self.adj_p = []
        self.mass = None
        self.funcIM = None
        self.funcEX = None
        self.flat_params = None
        self.npIM = None
        self.npEX = None
        self.np = None
        self.imex = None
        self.use_dlpack = True
        self.use_cuda = False
        self.use_hpddm = False
        self.matrixfree_hpddm = True
        self.shift = None
        self.innerkspmat = None
        self.jacobianIM = None
        self.reset_jacobianIM = True

    def evalRHSFunction(self, ts, t, U, F):
        if self.use_dlpack:
            # have to call to() or type() to avoid a PETSc seg fault
            U.attachDLPackInfo(self.cached_U)
            u_tensor = dlpack.from_dlpack(U.toDLPack(mode="r"))
            # u_tensor = dlpack.from_dlpack(U.toDLPack(mode='r')).view(self.tensor_size).type(self.tensor_type)
            F.attachDLPackInfo(self.cached_U)
            dudt = dlpack.from_dlpack(F)
            # Resotring the handle set the offloadmask flag to PETSC_OFFLOAD_GPU, but it zeros out the GPU memory accidentally, which is probably a bug
            if self.use_cuda:
                hdl = F.getCUDAHandle("w")
                F.restoreCUDAHandle(hdl, "w")
            dudt.copy_(self.funcEX(t, u_tensor))
        else:
            f = F.array
            u_tensor = torch.from_numpy(U.array_r.reshape(self.tensor_size)).to(
                device=self.device, dtype=self.tensor_dtype
            )
            dudt = self.funcEX(t, u_tensor).cpu().detach().numpy()
            f[:] = dudt.flatten()

    def evalIFunction(self, ts, t, U, Udot, F):
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            u_tensor = dlpack.from_dlpack(U.toDLPack(mode="r"))
            Udot.attachDLPackInfo(self.cached_U)
            udot_tensor = dlpack.from_dlpack(Udot.toDLPack(mode="r"))
            # Resotring the handle set the offloadmask flag to PETSC_OFFLOAD_GPU, but it zeros out the GPU memory accidentally, which is probably a bug
            if self.use_cuda:
                hdl = F.getCUDAHandle("w")
                F.restoreCUDAHandle(hdl, "w")
            F.attachDLPackInfo(self.cached_U)
            dudt = dlpack.from_dlpack(F)
            if self.mass is None:
                dudt.copy_(udot_tensor - self.funcIM(t, u_tensor))
            else:
                dudt.copy_(
                    torch.matmul(self.mass, udot_tensor) - self.funcIM(t, u_tensor)
                )
        else:
            f = F.array
            u_tensor = torch.from_numpy(U.array_r.reshape(self.tensor_size)).to(
                device=self.device, dtype=self.tensor_dtype
            )
            dudt = self.funcIM(t, u_tensor).cpu().detach().numpy()
            if self.mass is None:
                f[:] = Udot.array_r - dudt.flatten()
            else:
                f[:] = self.mass.cpu().numpy().dot(Udot.array_r) - dudt.flatten()

    def evalRHSJacobian(self, ts, t, U, Jac, JacPre):
        """Cache t and U for matrix-free Jacobian"""
        self.t = t
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            # x = dlpack.from_dlpack(U.toDLPack(mode='r'))
            # self.cached_u_tensor.copy_(x)
            self.cached_u_tensor = dlpack.from_dlpack(U.toDLPack(mode="r"))
        else:
            self.cached_u_tensor = torch.from_numpy(
                U.array_r.reshape(self.tensor_size)
            ).to(device=self.device, dtype=self.tensor_dtype)
        # JacShell = Jac.getPythonContext()
        # JacShell.vshift = 0.0
        # JacShell.vscale = 1.0

    def evalIJacobian(self, ts, t, U, Udot, shift, Jac, JacPre):
        """Cache t and U for matrix-free Jacobian"""
        self.t = t
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            # x = dlpack.from_dlpack(U.toDLPack(mode='r'))
            # self.cached_u_tensor.copy_(x)
            self.cached_u_tensor = dlpack.from_dlpack(U.toDLPack(mode="r"))
        else:
            self.cached_u_tensor = torch.from_numpy(
                U.array_r.reshape(self.tensor_size)
            ).to(device=self.device, dtype=self.tensor_dtype)
        # JacShell = Jac.getPythonContext()
        # JacShell.vshift = 0.0
        # JacShell.vscale = 1.0
        if self.innerkspmat is not None and self.reset_jacobianIM:
            from torch.func import jacrev
            shape = self.innerkspmat.getSizes()[0]
            if self.reset_jacobianIM:
                func_eval = self.funcIM(t, self.cached_u_tensor[0:1])
                jacobianIM = jacrev(self.funcIM, argnums=1)(
                    t, self.cached_u_tensor[0:1]
                )
                self.jacobianIM = jacobianIM.view(*shape)
                self.reset_jacobianIM = False
        if self.innerkspmat is not None and (self.reset_jacobianIM or self.shift != shift): # reshift
            shape = self.innerkspmat.getSizes()[0]
            if self.use_dlpack:
                innerjac_tensor = dlpack.from_dlpack(
                    self.innerkspmat
                ).T  # return a transposed view switch from Fotran order to C order
                innerjac_tensor[:] = self.jacobianIM
                if self.mass is None:
                    innerjac_tensor.mul_(-1).add_(
                        shift * torch.eye(shape[0], device=self.device)
                    )
                else:
                    innerjac_tensor.mul_(-1).add_(shift * self.mass)
            else:
                innerkspmat = self.innerkspmat.getDenseArray().T
                if self.mass is None:
                    innerkspmat[:] = (
                        shift * torch.eye(shape[0])
                        - self.jacobianIM.detach().cpu().numpy()
                    )
                else:
                    innerkspmat[:] = (
                        shift * self.mass - self.jacobianIM.detach().cpu().numpy()
                    )
        self.shift = shift

    def evalRHSJacobianP(self, ts, t, U, Jacp):
        """t and U are already cached in Jacobian evaluation functions"""
        pass

    def evalIJacobianP(self, ts, t, U, Udot, shift, Jacp):
        """t and U are already cached in Jacobian evaluation functions"""
        pass

    def tspanPostStep(self, ts):
        """Set time step size"""
        stepno = ts.getStepNumber()
        t = ts.getTime()
        if self.cur_sol_index < len(self.sol_times):
            if isinstance(self.step_size, list):
                if stepno < len(self.step_size):
                    ts.setTimeStep(self.step_size[stepno])
            self.cur_sol_steps[self.cur_sol_index] += 1
            if self.tensor_dtype == torch.double:
                delta = 1e-5
            else:
                delta = 1e-3
            if abs(t - self.sol_times[self.cur_sol_index]) < delta:  # ugly workaround
                self.cur_sol_index = self.cur_sol_index + 1

    def setupTS(
        self,
        u_tensor,
        func,
        step_size=0.01,
        enable_adjoint=True,
        implicit_form=False,
        use_dlpack=True,
        method="euler",
        mass=None,
        imex_form=False,
        func2=None,
        batch_size=1,
        use_hpddm=False,
        matrixfree_hpddm=True,
    ):
        """
        Set up the PETSc ODE solver before it is used.

        Args:
            u_tensor: N-D Tensor giving meta information such as size, dtype and device, its values are not used.
            func: The callback function passed to PETSc TS
            step_size: Specifies the step size for the ODE solver. It can be a scalar or a list.
                       The list corresponds to the step size at each time step.
            enable_adjoint: If true, checkpointing will be used as required by the adjoint solver.
            implicit_form: Specifies the formulation type for func. PETSc TS can handle explicit ODEs in the form
                           ```
                           du/dt = fun(t, u), u(t[0]) = u0
                           ```
                           and implicit ODEs in the form
                           ```
                           M(u)du/dt = fun(t, u), u(t[0]) = u0
                           ```
                           where M(u) is the mass matrix.
            use_dlpack: DLPack allows in-palce conversion between PETSc objects and tensors. If disabled, numpy
                        arrays will be used as a stepping stone for the conversion.
            method: Specifies the time integration method for PETSc TS. The choice can be overwritten by command
                    line option -ts_type <petsc_ts_method_name>
            mass: Mass matrix for DAE, should be a tensor
            imex_form: Flag for using the IMEX form to define the ODE
                           ```
                           du/dt = func(t, u) + func2(t, u), u(t[0]) = u0
                           ```
                           where func will be treated implicitly and func2 explicitly.
            func2: An additional callback function. In the IMEX setting, this function is treated explicitly while func() is treated implicitly.In non-IMEX settings, this function is ignored.
            batch_size: The batch size. Needed only when imex_form=True. It is used to inform HPDDM about the block size.
            use_hpddm: Specifies whether to use MatSolve to solve a linear system with multiple RHS. If not, gmres will be used by default.
            matrixfree_hpddm: Specifies whether to use matrix-free solvers (for implicit time integration methods). If not, explicit Jacobians are assembled by using functorch's jacrev().
        """
        if imex_form and func2 is None:
            raise ValueError("func2 must be provided to enable imex_form=True")
        self.imex = imex_form
        self.use_hpddm = use_hpddm
        self.matrixfree_hpddm = matrixfree_hpddm
        tensor_dtype = u_tensor.dtype
        tensor_size = u_tensor.size()
        device = u_tensor.device
        n = u_tensor.numel()
        if self.funcIM != func or self.funcEX != func2:
            if imex_form:
                self.funcIM = func
                self.funcEX = func2
                self.flat_params = _flatten(
                    filter(lambda p: p.requires_grad, func.parameters())
                )
                self.npIM = self.flat_params.numel()
                self.flat_params = torch.cat(
                    (
                        self.flat_params,
                        _flatten(filter(lambda p: p.requires_grad, func2.parameters())),
                    )
                )
                self.np = self.flat_params.numel()
                self.npEX = self.np - self.npIM
            else:  # func2 is ignored
                self.funcIM = func
                self.funcEX = func
                self.flat_params = _flatten(
                    filter(lambda p: p.requires_grad, func.parameters())
                )
                self.np = self.npIM = self.npEX = self.flat_params.numel()
        if self.mass != mass:
            self.mass = mass
        # self.tensor_type = u_tensor.type()
        # self.cached_u_tensor = u_tensor.detach().clone()
        # check if the input tensor has a different type, device or size
        if (
            tensor_size != self.tensor_size
            or tensor_dtype != self.tensor_dtype
            or device != self.device
        ):
            self.tensor_size = tensor_size
            self.tensor_dtype = tensor_dtype
            self.device = device
            self.use_dlpack = use_dlpack
            self.n = n
            self.ts.reset()
            self.ts.setType(PETSc.TS.Type.RK)
            self.ts.setEquationType(PETSc.TS.EquationType.ODE_EXPLICIT)
            self.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
            if method == "euler":
                self.ts.setRKType("1fe")
            elif method == "rk2":  # 2a is Heun's method, not midpoint.
                self.ts.setRKType("2b")
            elif method == "fixed_bosh3":
                self.ts.setRKType("3bs")
            elif method == "rk4":
                self.ts.setRKType("4")
            elif method == "fixed_dopri5":
                self.ts.setRKType("5dp")
            elif method == "beuler":
                self.ts.setType(PETSc.TS.Type.BE)
            elif method == "cn":
                self.ts.setType(PETSc.TS.Type.CN)
            elif method == "imex":
                self.ts.setType(PETSc.TS.Type.ARKIMEX)
            if use_dlpack:
                # self.cached_U = PETSc.Vec().createWithDLPack(dlpack.to_dlpack(self.cached_u_tensor)) # convert to PETSc vec
                self.cached_U = PETSc.Vec().createWithDLPack(
                    u_tensor.detach().clone()
                )  # convert to PETSc vec, used only for providing info
            else:
                self.cached_U = PETSc.Vec().createWithArray(
                    u_tensor.detach().clone().cpu().numpy()
                )  # convert to PETSc vec
            if implicit_form or imex_form:
                self.fIM_tensor = u_tensor.detach().clone()
                if use_dlpack:
                    FIM = PETSc.Vec().createWithDLPack(self.fIM_tensor)
                else:
                    FIM = PETSc.Vec().createWithArray(self.fIM_tensor.cpu().numpy())
                self.ts.setIFunction(self.evalIFunction, FIM)
                IJac = PETSc.Mat().create()
                IJac.setSizes([self.n, self.n])
                IJac.setType("python")
                shell = IJacShell(self)
                IJac.setPythonContext(shell)
                IJac.setUp()
                IJac.assemble()
                self.ts.setIJacobian(self.evalIJacobian, IJac)
                self.use_cuda = True if device.type == "cuda" else False

                if self.use_hpddm:
                    from .petsc_hpddm_linearsolve import PCShell
                    # set up for HPDDM
                    if self.matrixfree_hpddm:
                        innerkspmat = PETSc.Mat().createPython(
                            [n // batch_size, n // batch_size], shell
                        )
                        self.innerkspmat = None
                    else:
                        innerkspmat = PETSc.Mat().createDense(
                            [n // batch_size, n // batch_size]
                        )
                        if self.use_cuda:
                            innerkspmat.setType("densecuda")
                        self.innerkspmat = innerkspmat
                    snes = self.ts.getSNES()
                    ksp = snes.getKSP()
                    pc = PETSc.PC()
                    pcshell = PCShell(
                        innerkspmat, batch_size, n // batch_size, self.use_cuda
                    )
                    pc.createPython(pcshell, PETSc.COMM_WORLD)
                    kmat, _ = ksp.getOperators()
                    if self.use_cuda:
                        # kmat.setVecType('cuda')
                        innerkspmat.setVecType("cuda")
                    pc.setOperators(kmat)
                    ksp.setType(PETSc.KSP.Type.PREONLY)
                    ksp.setPC(pc)
            if not implicit_form or imex_form:
                self.f_tensor = u_tensor.detach().clone()
                if use_dlpack:
                    F = PETSc.Vec().createWithDLPack(self.f_tensor)
                else:
                    F = PETSc.Vec().createWithArray(self.f_tensor.cpu().numpy())
                self.ts.setRHSFunction(self.evalRHSFunction, F)
                RHSJac = PETSc.Mat().create()
                RHSJac.setSizes([self.n, self.n])
                RHSJac.setType("python")
                shell = RHSJacShell(self)
                RHSJac.setPythonContext(shell)
                RHSJac.setUp()
                RHSJac.assemble()
                self.ts.setRHSJacobian(self.evalRHSJacobian, RHSJac)
            # check if it is already enabled or the func has changed
            # if enable_adjoint and not self.adj_u and not self.adj_p:
            if enable_adjoint:
                self.ts.adjointReset()
                if implicit_form or imex_form:
                    IJacp = PETSc.Mat().create()
                    IJacp.setSizes([self.n, self.np])
                    IJacp.setType("python")
                    shell = IJacPShell(self)
                    IJacp.setPythonContext(shell)
                    IJacp.setUp()
                    IJacp.assemble()
                    self.ts.setIJacobianP(self.evalIJacobianP, IJacp)
                if not implicit_form or imex_form:
                    RHSJacp = PETSc.Mat().create()
                    RHSJacp.setSizes([self.n, self.np])
                    RHSJacp.setType("python")
                    shell = RHSJacPShell(self)
                    RHSJacp.setPythonContext(shell)
                    RHSJacp.setUp()
                    RHSJacp.assemble()
                    self.ts.setRHSJacobianP(self.evalRHSJacobianP, RHSJacp)
                self.adj_u = []
                if self.use_dlpack:
                    self.adj_u_tensor = u_tensor.detach().clone()
                    self.adj_u.append(PETSc.Vec().createWithDLPack(self.adj_u_tensor))
                else:
                    self.adj_u.append(PETSc.Vec().createSeq(self.n, comm=self.comm))
                self.adj_p = []
                if self.use_dlpack:
                    self.adj_p_tensor = self.flat_params.detach().clone()
                    self.adj_p.append(PETSc.Vec().createWithDLPack(self.adj_p_tensor))
                else:
                    self.adj_p.append(PETSc.Vec().createSeq(self.np, comm=self.comm))
                # self.adj_p.append(torch.zeros_like(self.flat_params))
                self.ts.setCostGradients(self.adj_u, self.adj_p)
        # self.ts.setMaxSteps(1000)
        self.step_size = step_size
        if not isinstance(step_size, list):  # scalar
            self.ts.setTimeStep(step_size)  # overwrite the command-line option
        if enable_adjoint:
            self.ts.setSaveTrajectory()
        else:
            self.ts.removeTrajectory()
        self.ts.setFromOptions()

    def odeint(self, u0, t):
        """
        Solves a system of ODEs
            ```
            du/dt = fun(t, u), u(t[0]) = u0
            ```
        where u is a Tensor or tuple of Tensors of any shape.

        Args:
            u0: N-D Tensor giving the initial condition.
            t: 1-D Tensor specifying a sequence of time points.

        Returns
            solution: Tensor, where the frist dimension corresponds to the input time points.
        """
        if self.use_hpddm and not self.matrixfree_hpddm:
            self.reset_jacobianIM = True  # recompute the jacobian for functionIM since the parameters have been updated
        # self.u0 = u0.clone().detach() # clone a new tensor that will be used by PETSc
        if self.use_dlpack:
            self.u0 = (
                u0.detach().clone()
            )  # increase the object reference, otherwise the dlpack object may be deleted early and cause a bug
            U = PETSc.Vec().createWithDLPack(self.u0)  # convert to PETSc vec
        else:
            U = PETSc.Vec().createWithArray(
                u0.cpu().clone().numpy()
            )  # convert to PETSc vec
        ts = self.ts

        self.sol_times = t.cpu().to(dtype=torch.float64)
        if not isinstance(self.step_size, list):
            ts.setTimeStep(
                self.step_size
            )  # reset the step size because the last time step of TSSolve() may be changed even the fixed time step is used.
        else:
            ts.setTimeStep(self.step_size[0])
        if t.shape[0] == 1:
            ts.setTime(0.0)
            ts.setMaxTime(self.sol_times[0])
        else:
            ts.setTimeSpan(t.cpu().numpy())  # overwrite the command line option
            self.cur_sol_steps = [0] * list(t.size())[
                0
            ]  # time steps taken to integrate from previous saved solution to current solution
            self.cur_sol_index = 1  # skip 0th element
            ts.setPostStep(self.tspanPostStep)
        ts.setStepNumber(0)
        ts.solve(U)
        if t.shape[0] == 1:
            if self.use_dlpack:
                solution = torch.stack(
                    [dlpack.from_dlpack(U.toDLPack(mode="r")).clone()], dim=0
                )
            else:
                solution = torch.stack(
                    [
                        torch.from_numpy(U.array_r.reshape(self.tensor_size)).to(
                            device=self.device, dtype=self.tensor_dtype
                        )
                    ],
                    dim=0,
                )
        else:
            tspan_sols = ts.getTimeSpanSolutions()
            if self.use_dlpack:
                solution = torch.stack(
                    [
                        dlpack.from_dlpack(tspan_sols[i].toDLPack(mode="r")).view(
                            self.tensor_size
                        )
                        for i in range(len(tspan_sols))
                    ],
                    dim=0,
                )
            else:
                solution = torch.stack(
                    [
                        torch.from_numpy(
                            tspan_sols[i].array_r.reshape(self.tensor_size)
                        ).to(device=self.device, dtype=self.tensor_dtype)
                        for i in range(len(tspan_sols))
                    ],
                    dim=0,
                )
            self.ts.setPostStep(None)
            if self.cur_sol_index != len(self.sol_times):
                raise Exception("TSSolve fails to step on all the specified points")
        return solution

    def petsc_adjointsolve(self, t, i=1):
        ts = self.ts
        dt = ts.getTimeStep()
        if t.shape[0] == 1:
            ts.adjointSetSteps(round((t / dt).abs().item()))
        else:
            ts.adjointSetSteps(self.cur_sol_steps[i])
        ts.adjointSolve()
        adj_u, adj_p = ts.getCostGradients()
        if self.use_dlpack:
            adj_u_tensor = self.adj_u_tensor
            adj_p_tensor = self.adj_p_tensor
        else:
            adj_u_tensor = torch.from_numpy(
                adj_u[0].getArray().reshape(self.tensor_size)
            ).to(device=self.device, dtype=self.tensor_dtype)
            adj_p_tensor = torch.from_numpy(adj_p[0].getArray().reshape(self.np)).to(
                device=self.device, dtype=self.tensor_dtype
            )
        return adj_u_tensor, adj_p_tensor

    def odeint_adjoint(self, y0, t):
        # We need this in order to access the variables inside this module,
        # since we have no other way of getting variables along the execution path.

        if not isinstance(self.funcIM, nn.Module):
            raise ValueError("func is required to be an instance of nn.Module.")

        ys = OdeintAdjointMethod.apply(y0, t, self.flat_params, self)
        return ys


class OdeintAdjointMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, t, flat_params, ode, *args):
        """
        Solve the ODE forward in time
        """
        ctx.ode = ode
        with torch.no_grad():
            ans = ode.odeint(y0, t)
        ctx.save_for_backward(t, flat_params, ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Compute adjoint using PETSc TSAdjoint
        """
        t, flat_params, ans = ctx.saved_tensors
        T = ans.shape[0]
        with torch.no_grad():
            if ctx.ode.use_dlpack:
                ctx.ode.adj_u_tensor.copy_(grad_output[0][-1])
                ctx.ode.adj_p_tensor.zero_()
                if ctx.ode.use_cuda:
                    hdl = ctx.ode.adj_u[0].getCUDAHandle("w")
                    ctx.ode.adj_u[0].restoreCUDAHandle(hdl, "w")
                    hdl = ctx.ode.adj_p[0].getCUDAHandle("w")
                    ctx.ode.adj_p[0].restoreCUDAHandle(hdl, "w")
            else:
                ctx.ode.adj_u[0].setArray(grad_output[0][-1].cpu().numpy())
                ctx.ode.adj_p[0].zeroEntries()
            if T == 1:  # forward time interval is [0,t[0]] when t has a single element
                adj_u_tensor, adj_p_tensor = ctx.ode.petsc_adjointsolve(t)
            for i in range(T - 1, 0, -1):
                adj_u_tensor, adj_p_tensor = ctx.ode.petsc_adjointsolve(t, i)
                adj_u_tensor.add_(grad_output[0][i - 1])  # add forcing
                if (
                    not ctx.ode.use_dlpack
                ):  # if use_dlpack=True, adj_u_tensor shares memory with adj_u[0], so no need to set the values explicitly
                    ctx.ode.adj_u[0].setArray(
                        adj_u_tensor.cpu().numpy()
                    )  # update PETSc work vectors
            adj_u_tensor = adj_u_tensor.detach().clone()
            adj_p_tensor = adj_p_tensor.detach().clone()
        return (adj_u_tensor, None, adj_p_tensor, None)
