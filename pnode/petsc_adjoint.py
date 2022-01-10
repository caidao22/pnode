import torch
import torch.nn as nn
import torch.utils.dlpack as dlpack
from .misc import _flatten, _flatten_convert_none_to_zeros
import petsc4py
from petsc4py import PETSc

class RHSJacShell:
    def __init__(self, ode):
        self.ode_ = ode

    def mult(self, A, X, Y):
        """The Jacobian is A = shift*I - dFdU"""
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            x_tensor = dlpack.from_dlpack(X.toDLPack())
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y.toDLPack())
        else:
            x_tensor = torch.from_numpy(X.array.reshape(self.ode_.cached_u_tensor.size())).to(device=self.ode_.device,dtype=self.ode_.tensor_dtype)
            y = Y.array
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.func(self.ode_.t, self.ode_.cached_u_tensor)
            # grad_outputs = torch.zeros_like(func_eval, requires_grad=True)
            # vjp_u = torch.autograd.grad(
            #     func_eval, self.ode_.cached_u_tensor, grad_outputs,
            #     allow_unused=True, create_graph=True
            # )
            # jvp_u = torch.autograd.grad(
            #     vjp_u[0], grad_outputs, self.x_tensor,
            #     allow_unused=True
            # )
            vjp_u = torch.autograd.grad(
                func_eval, self.ode_.cached_u_tensor, x_tensor,
                allow_unused=True, create_graph=True
            )
            jvp_u = torch.autograd.grad(
                vjp_u[0], x_tensor, x_tensor,
                allow_unused=True
            )
            self.ode_.cached_u_tensor.requires_grad_(False)
        if jvp_u[0] is None: jvp_u[0] = torch.zeros_like(y)
        if self.ode_.use_dlpack:
            y.copy_(jvp_u[0])
        else:
            y[:] = jvp_u[0].cpu().numpy().flatten()

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            x_tensor = dlpack.from_dlpack(X.toDLPack())
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y.toDLPack())
        else:
            x_tensor = torch.from_numpy(X.array.reshape(self.ode_.cached_u_tensor.size())).to(device=self.ode_.device,dtype=self.ode_.tensor_dtype)
            y = Y.array
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.func(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_u = torch.autograd.grad(
               func_eval, self.ode_.cached_u_tensor,
               x_tensor, allow_unused=True, retain_graph=True
            )
        self.ode_.func_eval = func_eval
        # autograd.grad returns None if no gradient, set to zero.
        # vjp_u = tuple(torch.zeros_like(y_) if vjp_u_ is None else vjp_u_ for vjp_u_, y_ in zip(vjp_u, y))
        if vjp_u[0] is None: vjp_u[0] = torch.zeros_like(y)
        if self.ode_.use_dlpack:
            y.copy_(vjp_u[0])
        else:
            y[:] = vjp_u[0].cpu().numpy().flatten()

class IJacShell:
    def __init__(self, ode):
        self.ode_ = ode

    def mult(self, A, X, Y):
        """The Jacobian is A = shift*I - dFdU"""
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            self.x_tensor = dlpack.from_dlpack(X.toDLPack())
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y.toDLPack())
        else:
            self.x_tensor = torch.from_numpy(X.array.reshape(self.ode_.cached_u_tensor.size())).to(device=self.ode_.device,dtype=self.ode_.tensor_dtype)
            y = Y.array
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.func(self.ode_.t, self.ode_.cached_u_tensor)
            # grad_outputs = torch.zeros_like(func_eval, requires_grad=True)
            # vjp_u = torch.autograd.grad(
            #     func_eval, self.ode_.cached_u_tensor, grad_outputs,
            #     allow_unused=True, create_graph=True
            # )
            # jvp_u = torch.autograd.grad(
            #     vjp_u[0], grad_outputs, self.x_tensor,
            #     allow_unused=True
            # )
            self.x_tensor = self.x_tensor.detach().requires_grad_(True)
            vjp_u = torch.autograd.grad(
                func_eval, self.ode_.cached_u_tensor, self.x_tensor,
                allow_unused=True, create_graph=True
            )
            jvp_u = torch.autograd.grad(
                vjp_u[0], self.x_tensor, self.x_tensor,
                allow_unused=True
            )
        if jvp_u[0] is None: jvp_u[0] = torch.zeros_like(y)
        if self.ode_.use_dlpack:
            y.copy_(self.x_tensor.mul(self.ode_.shift)-jvp_u[0])
        else:
            y[:] = self.ode_.shift*X.array - jvp_u[0].cpu().numpy().flatten()

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            self.x_tensor = dlpack.from_dlpack(X.toDLPack())
            Y.attachDLPackInfo(self.ode_.cached_U)
            y = dlpack.from_dlpack(Y.toDLPack())
        else:
            self.x_tensor = torch.from_numpy(X.array.reshape(self.ode_.cached_u_tensor.size())).to(device=self.ode_.device,dtype=self.ode_.tensor_dtype)
            y = Y.array
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor.requires_grad_(True)
            func_eval = self.ode_.func(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_u = torch.autograd.grad(
               func_eval, self.ode_.cached_u_tensor,
               self.x_tensor, allow_unused=True, retain_graph=True
            )
        self.ode_.func_eval = func_eval
        # autograd.grad returns None if no gradient, set to zero.
        # vjp_u = tuple(torch.zeros_like(y_) if vjp_u_ is None else vjp_u_ for vjp_u_, y_ in zip(vjp_u, y))
        if vjp_u[0] is None: vjp_u[0] = torch.zeros_like(y)
        if self.ode_.use_dlpack:
            y.copy_(torch.mul(self.x_tensor,self.ode_.shift)-vjp_u[0])
        else:
            y[:] = self.ode_.shift*X.array - vjp_u[0].cpu().numpy().flatten()

class JacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            X.attachDLPackInfo(self.ode_.cached_U)
            x_tensor = dlpack.from_dlpack(X.toDLPack())
            Y.attachDLPackInfo(self.ode_.adj_p[0])
            y = dlpack.from_dlpack(Y.toDLPack())
        else:
            x_tensor = torch.from_numpy(X.array.reshape(self.ode_.cached_u_tensor.size())).to(device=self.ode_.device,dtype=self.ode_.tensor_dtype)
            y = Y.array
        f_params = tuple(self.ode_.func.parameters())
        with torch.set_grad_enabled(True):
            # t = t.to(self.u_tensor.device).detach().requires_grad_(False)
            if self.ode_.func_eval is not None:
                func_eval = self.ode_.func_eval
            else:
                func_eval = self.ode_.func(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_params = torch.autograd.grad(
                func_eval, f_params,
                x_tensor, allow_unused=True
            )
        # autograd.grad returns None if no gradient, set to zero.
        vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)
        if self.ode_.func_eval is not None:
            self.ode_.func_eval = None
        if self.ode_.use_dlpack:
            if self.ode_.ijacp:
                y.copy_(torch.mul(vjp_params,-1.0))
            else:
                y.copy_(vjp_params)
        else:
            if self.ode_.ijacp:
                y[:] = -vjp_params.cpu().numpy().flatten()
            else:
                y[:] = vjp_params.cpu().numpy().flatten()

class ODEPetsc(object):
    comm = PETSc.COMM_SELF

    def __init__(self):
        self.ts = PETSc.TS().create(comm=self.comm)
        self.func_eval = None

    def evalFunction(self, ts, t, U, F):
        if self.use_dlpack:
            # have to call to() or type() to avoid a PETSc seg fault
            U.attachDLPackInfo(self.cached_U)
            u_tensor = dlpack.from_dlpack(U.toDLPack())
            # u_tensor = dlpack.from_dlpack(U.toDLPack()).view(self.cached_u_tensor.size()).type(self.tensor_type)
            F.attachDLPackInfo(self.cached_U)
            dudt = dlpack.from_dlpack(F.toDLPack())
            # Resotring the handle set the offloadmask flag to PETSC_OFFLOAD_GPU, but it zeros out the GPU memory accidentally, which is probably a bug
            if torch.cuda.is_initialized():
                hdl = F.getCUDAHandle('w')
                F.restoreCUDAHandle(hdl,'w')
            dudt.copy_(self.func(t, u_tensor))
        else:
            f = F.array
            u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype)
            dudt = self.func(t, u_tensor).cpu().detach().numpy()
            f[:] = dudt.flatten()

    def evalIFunction(self, ts, t, U, Udot, F):
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            u_tensor = dlpack.from_dlpack(U.toDLPack())
            Udot.attachDLPackInfo(self.cached_U)
            udot_tensor = dlpack.from_dlpack(Udot.toDLPack())
            # Resotring the handle set the offloadmask flag to PETSC_OFFLOAD_GPU, but it zeros out the GPU memory accidentally, which is probably a bug
            if torch.cuda.is_initialized():
                hdl = F.getCUDAHandle('w')
                F.restoreCUDAHandle(hdl,'w')
            F.attachDLPackInfo(self.cached_U)
            dudt = dlpack.from_dlpack(F.toDLPack())
            dudt.copy_(udot_tensor-self.func(t, u_tensor))
        else:
            f = F.array
            u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype)
            dudt = self.func(t, u_tensor).cpu().detach().numpy()
            f[:] = Udot.array - dudt.flatten()

    def evalJacobian(self, ts, t, U, Jac, JacPre):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            # x = dlpack.from_dlpack(U.toDLPack())
            # self.cached_u_tensor.copy_(x)
            self.cached_u_tensor = dlpack.from_dlpack(U.toDLPack())
        else:
            self.cached_u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype)

    def evalIJacobian(self, ts, t, U, Udot, shift, Jac, JacPre):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        self.shift = shift
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            x = dlpack.from_dlpack(U.toDLPack())
            self.cached_u_tensor.copy_(x)
            # self.cached_u_tensor = dlpack.from_dlpack(U.toDLPack()).view(self.cached_u_tensor.size()).type(self.tensor_type)
        else:
            self.cached_u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype)

    def evalJacobianP(self, ts, t, U, Jacp):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            x = dlpack.from_dlpack(U.toDLPack())
            self.cached_u_tensor.copy_(x)
            # self.cached_u_tensor = dlpack.from_dlpack(U.toDLPack()).view(self.cached_u_tensor.size()).type(self.tensor_type)
        else:
            self.cached_u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype)

    def evalIJacobianP(self, ts, t, U, Udot, shift, Jacp):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        if self.use_dlpack:
            U.attachDLPackInfo(self.cached_U)
            x = dlpack.from_dlpack(U.toDLPack())
            self.cached_u_tensor.copy_(x)
            # self.cached_u_tensor = dlpack.from_dlpack(U.toDLPack()).view(self.cached_u_tensor.size()).type(self.tensor_type)
        else:
            self.cached_u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype)

    def saveSolution(self, ts, stepno, t, U):
        """"Save the solutions at intermediate points"""
        if self.cur_index < len(self.sol_list) and self.sol_list[self.cur_index] is None:
            if self.tensor_dtype == torch.double:
                delta = 1e-5
            else:
                delta = 1e-3
            if abs(t-self.sol_times[self.cur_index]) < delta: # ugly workaround
                if self.use_dlpack:
                    unew = dlpack.from_dlpack(U.toDLPack()).clone()
                else:
                    unew = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype,copy=True)
                self.sol_list[self.cur_index] = unew
                self.cur_index = self.cur_index+1

    def setupTS(self, u_tensor, func, step_size=0.01, enable_adjoint=True, implicit_form=False, use_dlpack=True, method='euler'):
        self.device = u_tensor.device
        self.tensor_dtype = u_tensor.dtype
        #self.tensor_type = u_tensor.type()
        #self.cached_u_tensor = u_tensor.detach().clone()
        self.n = u_tensor.numel()
        self.use_dlpack = use_dlpack
        if use_dlpack:
            # self.cached_U = PETSc.Vec().createWithDLPack(dlpack.to_dlpack(self.cached_u_tensor)) # convert to PETSc vec
            self.cached_U = PETSc.Vec().createWithDLPack(dlpack.to_dlpack(u_tensor)) # convert to PETSc vec, used only for providing info
        else:
            self.cached_U = PETSc.Vec().createWithArray(u_tensor.cpu().numpy()) # convert to PETSc vec

        self.func = func
        self.step_size = step_size
        self.flat_params = _flatten(func.parameters())
        self.np = self.flat_params.numel()

        self.ts.reset()
        self.ts.setType(PETSc.TS.Type.RK)
        self.ts.setEquationType(PETSc.TS.EquationType.ODE_EXPLICIT)
        self.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        if method=='euler':
            self.ts.setRKType('1fe')
        elif method == 'midpoint':  # 2a is Heun's method, not midpoint.
            self.ts.setRKType('2a')
        elif method == 'rk4':
            self.ts.setRKType('4')
        elif method == 'dopri5_fixed':
            self.ts.setRKType('5dp')
        elif method == 'beuler':
            self.ts.setType(PETSc.TS.Type.BE)
        elif method == 'cn':
            self.ts.setType(PETSc.TS.Type.CN)

        self.f_tensor = u_tensor.detach().clone()
        if use_dlpack:
            F = PETSc.Vec().createWithDLPack(dlpack.to_dlpack(self.f_tensor))
        else:
            F = PETSc.Vec().createWithArray(self.f_tensor.cpu().numpy())

        if implicit_form :
            self.ts.setIFunction(self.evalIFunction, F)
        else :
            self.ts.setRHSFunction(self.evalFunction, F)

        Jac = PETSc.Mat().create()
        Jac.setSizes([self.n, self.n])
        Jac.setType('python')
        if implicit_form :
            shell = IJacShell(self)
        else :
            shell = RHSJacShell(self)
        Jac.setPythonContext(shell)
        Jac.setUp()
        Jac.assemble()
        if implicit_form :
            self.ts.setIJacobian(self.evalIJacobian, Jac)
        else :
            self.ts.setRHSJacobian(self.evalJacobian, Jac)

        if enable_adjoint :
            Jacp = PETSc.Mat().create()
            Jacp.setSizes([self.n, self.np])
            Jacp.setType('python')
            shell = JacPShell(self)
            Jacp.setPythonContext(shell)
            Jacp.setUp()
            Jacp.assemble()
            if implicit_form :
                self.ijacp = True
                self.ts.setIJacobianP(self.evalIJacobianP, Jacp)
            else :
                self.ijacp = False
                self.ts.setRHSJacobianP(self.evalJacobianP, Jacp)

            self.adj_u = []

            if self.use_dlpack:
                self.adj_u_tensor = u_tensor.detach().clone()
                self.adj_u.append(PETSc.Vec().createWithDLPack(dlpack.to_dlpack(self.adj_u_tensor)))
            else:
                self.adj_u.append(PETSc.Vec().createSeq(self.n, comm=self.comm))
            self.adj_p = []
            if self.use_dlpack:
                self.adj_p_tensor = self.flat_params.detach().clone()
                self.adj_p.append(PETSc.Vec().createWithDLPack(dlpack.to_dlpack(self.adj_p_tensor)))
            else:
                self.adj_p.append(PETSc.Vec().createSeq(self.np, comm=self.comm))
            # self.adj_p.append(torch.zeros_like(self.flat_params))
            self.ts.setCostGradients(self.adj_u, self.adj_p)
            self.ts.setSaveTrajectory()

        # self.ts.setMaxSteps(1000)
        self.ts.setFromOptions()
        self.ts.setTimeStep(step_size) # overwrite the command-line option

    def odeint(self, u0, t):
        """Return the solutions in tensor"""
        # self.u0 = u0.clone().detach() # clone a new tensor that will be used by PETSc
        if self.use_dlpack:
            self.u0 = u0.detach().clone() # increase the object reference, otherwise the dlpack object may be deleted early and cause a bug
            U = PETSc.Vec().createWithDLPack(dlpack.to_dlpack(self.u0)) # convert to PETSc vec
        else:
            U = PETSc.Vec().createWithArray(u0.cpu().clone().numpy()) # convert to PETSc vec
        ts = self.ts

        self.sol_times = t.cpu().to(dtype=torch.float64)
        self.cur_index = 0
        if t.shape[0] == 1:
            ts.setTime(0.0)
            ts.setMaxTime(self.sol_times[0])
        else:
            ts.setTime(self.sol_times[0])
            ts.setMaxTime(self.sol_times[-1])
            self.sol_list =  [None]*list(t.size())[0]
            self.ts.setMonitor(self.saveSolution)
        ts.setStepNumber(0)
        ts.setTimeStep(self.step_size) # reset the step size because the last time step of TSSolve() may be changed even the fixed time step is used.
        ts.solve(U)
        if t.shape[0] == 1:
            solution = torch.stack([dlpack.from_dlpack(U.toDLPack()).clone()], dim=0)
        else:
            solution = torch.stack([self.sol_list[i] for i in range(len(self.sol_times))], dim=0)
            ts.cancelMonitor()
        return solution

    def petsc_adjointsolve(self, t):
        t = t.to(self.device, torch.float64)
        ts = self.ts
        dt = ts.getTimeStep()
        # print('do {} adjoint steps'.format(round(((t[1]-t[0])/dt).abs().item())))
        ts.adjointSetSteps(round(((t[1]-t[0])/dt).abs().item()))
        ts.adjointSolve()
        adj_u, adj_p = ts.getCostGradients()
        if self.use_dlpack:
            adj_u_tensor = self.adj_u_tensor
            adj_p_tensor = self.adj_p_tensor
        else:
            adj_u_tensor = torch.from_numpy(adj_u[0].getArray().reshape(self.cached_u_tensor.size())).to(device=self.device,dtype=self.tensor_dtype)
            adj_p_tensor = torch.from_numpy(adj_p[0].getArray().reshape(self.np)).to(device=self.device,dtype=self.tensor_dtype)
        return adj_u_tensor, adj_p_tensor

    def odeint_adjoint(self, y0, t):
        # We need this in order to access the variables inside this module,
        # since we have no other way of getting variables along the execution path.

        if not isinstance(self.func, nn.Module):
            raise ValueError('func is required to be an instance of nn.Module.')

        ys = OdeintAdjointMethod.apply(y0,t,self.flat_params,self)
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
                if torch.cuda.is_initialized():
                    hdl = ctx.ode.adj_u[0].getCUDAHandle('w')
                    ctx.ode.adj_u[0].restoreCUDAHandle(hdl,'w')
                    hdl = ctx.ode.adj_p[0].getCUDAHandle('w')
                    ctx.ode.adj_p[0].restoreCUDAHandle(hdl,'w')
            else:
                ctx.ode.adj_u[0].setArray(grad_output[0][-1].cpu().numpy())
                ctx.ode.adj_p[0].zeroEntries()
            if T == 1: # forward time interval is [0,t[0]] when t has a single element
                adj_u_tensor, adj_p_tensor = ctx.ode.petsc_adjointsolve(torch.tensor([t, 0.0]))
            for i in range(T-1, 0, -1):
                adj_u_tensor, adj_p_tensor = ctx.ode.petsc_adjointsolve(torch.tensor([t[i], t[i-1]]))
                adj_u_tensor.add_(grad_output[0][i-1]) # add forcing
                if not ctx.ode.use_dlpack: # if use_dlpack=True, adj_u_tensor shares memory with adj_u[0], so no need to set the values explicitly
                    ctx.ode.adj_u[0].setArray(adj_u_tensor.cpu().numpy()) # update PETSc work vectors
            adj_u_tensor = adj_u_tensor.detach().clone()
            adj_p_tensor = adj_p_tensor.detach().clone()
        return (adj_u_tensor, None, adj_p_tensor, None)
