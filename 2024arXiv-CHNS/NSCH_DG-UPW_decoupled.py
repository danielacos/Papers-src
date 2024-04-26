#
# NSCH Test with decoupled DG-UPW
# =============================
#

import dolfinx
from dolfinx import fem
from dolfinx.fem import (
    Expression, Function, FunctionSpace, VectorFunctionSpace, Constant,
    assemble_scalar, form, petsc
)
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import log
from ufl import(
     TestFunction, TrialFunction, FiniteElement, VectorElement, MixedElement,
     SpatialCoordinate,
     dx, dS, ds, inner, dot, grad, div, nabla_grad, sym,
     avg, jump,
     tanh, sqrt, sign, exp, cos,
     split,
     derivative,
     FacetArea, FacetNormal 
)
from dolfinx.io import (
    XDMFFile
)
import dolfinx.plot as plot
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def printMPI(string, end='\n'):
    if rank == 0:
        print(string, end=end)

DOLFIN_EPS = 1e-16
dx_ML = dx(scheme="vertex", metadata={"degree":1, "representation":"quadrature"}) # mass lumped terms

#
# Problem class
#
class NSCH_DG_UPW(object):
    r"""
    DG numerical solution of Navier-Stokes-Cahn-Hilliard equation
    with Neumann homogeneous conditions
    """

    def __init__(self, NSCH_parameters):
        #
        # Load PDE and discretization parameters
        #
        NSCH = self
        params = NSCH.parameters = NSCH_parameters

        NSCH.eps = float(params.eps)
        NSCH.lamb = float(params.lamb)
        NSCH.rho1 = float(params.rho1)
        NSCH.rho2 = float(params.rho2)
        NSCH.rho_avg = (NSCH.rho1 + NSCH.rho2)/2
        NSCH.rho_dif = (NSCH.rho2 - NSCH.rho1)/2
        NSCH.eta_val = float(params.eta)

        NSCH.sigma = float(params.sigma)

        file_path = os.path.dirname(os.path.abspath(__file__))
        printMPI(f"file_path = {file_path}")
        mesh_file = f"{file_path}/meshes/" + f"mesh_{params.mesh}_nx-{params.nx}.xdmf"
        printMPI(f"mesh_file = {mesh_file}")

        #
        # Read mesh
        #
        with XDMFFile(comm, mesh_file, 'r') as infile:
            mesh = NSCH.mesh = infile.read_mesh()
        
        NSCH.nx = int(params.nx)
        NSCH.dt = float(params.dt)
        NSCH.t = 0.

        NSCH.e_len = FacetArea(NSCH.mesh)
        NSCH.n_e = FacetNormal(NSCH.mesh)

        #
        # Build DG, FE spaces and functions
        #
        NSCH.P0d = FiniteElement("DG", mesh.ufl_cell(), 0)
        NSCH.P1d = FiniteElement("DG", mesh.ufl_cell(), 1)
        NSCH.P2bv = VectorElement(FiniteElement("Lagrange", mesh.ufl_cell(), 2) + FiniteElement("Bubble", mesh.ufl_cell(), 3))
        NSCH.P1c = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        NSCH.P1cu = VectorElement(FiniteElement("Lagrange", mesh.ufl_cell(), 1))


        NSCH.P0ds = FunctionSpace(mesh, NSCH.P0d)
        NSCH.P1ds = FunctionSpace(mesh, NSCH.P1d)
        NSCH.P2bvs = FunctionSpace(mesh, NSCH.P2bv)
        NSCH.P1cus = FunctionSpace(mesh, NSCH.P1cu)
        NSCH.Wh = FunctionSpace(mesh, MixedElement([NSCH.P0d, NSCH.P1c, NSCH.P1c]))

        NSCH.solvector_phi, NSCH.testvector_phi = Function(NSCH.Wh), TestFunction(NSCH.Wh)
        NSCH.solvector_phi0 = Function(NSCH.Wh)

        NSCH.phi, NSCH.mu, NSCH.p1c_phi = split(NSCH.solvector_phi)
        NSCH.phib, NSCH.mub, NSCH.p1c_phib = split(NSCH.testvector_phi)
        NSCH.phi0, NSCH.mu0, NSCH.p1c_phi0 = split(NSCH.solvector_phi0)

        NSCH.v, NSCH.v_trial, NSCH.vb, NSCH.v0 = Function(NSCH.P2bvs), TrialFunction(NSCH.P2bvs), TestFunction(NSCH.P2bvs), Function(NSCH.P2bvs)

        NSCH.tau, NSCH.tau_trial, NSCH.taub, NSCH.tau0 = Function(NSCH.P1ds), TrialFunction(NSCH.P1ds), TestFunction(NSCH.P1ds), Function(NSCH.P1ds)

        NSCH.p, NSCH.p_trial, NSCH.pb, NSCH.p0 = Function(NSCH.P1ds), TrialFunction(NSCH.P1ds), TestFunction(NSCH.P1ds), Function(NSCH.P1ds)

        # Compute subspaces and maps from subspaces to main space in MixedElement space
        NSCH.num_subs = NSCH.Wh.num_sub_spaces
        NSCH.spaces = []
        NSCH.maps = []
        for i in range(NSCH.num_subs):
            space_i, map_i = NSCH.Wh.sub(i).collapse()
            NSCH.spaces.append(space_i)
            NSCH.maps.append(map_i)

        # Domain size
        aux = Function(NSCH.P1ds)
        aux.x.array[:] = 1.0
        NSCH.domain_size = assemble_scalar(form(aux * dx))

    def project(self, u, space, mass_lumping=False):
        NSCH = self

        Piu_trial = TrialFunction(space)
        Piub = TestFunction(space)

        if mass_lumping:
            a = inner(Piu_trial, Piub) * dx_ML
        else:
            a = inner(Piu_trial, Piub) * dx

        L = inner(u, Piub) * dx

        problem = LinearProblem(a, L)
        return problem.solve()
    
    def rho(self, phi):
        NSCH = self
        "Density of the mixture"
        return (NSCH.rho_avg + NSCH.rho_dif * phi)

    def eta(self, phi):
            return NSCH.eta_val
    
    def pos(self, phi):
        return ((abs(phi) + phi) / 2.0)
    
    def neg(self, phi):
        return ((abs(phi) - phi) / 2.0)

    def M(self, phi):
            """Mobility function"""
            return (1 - phi**2)
    
    def Mpos(self, phi):
        """Positive part of mobility function"""
        NSCH = self
        return NSCH.pos(NSCH.M(phi))

    def load_initial_values(self):
        """Initialize variables"""
        NSCH = self
        eps = NSCH.eps
        lamb = NSCH.lamb

        #
        # Initial condition
        #
        x = SpatialCoordinate(NSCH.mesh)

        if NSCH.parameters.test == "bubble":
            phi_init = Expression(tanh((0.2 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * eps)), NSCH.spaces[0].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                return vals
        elif NSCH.parameters.test == "rayleigh":
            phi_init = Expression(tanh((x[1] - (0.1 * exp(-(x[0]+0.2)**2/0.1)))/(sqrt(2.0) * eps)), NSCH.spaces[0].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                return vals
        elif NSCH.parameters.test == "circle":
            phi_init = Expression(2 * tanh((NSCH.pos(0.25 - sqrt(pow(x[0]-0.1, 2) + pow(x[1]-0.1, 2))) + NSCH.pos(0.15 - sqrt(pow(x[0]+0.15, 2) + pow(x[1]+0.15, 2))))/(sqrt(2.0) * eps)) - 1.0, NSCH.spaces[0].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                vals[0] = 100 * x[1] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                vals[1] = -100 * x[0] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                return vals
        elif NSCH.parameters.test == "order":
            phi_init = Expression(2 * tanh((NSCH.pos(0.25 - sqrt(pow(x[0]-0.1, 2) + pow(x[1]-0.1, 2))) + NSCH.pos(0.15 - sqrt(pow(x[0]+0.15, 2) + pow(x[1]+0.15, 2))))/(sqrt(2.0) * eps)) - 1.0, NSCH.spaces[0].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                vals[0] = x[1] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                vals[1] = -x[0] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                return vals

        # Initial values
        NSCH.solvector_phi0.sub(0).interpolate(phi_init)
        NSCH.solvector_phi0.x.scatter_forward()
        phi0 = NSCH.phi0 = NSCH.solvector_phi0.sub(0)

        NSCH.solvector_phi0.x.array[NSCH.maps[2]] = NSCH.project(phi0, NSCH.spaces[2], mass_lumping=True).x.array
        NSCH.solvector_phi0.x.scatter_forward()
        p1c_phi0 = NSCH.p1c_phi0 = NSCH.solvector_phi0.sub(2)

        dF_phi0 = lambda phi0: (phi0**2 - 1) * phi0
        NSCH.solvector_phi0.x.array[NSCH.maps[1]] = NSCH.project(lamb/eps * dF_phi0(phi0) - lamb * eps * div(grad(phi0)), NSCH.spaces[1]).x.array
        NSCH.solvector_phi0.x.scatter_forward()
        mu0 = NSCH.mu0 = NSCH.solvector_phi0.sub(1)

        NSCH.v0.interpolate(u_init)
        NSCH.v0.x.scatter_forward()
        v0 = NSCH.v0

        NSCH.tau0.x.array[:] = 0.0
        NSCH.tau0.x.scatter_forward()
        tau0 = NSCH.tau0

        NSCH.p0.x.array[:] = 0.0
        NSCH.p0.x.scatter_forward()
        p0 = NSCH.p0

    def variational_problem_v(self):
        """Build variational problem"""
        #
        # Load variables from NSCH problem
        #
        NSCH = self
        params = NSCH.parameters
        dt = NSCH.dt
        nx = NSCH.nx

        eps = NSCH.eps
        lamb = NSCH.lamb
        rho1 = NSCH.rho1
        rho2 = NSCH.rho2
        rho_avg = NSCH.rho_avg
        rho_dif = NSCH.rho_dif

        v_trial, vb, v0 = NSCH.v_trial, NSCH.vb, NSCH.v0
        tau0, p0 = NSCH.tau0, NSCH.p0
        phi0, mu0 = NSCH.phi0, NSCH.mu0

        rho = NSCH.rho
        Mpos = NSCH.Mpos

        n_e = NSCH.n_e
        e_len = NSCH.e_len

        eta = NSCH.eta

        NSCH.phiold = Function(NSCH.P0ds)
        NSCH.phiold.x.array[:] = 0.0
        u0 = NSCH.u0 = NSCH.v0 - NSCH.dt/NSCH.rho(NSCH.phiold) * grad(NSCH.tau0)
        u = NSCH.u = NSCH.v - NSCH.dt/NSCH.rho(NSCH.phi0) * grad(NSCH.tau)

        NSCH.a_v = inner(rho(phi0) * v_trial, vb) * dx \
                + dt * inner(2 * eta(phi0)*sym(grad(v_trial)), sym(grad(vb))) * dx \
                + dt * inner(dot(rho(phi0) * v0 - rho_dif * Mpos(phi0) * grad(mu0), nabla_grad(v_trial)), vb) * dx
        NSCH.L_v = inner(rho(phi0) * u0, vb) * dx \
                    + dt * inner(p0, div(vb)) * dx \
                    - dt * inner(grad(mu0) * phi0, vb) * dx

        if params.dim == '3d':
            raise(Exception("Change how to deal with boundary conditions in 3D"))
        else:
            if NSCH.parameters.test == "rayleigh_std":
                def no_slip(x):
                    return (np.logical_or(np.isclose(x[0], 0), np.logical_or(np.isclose(x[0], 1), np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 4)))))
            else:
                def no_slip(x):
                    return (np.logical_or(np.isclose(np.abs(x[0]), 0.5), np.isclose(np.abs(x[1]), 0.5)))
            v_bc = Function(NSCH.P2bvs)
            v_bc.x.set(0)
            facets = dolfinx.mesh.locate_entities_boundary(NSCH.mesh, 1, no_slip)
            dofs = dolfinx.fem.locate_dofs_topological(NSCH.P2bvs, 1, facets)
            NSCH.bcv = bcv = dolfinx.fem.dirichletbc(
                v_bc, dofs)

        if int(params.gravity) and params.dim=='2d':
            NSCH.L_v += - rho(phi0) *  inner(Constant(NSCH.mesh, PETSc.ScalarType((0.0, 1))), vb) * dx
        elif int(params.gravity) and params.dim=='3d':
            NSCH.L_v += - rho(phi0) *  inner(Constant(NSCH.mesh, PETSc.ScalarType((0.0, 0.0, 1))), vb) * dx
        
    def variational_problem_tau(self):
        """Build variational problem"""
        #
        # Load variables from NSCH problem
        #
        NSCH = self
        params = NSCH.parameters
        dt = NSCH.dt
        nx = NSCH.nx

        eps = NSCH.eps
        lamb = NSCH.lamb
        rho1 = NSCH.rho1
        rho2 = NSCH.rho2
        rho_avg = NSCH.rho_avg
        rho_dif = NSCH.rho_dif

        tau_trial, taub, tau0 = NSCH.tau_trial, NSCH.taub, NSCH.tau0
        phi0, v = NSCH.phi0, NSCH.v

        u0 = NSCH.u0

        rho = NSCH.rho
        Mpos = NSCH.Mpos

        n_e = NSCH.n_e
        e_len = NSCH.e_len
        sigma = NSCH.sigma

        def aSIP(k_pen, tau, taub):
            k = 1.0/k_pen
            return(
                k * inner(grad(tau), grad(taub)) * dx \
                - inner(avg(k * grad(tau)), n_e('+')) * jump(taub) * dS \
                - inner(avg(k * grad(taub)), n_e('+')) * jump(tau) * dS
                + sigma/e_len * jump(tau) * jump(taub) * dS
            )

        NSCH.a_tau = aSIP(rho(phi0), tau_trial, taub)
        NSCH.L_tau = - 1.0/dt * inner(div(v), taub) * dx

    def variational_problem_p(self):
        """Build variational problem"""
        #
        # Load variables from NSCH problem
        #
        NSCH = self
        params = NSCH.parameters
        dt = NSCH.dt
        nx = NSCH.nx

        eps = NSCH.eps
        lamb = NSCH.lamb
        rho1 = NSCH.rho1
        rho2 = NSCH.rho2
        rho_avg = NSCH.rho_avg
        rho_dif = NSCH.rho_dif

        p_trial, pb, p0 = NSCH.p_trial, NSCH.pb, NSCH.p0
        phi0, v, tau = NSCH.phi0, NSCH.v, NSCH.tau

        u = NSCH.u

        rho = NSCH.rho
        Mpos = NSCH.Mpos

        n_e = NSCH.n_e
        e_len = NSCH.e_len

        eta = NSCH.eta

        NSCH.a_p = inner(p_trial, pb) * dx 
        NSCH.L_p = inner(p0, pb) * dx \
                    + inner(tau, pb) * dx \
                    - 2 * inner(eta(phi0)*div(v), pb) * dx

    def variational_problem_phi(self):
        """Build variational problem"""
        #
        # Load variables from NSCH problem
        #
        NSCH = self
        params = NSCH.parameters
        dt = NSCH.dt
        nx = NSCH.nx

        eps = NSCH.eps
        lamb = NSCH.lamb
        rho1 = NSCH.rho1
        rho2 = NSCH.rho2
        rho_avg = NSCH.rho_avg
        rho_dif = NSCH.rho_dif

        phi, mu, p1c_phi = NSCH.phi, NSCH.mu, NSCH.p1c_phi
        phib, mub, p1c_phib = NSCH.phib, NSCH.mub, NSCH.p1c_phib
        phi0, mu0, p1c_phi0 = NSCH.phi0, NSCH.mu0, NSCH.p1c_phi0
        u, tau = NSCH.u, NSCH.tau

        pos = NSCH.pos
        neg = NSCH.neg

        M = NSCH.M
        Mpos = NSCH.Mpos

        def Mup(phi):
            """Increasing part of Mpos"""
            return Mpos(1.0 / 2.0 * (phi - abs(phi)))
        def Mdown(phi):
            """Decreasing part of Mpos"""
            return (Mpos(1.0 / 2.0 * (phi + abs(phi))) - Mpos(0))
        
        def f(p1c_phi, p1c_phi0):
            return (2 * p1c_phi + p1c_phi0**3 - 3 * p1c_phi0)
        
        rho = NSCH.rho
        
        eta = NSCH.eta

        #
        # Variational problem
        #
        e_len = NSCH.e_len
        n_e = NSCH.n_e
        l = 1.0/nx
        sigma = NSCH.sigma

        def aupw(u, phi, phib):
            # UPW bilinear form
            k_pen = rho(phi0)
            return (
                pos(inner(avg(u), n_e('+')) + dt * sigma/e_len * jump(tau)) * phi('+') * jump(phib) * dS \
                - neg(inner(avg(u), n_e('+')) + dt * sigma/e_len * jump(tau)) * phi('-') * jump(phib) * dS
            )
        
        def bupw(mu, phi, phib):
            # UPW bilinear form
            return (
                pos(inner(avg(grad(mu)), n_e('+'))) * pos(Mup(phi('+')) + Mdown(phi('-'))) * jump(phib) * dS \
                - neg(inner(avg(grad(mu)), n_e('+'))) * pos(Mup(phi('-')) + Mdown(phi('+'))) * jump(phib) * dS
            )

        NSCH.a_phi = a_phi = inner(phi, phib) * dx \
                - inner(phi0, phib) * dx \
                + dt * bupw(-mu, phi, phib) \
                + dt * aupw(u, phi, phib) \
        
        NSCH.a_mu = a_mu = lamb * eps * inner(grad(p1c_phi), grad(mub)) * dx \
                + lamb/eps * inner(f(p1c_phi, p1c_phi0), mub) * dx \
                - inner(mu, mub) * dx
        
        NSCH.a_p1c_phi = a_p1c_phi = inner(phi, p1c_phib) * dx \
                    - inner(p1c_phi, p1c_phib) * dx_ML

        NSCH.F_phi = a_phi + a_mu + a_p1c_phi

    def create_systems(self):
        NSCH = self
        params = NSCH.parameters

        #
        # Initialization
        #
        NSCH.solvector_phi.x.array[:] = NSCH.solvector_phi0.x.array
        NSCH.solvector_phi.x.scatter_forward()

        #
        # Nullspace
        #
        nsp = PETSc.NullSpace().create(constant=True)

        #
        # Define problem
        #
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
        problem_v = NSCH.problem_v = LinearProblem(NSCH.a_v, NSCH.L_v, bcs=[NSCH.bcv], petsc_options=petsc_options)
        problem_tau = NSCH.problem_tau = LinearProblem(NSCH.a_tau, NSCH.L_tau)
        problem_tau.A.setNullSpace(nsp)
        problem_p = NSCH.problem_p = LinearProblem(NSCH.a_p, NSCH.L_p)
        problem_phi = NSCH.problem_phi = NonlinearProblem(NSCH.F_phi, NSCH.solvector_phi)

        solver_phi = NSCH.solver_phi = NewtonSolver(MPI.COMM_WORLD, problem_phi)
        solver_phi.convergence_criterion = "incremental"
        solver_phi.rtol = 1e-6
        solver_phi.max_it = 100
        # solver.relaxation_parameter = 1e-3
    
    def time_iterator(self, tsteps=1, first_step=1, verbosity=0):
        """Time iterator"""
        #
        # Load variables from NSCH problem
        #
        NSCH = self
        params = NSCH.parameters

        problem_v = NSCH.problem_v
        problem_tau = NSCH.problem_tau
        problem_p = NSCH.problem_p
        solver_phi = NSCH.solver_phi    

        #
        # Run time iterations
        #
        step = first_step - 1
        last_step = first_step + tsteps

        #
        # Load variables from NSCH problem
        #
        NSCH = self
        params = NSCH.parameters

        while step < last_step:
            if step == first_step - 1:
                phi0, mu0, p1c_phi0 = NSCH.solvector_phi0.split()
                v0, tau0, p0, u0 = NSCH.v0, NSCH.tau0, NSCH.p0, NSCH.u0

                # --- Yield initial data
                yield {'step': step, 't': NSCH.t, 'u':u0, 'p':p0, 'phi': phi0.collapse(), 'mu':mu0.collapse(), 'p1c_phi':p1c_phi0.collapse(), 'v':v0, 'tau':tau0}

            else:
                NSCH.t += NSCH.dt

                # Solve

                NSCH.v.x.array[:] = problem_v.solve().x.array
                NSCH.v.x.scatter_forward()

                NSCH.tau.x.array[:] = problem_tau.solve().x.array
                NSCH.tau.x.scatter_forward()
                # Correct the potential
                NSCH.tau.x.array[:] = NSCH.tau.x.array[:] - 1.0/NSCH.domain_size * assemble_scalar(form(NSCH.tau * dx))
                NSCH.tau.x.scatter_forward()

                NSCH.p.x.array[:] = problem_p.solve().x.array
                NSCH.p.x.scatter_forward()

                # Newton
                solver_phi.solve(NSCH.solvector_phi)
                NSCH.solvector_phi.x.scatter_forward()

                # --- Save solution (to be used in next iteration)
                phi, mu, p1c_phi = NSCH.solvector_phi.split()
                v, u, tau, p = NSCH.v, NSCH.u, NSCH.tau, NSCH.p
                phi0, mu0, p1c_phi0 = NSCH.solvector_phi0.split()
                v0, u0, tau0, p0 = NSCH.v0, NSCH.u0, NSCH.tau0, NSCH.p0

                # --- Yield data computed in current iteration
                yield {'step': step, 't': NSCH.t, 'u':u, 'p':p, 'phi': phi.collapse(), 'mu':mu.collapse(), 'p1c_phi':p1c_phi.collapse(), 'v':v, 'tau':tau, 'u0':u0, 'p0':p0, 'phi0': phi0.collapse(), 'mu0':mu0.collapse(), 'p1c_phi0':p1c_phi0.collapse(), 'v0': v0, 'tau0': tau0}

                # --- Update solution
                NSCH.phiold.x.array[:] = NSCH.solvector_phi0.x.array[NSCH.maps[0]]
                NSCH.phiold.x.scatter_forward()
                NSCH.solvector_phi0.x.array[:] = NSCH.solvector_phi.x.array
                NSCH.solvector_phi0.x.scatter_forward()
                NSCH.v0.x.array[:] = NSCH.v.x.array
                NSCH.v0.x.scatter_forward()
                NSCH.tau0.x.array[:] = NSCH.tau.x.array
                NSCH.tau0.x.scatter_forward()
                NSCH.p0.x.array[:] = NSCH.p.x.array
                NSCH.p0.x.scatter_forward()

            step = step + 1


# ---------------------------

def print_info(i, t, phi_data, p1c_phi_data, p_data,energy, dynamics=0):
# energy, dynamics = 0):
    phi_max, phi_min, phi_mass = phi_data
    p1c_phi_max, p1c_phi_min, p1c_phi_mass = p1c_phi_data
    p_max, p_min, p_mass = p_data
    s = f"{i:3} {t:.6e} {phi_max:.4e} {p1c_phi_max:.4e} {NSCH.rho(phi_max):.4e} {NSCH.rho(p1c_phi_max):.4e} {p_max:.4e}"
    s += f" {phi_min:.4e} {p1c_phi_min:.4e} {NSCH.rho(phi_min):.4e} {NSCH.rho(p1c_phi_min):.4e} {p_min:.4e}"
    s += f" {phi_mass:.4e} {p1c_phi_mass:.4e} {NSCH.rho_dif * phi_mass:.4e} {NSCH.rho_dif * p1c_phi_mass:.4e} {p_mass: 4e}"
    s += f" {energy:.4e}"
    if dynamics:
        dynamics_phi, dynamics_p1c_phi = dynamics
        s += f" {dynamics_phi:.4e} {dynamics_p1c_phi:.4e}"
    printMPI(s)


def define_parameters():

    parser = argparse.ArgumentParser()

    # Define remaining parameters
    parser.add_argument('--eps', default=1e-2)
    parser.add_argument('--lamb', default=1e-2)
    parser.add_argument('--rho1', default=1.0)
    parser.add_argument('--rho2', default=100.0)
    parser.add_argument('--eta', default=1.0)
    parser.add_argument('--sigma', default=4)

    parser.add_argument('--dim', choices=['2d', '3d'], default='2d')
    parser.add_argument('--test', choices=['spinoidal', 'bubble', 'rayleigh', 'square', 'circle', 'order', 'rayleigh_std'], default='spinoidal')
    parser.add_argument('--mesh', default="square")

    parser.add_argument('--gravity', default=1, help="Introduce gravity acceleration")

    # Params for the discrete scheme
    parser.add_argument('--nx', default=30)
    parser.add_argument('--dt', default=1e-3)
    parser.add_argument('--tsteps', default=100)

    # Other parameters
    parser.add_argument('--verbosity', default=0, help="Extra information shown")
    parser.add_argument('--plot', default=10, help="Plot shown every number of time steps")
    parser.add_argument('--plot_mesh', default=0, help="Plot mesh")
    parser.add_argument('--vtk', default=0, help="vtk photogram saved to disk")
    parser.add_argument('--vtkfile', default="NSCH_DG-UPW", help="Name of vtk file")
    parser.add_argument('--save', default=1, help="Figures and output saved")
    parser.add_argument('--savefile', default="NSCH_DG-UPW", help="Name of output file")
    parser.add_argument('--savefunc', default=0, help="Functions saved in HDF5 format")
    parser.add_argument('--server', default=0, help="Set to 1 if the code is set to run on a server")

    param = parser.parse_args()

    return param

#
# Main program
#
if(__name__ == "__main__"):
    #
    # Define parameters
    #
    params = parameters = define_parameters()
    printMPI("Parameters:")
    for k, v in vars(parameters).items():
        printMPI(f"  {k} = {v}")
    
    if int(params.verbosity):
        log.set_log_level(log.LogLevel.INFO)
        opts = PETSc.Options()
        opts["ksp_monitor"] = True
    else:
        log.set_log_level(log.LogLevel.ERROR)

    #
    # Init problem
    #
    NSCH = NSCH_DG_UPW(parameters)
    NSCH.load_initial_values()
    NSCH.variational_problem_v()
    NSCH.variational_problem_tau()
    NSCH.variational_problem_p()
    NSCH.variational_problem_phi()
    NSCH.create_systems()

    #
    # Save output
    #
    do_save = int(params.save)
    server = int(params.server)
    base_name_save = params.savefile
    savefunc = int(params.savefunc)

    #
    # Save mesh to XDMF
    #
    if savefunc:
        import adios4dolfinx as adx
        adx.write_mesh(NSCH.mesh, f"{base_name_save}_mesh")
    
        with XDMFFile(comm, f"{base_name_save}.xdmf", "w") as xdmf:
            xdmf.write_mesh(NSCH.mesh)

    #
    # Plot
    #
    if do_save:
        pyvista.OFF_SCREEN = True
        if server:
            pyvista.start_xvfb()

    do_plot = (int(params.plot) > 0)
    plot_mesh = (int(params.plot_mesh) > 0)
    pyvista.set_plot_theme("document")

    if plot_mesh: # Plot mesh
        topology, cell_types, geometry = plot.create_vtk_mesh(NSCH.mesh, NSCH.mesh.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True, color="white")
        plotter.view_xy()
        if pyvista.OFF_SCREEN:
            plotter.screenshot("mesh.png", transparent_background=True)
            plotter.close()

            comm.Barrier()
            if rank == 0:
                img = Image.open(f"mesh.png")
                width, height = img.size
                # Setting the points for cropped image
                left = width/6
                top = 0.08 * height
                right = 5 * width/6
                bottom = 0.92 * height
                im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                im_cropped.save(f"mesh.png")
                img.close()
            comm.Barrier()
        else:
            plotter.show()
            plotter.close()

    #
    # More info
    #  
    printMPI("More info:")
    tdim = NSCH.mesh.topology.dim
    num_cells = NSCH.mesh.topology.index_map(tdim).size_local
    h = dolfinx.cpp.mesh.h(NSCH.mesh, tdim, range(num_cells))
    printMPI(f"h = {comm.allreduce(max(h),op=MPI.MAX)}")

    #
    # Save max, min and energy
    #
    max_phi_list = []
    min_phi_list = []
    max_p1c_phi_list = []
    min_p1c_phi_list = []
    max_p_list = []
    min_p_list = []
    E = []
    dynam_phi_list = []
    dynam_p1c_phi_list = []
    dynamics = 0

    #
    # Print info
    #
    printMPI("Time steps:\n  i t phi_max p1c_phi_max rho_max p1c_rho_max p_max phi_min p1c_phi_min rho_min p1c_rho_min p_min phi_mass p1c_phi_mass rho_mass p1c_rho_mass p_mass energy dynamics_phi dynamics_p1c_phi")

    #
    # Time iterations
    #
    NSCH_iterations = NSCH.time_iterator(tsteps=int(params.tsteps), verbosity=int(params.verbosity))
    
    for t_step in NSCH_iterations:

        i, t = t_step['step'], t_step['t']
        phi, p1c_phi, mu, u, p = t_step['phi'], t_step['p1c_phi'], t_step['mu'], t_step['u'], t_step['p']

        #
        # Print info
        #
        phi_max, phi_min = comm.allreduce(max(phi.x.array), op=MPI.MAX), comm.allreduce(min(phi.x.array), op=MPI.MIN)
        p1c_phi_max, p1c_phi_min = comm.allreduce(max(p1c_phi.x.array), op=MPI.MAX), comm.allreduce(min(p1c_phi.x.array), op=MPI.MIN)
        p_max, p_min = comm.allreduce(max(p.x.array), op=MPI.MAX), comm.allreduce(min(p.x.array), op=MPI.MIN)
        p_mass = assemble_scalar(form(p*dx))
        phi_mass, p1c_phi_mass = assemble_scalar(form(phi*dx)), assemble_scalar(form(p1c_phi *dx))
        energy = assemble_scalar(form(
            NSCH.rho(p1c_phi) * inner(u, u)/2 * dx \
            + NSCH.lamb * NSCH.eps/2 * inner(grad(p1c_phi), grad(p1c_phi)) * dx \
            + NSCH.lamb/NSCH.eps * 1/4 * (1 - p1c_phi**2)**2 * dx
        ))
        if rank == 0:
            max_phi_list.append(phi_max)
            min_phi_list.append(phi_min)
            max_p1c_phi_list.append(p1c_phi_max)
            min_p1c_phi_list.append(p1c_phi_min)
            max_p_list.append(p_max)
            min_p_list.append(p_min)
            E.append(energy)

        if t>DOLFIN_EPS:
            phi0, p1c_phi0 = t_step['phi0'], t_step['p1c_phi0']

            dynamics_phi = comm.allreduce(max(np.abs(phi.x.array - phi0.x.array)), MPI.MAX) / comm.allreduce(max(np.abs(phi0.x.array)), MPI.MAX)
            dynamics_p1c_phi = comm.allreduce(max(np.abs(p1c_phi.x.array - p1c_phi0.x.array)), MPI.MAX) / comm.allreduce(max(np.abs(p1c_phi0.x.array)), MPI.MAX)

            if rank == 0:
                dynam_phi_list.append(dynamics_phi)
                dynam_p1c_phi_list.append(dynamics_p1c_phi)

                dynamics = (dynamics_phi, dynamics_p1c_phi)

        print_info(i, t,
                (phi_max, phi_min, phi_mass),
                (p1c_phi_max, p1c_phi_min, p1c_phi_mass),
                (p_max, p_min, p_mass),
                energy,
                dynamics)
            
        #
        # Plot
        #
        if (do_plot and i % int(params.tsteps) % int(params.plot) == 0):

            # Properties of the scalar bar
            if NSCH.parameters.dim == '2d':
                sargs = dict(height=0.6, vertical=True, position_x=0.8, position_y=0.2, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2f", font_family="arial")
            else:
                sargs = dict(height=0.1, width=0.4, vertical=False, position_x=0.3, position_y=0.4, title='', label_font_size=20, shadow=True,n_labels=5, fmt="%.2g", font_family="arial", color="white")

            # Create a grid to attach the DoF values
            topology, cell_types, geometry = plot.create_vtk_mesh(NSCH.spaces[2])
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            aux = p1c_phi.x.array
            aux[np.abs(aux + 1.0) < 1e-16] = -1.0
            aux[np.abs(aux - 1.0) < 1e-16] = 1.0
            grid.point_data["Pi1_phi"] = aux

            grid.set_active_scalars("Pi1_phi")
            
            # Velocity field

            u_values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
            u_values[:, :len(u)] = NSCH.project(u, NSCH.P1cus).x.array.real.reshape((geometry.shape[0], len(u)))

            # Create a point cloud of glyphs
            u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            u_grid["u"] = u_values
            max_u = np.max(np.linalg.norm(u_values, axis=1))
            if max_u > 5e-2: factor = 5e-2/max_u
            else: factor = 1.0
            glyphs = u_grid.glyph(orient="u", factor=factor, tolerance=0.05)

            plotter = pyvista.Plotter()
            if NSCH.parameters.dim == '2d':
                plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["coolwarm"], scalar_bar_args=sargs)
                plotter.add_mesh(glyphs, color="white")
                plotter.view_xy()
            else:
                clipped = grid.clip('y')
                plotter.add_mesh(clipped, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["plasma"], scalar_bar_args=sargs)
                plotter.view_xy()
                plotter.camera.elevation = 45.0 # Angle of view
                plotter.camera.zoom(1.2)

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                figure = plotter.screenshot(f"./{base_name_save}_Pi1_phi_i-{i}.png", transparent_background=True)
                plotter.close()
                
                comm.Barrier()
                if rank == 0:
                    img = Image.open(f"./{base_name_save}_Pi1_phi_i-{i}.png")
                    width, height = img.size
                    # Setting the points for cropped image
                    if NSCH.parameters.dim == "2d":
                        left = width/6
                        top = 0.08 * height
                        right = 0.96 * width
                        bottom = 0.92 * height
                    else:
                        left = width/6
                        top = 0.08 * height
                        right = 0.83 * width
                        bottom = 0.8 * height
                    im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                    im_cropped.save(f"./{base_name_save}_Pi1_phi_i-{i}.png")
                    img.close()
                comm.Barrier()
            else:
                plotter.show()
                plotter.close()

        #
        # Save functions to XDMF
        #
        if (savefunc and i % int(params.tsteps) % savefunc == 0):
            u_P2bvs = NSCH.project(u, NSCH.P2bvs)

            adx.write_function(phi, f"{base_name_save}_phi_i-{i}")
            adx.write_function(p1c_phi, f"{base_name_save}_p1c_phi_i-{i}")
            adx.write_function(u_P2bvs, f"{base_name_save}_u_i-{i}")
            adx.write_function(p, f"{base_name_save}_p_i-{i}")

            with dolfinx.io.XDMFFile(comm, f"{base_name_save}.xdmf", "a") as xdmf:
                phi.name = "phi"
                p1c_phi.name = "p1c_phi"
                u_P2bvs.name = "u"
                p.name = "p"
                xdmf.write_function(phi, t=t)
                xdmf.write_function(p1c_phi, t=t)
                xdmf.write_function(u_P2bvs, t=t)
                xdmf.write_function(p, t=t)

    #
    # Plot
    #
    if do_plot:
        time_steps = np.linspace(0, t, int(params.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(params.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_phi_list,'--',c='orange')
        axs[1].plot(time_steps,np.full(int(params.tsteps) + 1, -1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_phi_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_phi.png")
        else: plt.show()
        plt.close()

        time_steps = np.linspace(0, t, int(params.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(params.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_p1c_phi_list,'--',c='orange')
        axs[1].plot(time_steps,np.full(int(params.tsteps) + 1, -1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_p1c_phi_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_p1c_phi.png")
        else: plt.show()
        plt.close()

        plt.plot(time_steps, E, color='red')
        plt.title("Discrete energy")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        if do_save: plt.savefig(f"{base_name_save}_energy.png")
        else: plt.show()
        plt.close()

        plt.plot(np.linspace(0, t, int(params.tsteps)), dynam_phi_list, color='darkblue')
        plt.title("Dynamics phi")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_phi.png")
        else: plt.show()
        plt.close()

        plt.plot(np.linspace(0, t, int(params.tsteps)), dynam_p1c_phi_list, color='darkblue')
        plt.title("Dynamics p1c_phi")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_p1c_phi.png")
        else: plt.show()
        plt.close()