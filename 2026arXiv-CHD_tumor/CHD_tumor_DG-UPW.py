# 
# Fenicsx v.0.10.0
#
# ====================================
# CHD tumor model with coupled DG-UPW
# ====================================
#

import dolfinx
from dolfinx.fem import (
    Expression, Function, functionspace,
    assemble_scalar, form
)
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx import log
from basix.ufl import element, mixed_element
from ufl import(
     TestFunction, TrialFunction,
     SpatialCoordinate,
     dx, dS, inner, grad, div,
     avg, jump,
     tanh, sqrt, sign,
     split,
     FacetArea, FacetNormal,
     Measure 
)
from dolfinx.io import (
    XDMFFile, VTKFile
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
# dx_ML = dx(scheme="vertex", metadata={"degree":1, "representation":"quadrature"}) # mass lumped terms
dx_ML = Measure("dx", metadata = {"quadrature_rule": "vertex"})

#
# Problem class
#
class CHD_tumor_DG_UPW(object):
    r"""
    DG numerical solution of Navier-Stokes-Cahn-Hilliard equation
    with Neumann homogeneous conditions
    """

    def __init__(self, CHD_parameters):
        #
        # Load PDE and discretization parameters
        #
        CHD = self
        params = CHD.parameters = CHD_parameters

        CHD.eps = float(params.eps)

        CHD.K = float(params.K)
        CHD.delta = float(params.delta)
        CHD.P0 = float(params.P0)
        CHD.chi0 = float(params.chi0)
        CHD.Cu = float(params.Cu)
        CHD.Cn = float(params.Cn)

        CHD.eta = float(params.eta)
        CHD.p_unique = float(params.p_unique)

        CHD.solver = params.solver

        file_path = os.path.dirname(os.path.abspath(__file__))
        printMPI(f"file_path = {file_path}")
        mesh_file = f"{file_path}/meshes/" + f"mesh_big_square_nx-{params.nx}.xdmf"
        printMPI(f"mesh_file = {mesh_file}")

        #
        # Read mesh
        #
        with XDMFFile(comm, mesh_file, 'r') as infile:
            mesh = CHD.mesh = infile.read_mesh()

        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_local
        h = dolfinx.cpp.mesh.h(
                mesh._cpp_object, 2, np.arange(mesh.topology.index_map(2).size_local,dtype=np.int32))
        CHD.h_max = comm.allreduce(max(h),op=MPI.MAX)
        
        CHD.nx = int(params.nx)
        CHD.dt = float(params.dt)
        CHD.t = 0.

        #
        # Build DG, FE spaces and functions
        #
        CHD.P0d = element("DG", mesh.topology.cell_name(), 0)
        CHD.P1d = element("DG", mesh.topology.cell_name(), 1)
        CHD.P1c = element("Lagrange", mesh.topology.cell_name(), 1)
        CHD.P2c = element("Lagrange", mesh.topology.cell_name(), 2)
        CHD.P1cvec = element("Lagrange", mesh.topology.cell_name(), degree=1, shape=(mesh.topology.dim,))
        CHD.BDM1 = element("BDM", mesh.topology.cell_name(), 1, shape=(mesh.topology.dim,))

        CHD.P0ds = functionspace(mesh, CHD.P0d)
        CHD.P1cs = functionspace(mesh, CHD.P1c)
        CHD.P2cs = functionspace(mesh, CHD.P2c)
        CHD.P1cvecs = functionspace(mesh, CHD.P1cvec)
        CHD.Vh = functionspace(mesh, CHD.BDM1)
        CHD.Wh = functionspace(mesh, mixed_element([CHD.BDM1, CHD.P0d, CHD.P0d, CHD.P1c, CHD.P1c, CHD.P0d, CHD.P0d]))

        CHD.solvector, CHD.testvector = Function(CHD.Wh), TestFunction(CHD.Wh)
        CHD.solvector0 = Function(CHD.Wh)

        CHD.v, CHD.p, CHD.u, CHD.p1c_u, CHD.mu_u, CHD.p0d_mu_u, CHD.n = split(CHD.solvector)
        CHD.vb, CHD.pb, CHD.ub, CHD.p1c_ub, CHD.mu_ub, CHD.p0d_mu_ub, CHD.nb = split(CHD.testvector)
        CHD.v0, CHD.p0, CHD.u0, CHD.p1c_u0, CHD.mu_u0, CHD.p0d_mu_u0, CHD.n0 = split(CHD.solvector0)
        CHD.p0d_p1c_u0 = Function(CHD.P0ds)

        def mu_n(n, p0d_p1c_u0):
            return (1/CHD.delta * n - CHD.chi0*p0d_p1c_u0)
        CHD.mu_n = mu_n

        # Compute subspaces and maps from subspaces to main space in MixedElement space
        CHD.num_subs = CHD.Wh.num_sub_spaces
        CHD.spaces = []
        CHD.maps = []
        for i in range(CHD.num_subs):
            space_i, map_i = CHD.Wh.sub(i).collapse()
            CHD.spaces.append(space_i)
            CHD.maps.append(map_i)

        # Domain size
        aux = Function(CHD.spaces[1])
        aux.x.array[:] = 1.0
        CHD.domain_size = assemble_scalar(form(aux * dx))

    def project(self, u, space, mass_lumping=False):
        CHD = self

        Piu_trial = TrialFunction(space)
        Piub = TestFunction(space)

        if mass_lumping:
            a = inner(Piu_trial, Piub) * dx_ML
        else:
            a = inner(Piu_trial, Piub) * dx

        L = inner(u, Piub) * dx

        problem = LinearProblem(a, L, petsc_options_prefix="project_")
        return problem.solve()
    
    def fwell(self, p1c_u, p1c_u0):
        return 0.25 * (3.0 * p1c_u + 4.0*(pow(p1c_u0,3)) - 6.0*(pow(p1c_u0,2)) - p1c_u0)

    def load_initial_values(self):
        """Initialize variables"""
        CHD = self
        eps = CHD.eps
        K = CHD.K
        delta = CHD.delta
        chi0 = CHD.chi0

        #
        # Initial condition
        #
        x = SpatialCoordinate(CHD.mesh)

        u0_dict = {
            "three_tumors": (1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 2, 2) + pow(x[1] - 2, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 3, 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 1.0/2.0 * (tanh((1.73 - sqrt(pow(x[0] + 1.5,2) + pow(x[1] + 1.5, 2)))/(sqrt(2.0) * eps)) + 1.0)),
            "single_tumor": (1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * eps)) + 1.0))

            }
        
        n0_dict = {
            "three_tumors": (1.0 -
                    1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 2, 2) + pow(x[1] - 2, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    - 1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 3, 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    - 1.0/2.0 * (tanh((1.73 - sqrt(pow(x[0] + 1.5,2) + pow(x[1] + 1.5, 2)))/(sqrt(2.0) * eps)) + 1.0)),
            "single_tumor": (0.5 * (1.0 - 1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * eps)) + 1.0))  + 0.5 * 1.0/2.0 * (tanh((1.0 - sqrt(pow(x[0] - 2.45, 2) + pow(x[1] - 1.45, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 0.5 * 1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0] + 3.75, 2) + pow(x[1] - 1.0, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 0.5 * 1.0/2.0 * (tanh((2.5 - sqrt(pow(x[0], 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * eps)) + 1.0))
            }
        
        print("u0 =", str(u0_dict[params.initial_cond]))
        print("n0 =", str(n0_dict[params.initial_cond]))

        u_init = Expression(u0_dict[params.initial_cond], CHD.Wh.sub(2).element.interpolation_points)
        n_init = Expression(n0_dict[params.initial_cond], CHD.Wh.sub(6).element.interpolation_points)

        # Initial values
        CHD.solvector0.sub(2).interpolate(u_init)
        CHD.solvector0.x.scatter_forward()
        u0 = CHD.u0 = CHD.solvector0.sub(2)

        CHD.solvector0.sub(6).interpolate(n_init)
        CHD.solvector0.x.scatter_forward()
        n0 = CHD.solvector0.sub(6)

        CHD.solvector0.x.array[CHD.maps[1]] = 0.0
        CHD.solvector0.x.scatter_forward()
        p0 = CHD.p0 = CHD.solvector0.sub(1)

        CHD.solvector0.x.array[CHD.maps[3]] = CHD.project(u0, CHD.spaces[3], mass_lumping=True).x.array
        CHD.solvector0.x.scatter_forward()
        p1c_u0 = CHD.p1c_u0 = CHD.solvector0.sub(3)

        p2c_u0 = Function(CHD.P2cs)
        p2c_u0.interpolate(Expression(u0_dict[params.initial_cond], CHD.P2cs.element.interpolation_points))
        CHD.solvector0.x.array[CHD.maps[4]] = CHD.project(CHD.fwell(p2c_u0, p2c_u0) - CHD.eps**2*div(grad(p2c_u0)) - CHD.chi0*n0, CHD.spaces[4]).x.array
        CHD.solvector0.x.scatter_forward()
        mu_u0 = CHD.mu_u0 = CHD.solvector0.sub(4)

        CHD.solvector0.x.array[CHD.maps[5]] = CHD.project(mu_u0, CHD.spaces[5]).x.array
        CHD.solvector0.x.scatter_forward()
        p0d_mu0 = CHD.p0d_mu0 = CHD.solvector0.sub(5)

        CHD.p0d_p1c_u0.x.array[:] = CHD.project(p1c_u0, CHD.P0ds).x.array
        CHD.p0d_p1c_u0.x.scatter_forward()

        CHD.solvector0.x.array[CHD.maps[0]] = 0.0
        CHD.solvector0.x.scatter_forward()
        v0 = CHD.v0 = CHD.solvector0.sub(0)

    def variational_problem(self):
        """Build variational problem"""
        #
        # Load variables from DCH problem
        #
        CHD = self
        params = CHD.parameters
        dt = CHD.dt
        nx = CHD.nx

        eps = CHD.eps

        K = CHD.K
        delta = CHD.delta
        P0 = CHD.P0
        chi0 = CHD.chi0
        Cu = CHD.Cu
        Cn = CHD.Cn

        eta = CHD.eta
        p_unique = CHD.p_unique

        v, p, u, p1c_u, mu_u, p0d_mu_u, n = CHD.v, CHD.p, CHD.u, CHD.p1c_u, CHD.mu_u, CHD.p0d_mu_u, CHD.n
        vb, pb, ub, p1c_ub, mu_ub, p0d_mu_ub, nb = CHD.vb, CHD.pb, CHD.ub, CHD.p1c_ub, CHD.mu_ub, CHD.p0d_mu_ub, CHD.nb
        v0, p0, u0, p1c_u0, mu_u0, p0d_mu_u0, n0 = CHD.v0, CHD.p0, CHD.u0, CHD.p1c_u0, CHD.mu_u0, CHD.p0d_mu_u0, CHD.n0
        p0d_p1c_u0 = CHD.p0d_p1c_u0
        mu_n = CHD.mu_n

        def pos(phi):
            return ((abs(phi) + phi) / 2.0)
        def neg(phi):
            return ((abs(phi) - phi) / 2.0)
        
        symmetric = int(params.symmetric)
        
        if symmetric:
            def M(phi):
                """Mobility function"""
                return (phi * (1.0 - phi))/(1/2 * (1 - 1/2))
            def Mpos(phi):
                """Positive part of mobility function"""
                return pos(M(phi))
            def Mup(phi):
                """Increasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (phi + 1.0 / 2.0 - abs(phi - 1.0 / 2.0)))
            def Mdown(phi):
                """Decreasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (phi + 1.0 / 2.0 + abs(phi - 1.0 / 2.0))) - Mpos(1.0 / 2.0)
            def P(phi):
                return Mpos(phi)
        else:
            def M(phi):
                """Mobility function"""
                return (phi**5 * (1.0 - phi))/((5/6)**5 * (1.0 - 5/6))
            def Mpos(phi):
                """Positive part of mobility function"""
                return pos(M(phi))
            def Mup(phi):
                """Increasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (phi + 5.0 / 6.0 - abs(phi - 5.0 / 6.0)))
            def Mdown(phi):
                """Decreasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (phi + 5.0 / 6.0 + abs(phi - 5.0 / 6.0))) - Mpos(5.0 / 6.0)
            def P(phi):
                return pos(phi * (1.0 - phi)**3)/(1/4 * (1.0 - 1/4)**3)
        
        #
        # PDE functions
        #
        CHD.Fwell = lambda u: 0.25 * (pow(u,2)) * (pow(u-1,2))

        #
        # Variational problem
        #
        e_len = FacetArea(CHD.mesh)
        n_e = FacetNormal(CHD.mesh)
        l = 20.0/nx

        def aupw(v, phi, phib):
            # UPW bilinear form
            return (
                pos(inner(v, n_e)('+')) * phi('+') * jump(phib) * dS \
                - neg(inner(v, n_e)('+')) * phi('-') * jump(phib) * dS
            )

        def bupw(p0d_mu, phi, phib):
            # UPW bilinear form
            return (
                pos(jump(p0d_mu)/((2.0*pow(l,2))/(3.0*avg(e_len)))) * pos(Mup(phi('+')) + Mdown(phi('-'))) * jump(phib) * dS \
                - neg(jump(p0d_mu)/((2.0*pow(l,2))/(3.0*avg(e_len)))) * pos(Mup(phi('-')) + Mdown(phi('+'))) * jump(phib) * dS
            )
        
        def ch(phi, p0d_mu, vb):
            # Centered discretization
            return(
                - inner(vb, n_e)('+') * avg(phi) * jump(p0d_mu) * dS 
                - div(vb) * phi * p0d_mu * dx
            )

        def sh(v, phi, p0d_mu, vb):
            return(- 1/2 * inner(vb, n_e)('+') * sign(inner(v, n_e)('+')) * jump(phi) * jump(p0d_mu) * dS)
        def shd(v, phi, p0d_mu, vb, eta = 1e-10):
            return(- 1/2 * inner(vb, n_e)('+') * inner(v, n_e)('+')/(abs(inner(v, n_e)('+')) + eta) * jump(phi) * jump(p0d_mu) * dS)
        
        if K > DOLFIN_EPS:
            CHD.F_v = F_v = 1/K * inner(v, vb) * dx \
                    - inner(p, div(vb)) * dx \
                    + ch(u, p0d_mu_u, vb) \
                    + ch(n, mu_n(n, p0d_p1c_u0), vb)

            if eta > DOLFIN_EPS:
                CHD.F_v = F_v = F_v + shd(v, u, p0d_mu_u, vb, eta) \
                        + shd(v, n, mu_n(n, p0d_p1c_u0), vb, eta)
            elif not(eta < -DOLFIN_EPS):
                CHD.F_v = F_v = F_v + sh(v, u, p0d_mu_u, vb) \
                        + sh(v, n, mu_n(n, p0d_p1c_u0), vb)
            
            CHD.F_p = F_p = inner(div(v), pb) * dx
            
            if p_unique > DOLFIN_EPS:
                CHD.F_p = F_p = F_p + p_unique *  inner(p, pb) * dx

        def no_slip(x):
            return (np.logical_or(np.isclose(np.abs(x[0]), 10), np.isclose(np.abs(x[1]), 10)))
        
        v_bc = Function(CHD.spaces[0])
        v_bc.x.petsc_vec.set(0)
        facets = dolfinx.mesh.locate_entities_boundary(CHD.mesh, dim=1, marker=no_slip)
        CHD.bcv = bcv = dolfinx.fem.dirichletbc(
            v_bc, dolfinx.fem.locate_dofs_topological((CHD.Wh.sub(0), CHD.spaces[0]), 1, facets), CHD.Wh.sub(0))

        CHD.F_u = F_u = inner(u, ub) * dx \
                - inner(u0, ub) * dx \
                + dt * Cu * bupw(p0d_mu_u, u, ub) \
                + dt * aupw(v, u, ub) \
                - dt * delta * P0 * P(u) * pos(n) * pos(mu_n(n, p0d_p1c_u0) - p0d_mu_u) * ub * dx
        
        CHD.F_p1c_u = F_p1c_u = inner(p1c_u, p1c_ub) * dx_ML \
                - inner(u, p1c_ub) * dx
        
        CHD.F_mu_u = F_mu_u = inner(mu_u, mu_ub) * dx_ML \
            - pow(eps, 2) * inner(grad(p1c_u), grad(mu_ub)) * dx \
            - inner(CHD.fwell(p1c_u,p1c_u0), mu_ub) * dx \
            + chi0 * inner(n, mu_ub) * dx
        
        CHD.F_p0d_mu_u = F_p0d_mu_u = inner(mu_u, p0d_mu_ub) * dx \
                    - inner(p0d_mu_u, p0d_mu_ub) * dx
        
        CHD.F_n = F_n = inner(n, nb) * dx \
                - inner(n0, nb) * dx \
                + dt * Cn * bupw(mu_n(n, p0d_p1c_u0), n, nb) \
                + dt * aupw(v, n, nb) \
                + dt * delta * P0 * P(u) * pos(n) * pos(mu_n(n, p0d_p1c_u0) - p0d_mu_u) * nb * dx
        
        if K == 0:
            CHD.F = F_u + F_p1c_u + F_mu_u + F_p0d_mu_u + F_n
        else:
            CHD.F = F_v + F_p + F_u + F_p1c_u + F_mu_u + F_p0d_mu_u + F_n

    def create_system(self):
        #
        # Load variables from DCH problem
        #
        CHD = self
        params = CHD.parameters

        #
        # PETSc options
        #
        if CHD.solver == "mumps":
            petsc_options = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "snes_atol": 1e-6,
                "snes_rtol": 1e-6,
                # "snes_monitor": None,
                "snes_error_if_not_converged": True,
                "ksp_error_if_not_converged": True,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
            if CHD.p_unique < DOLFIN_EPS:
                petsc_options["mat_mumps_icntl_24"] = 1  # Option for singular systems
                petsc_options["mat_mumps_icntl_25"] = 0  # Option for singular systems
        if CHD.solver == "gmres":
            petsc_options = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "snes_atol": 1e-6,
                "snes_rtol": 1e-6,
                "ksp_rtol": 1e-8,
                "ksp_atol": 1e-8,
                # "snes_monitor": None,
                "snes_error_if_not_converged": True,
                "ksp_error_if_not_converged": True,
                "ksp_type": "gmres",
            }
            if CHD.K > 0:
                petsc_options["pc_type"] = "ilu"
            else:
                # petsc_options["pc_type"] = "jacobi"
                petsc_options["pc_type"] = "ilu"
        CHD.petsc_options = petsc_options

        #
        # Initialization
        #
        CHD.solvector.x.array[:] = CHD.solvector0.x.array
        CHD.solvector.x.scatter_forward()

        #
        # Nullspace for pressure constraint
        #
        if CHD.solver=="gmres" and CHD.K > DOLFIN_EPS and CHD.p_unique < DOLFIN_EPS:
            ns_vec = Function(CHD.Wh)
            ns_vec.x.array[:] = 0.0
            ns_vec.x.array[CHD.maps[1]] = 1.0
            ns_vec.x.scatter_forward()

            # # Normalize the vector, create a nullspace object, and attach it to the matrix
            nsp = PETSc.NullSpace().create(vectors=[ns_vec.x.petsc_vec], comm=comm)

        #
        # Define problem
        #

        # Newton
        problem = CHD.problem = NonlinearProblem(CHD.F, CHD.solvector, bcs=[CHD.bcv], petsc_options_prefix="newton_", petsc_options=petsc_options)

        if CHD.solver=="gmres" and CHD.K > DOLFIN_EPS and CHD.p_unique < DOLFIN_EPS:
            CHD.problem.A.setNullSpace(nsp)
            CHD.problem.A.setNearNullSpace(nsp)
    
    def time_iterator(self, tsteps=1, first_step=1):
        """Time iterator"""
        CHD = self
        params = CHD.parameters

        #
        # Run time iterations
        #
        step = first_step - 1
        last_step = first_step + tsteps

        while step < last_step:
            if step == first_step - 1:
                v0, p0, u0, p1c_u0, mu_u0, p0d_mu0, n0 = CHD.solvector0.split()

                # --- Yield initial data
                yield {'step': step, 't': CHD.t, 'v':v0.collapse(), 'p':p0.collapse(), 'u':u0.collapse(), 'p1c_u':p1c_u0.collapse(), 'mu_u':mu_u0.collapse(), 'p0d_mu':p0d_mu0.collapse(), 'n':n0.collapse()}

            else:
                CHD.t += CHD.dt

                # Solve

                # Newton
                CHD.problem.solve()
                # converged_reason = CHD.problem.solver.getConvergedReason()
                # assert converged_reason > 0
                # if converged_reason == -3: # Divergence, try direct solver
                #     CHD.problem.solver.ksp.setType("preonly")
                #     CHD.problem.solver.ksp.pc.setType("lu")
                #     CHD.problem.solve()
                #     CHD.problem.solver.ksp.setType(CHD.petsc_options["ksp_type"])
                #     CHD.problem.solver.ksp.pc.setType(CHD.petsc_options["pc_type"])
                # else:
                #     assert converged_reason > 0
                # num_iterations = CHD.problem.solver.getIterationNumber()
                # print(f"Step {step}: {converged_reason=} {num_iterations=}")

                CHD.solvector.x.scatter_forward()

                # --- Save solution (to be used in next iteration)
                v, p, u, p1c_u, mu_u, p0d_mu, n = CHD.solvector.split()
                v0, p0, u0, p1c_u0, mu_u0, p0d_mu0, n0 = CHD.solvector0.split()

                # --- Correct pressure to have zero mean
                p_mean = assemble_scalar(form(p * dx)) / CHD.domain_size
                p.x.array[CHD.maps[1]] = p.x.array[CHD.maps[1]] - p_mean
                p.x.scatter_forward()

                # --- Yield data computed in current iteration
                yield {'step': step, 't': CHD.t, 'v':v.collapse(), 'p':p.collapse(), 'u':u.collapse(), 'p1c_u':p1c_u.collapse(), 'mu_u':mu_u.collapse(), 'p0d_mu':p0d_mu.collapse(), 'n':n.collapse(), 'v0':v0.collapse(), 'p0':p0.collapse(), 'u0':u0.collapse(), 'p1c_u0':p1c_u0.collapse(), 'mu_u0':mu_u0.collapse(), 'p0d_mu0':p0d_mu0.collapse(), 'n0':n0.collapse()}

                # --- Update solution
                CHD.solvector0.x.array[:] = CHD.solvector.x.array
                CHD.solvector0.x.scatter_forward()
                CHD.p0d_p1c_u0.x.array[:] = CHD.project(p1c_u, CHD.P0ds).x.array
                CHD.p0d_p1c_u0.x.scatter_forward()

            step = step + 1


# ---------------------------

def print_info(i, t, u_data, p1c_u_data, n_data, u_n_mass_data, p_data, energy, dynamics=0):
# energy, dynamics = 0):
    u_max, u_min = u_data
    p1c_u_max, p1c_u_min = p1c_u_data
    n_max, n_min = n_data
    u_n_mass, p1c_u_n_mass = u_n_mass_data
    p_max, p_min, p_mass = p_data
    s = f"{i:3} {t:.6e} {u_max:.4e} {p1c_u_max:.4e} {n_max:.4e} {p_max:.4e}"
    s += f" {u_min:.4e} {p1c_u_min:.4e} {n_min:.4e} {p_min:.4e}"
    s += f" {u_n_mass:.4e} {p1c_u_n_mass:.4e} {p_mass:.4e}"
    s += f" {energy:.4e}"
    if dynamics:
        dynamics_u, dynamics_p1c_u, dynamics_n = dynamics
        s += f" {dynamics_u:.4e} {dynamics_p1c_u:.4e} {dynamics_n:.4e}"
    printMPI(s)


def define_parameters():

    parser = argparse.ArgumentParser()

    # Define remaining parameters
    parser.add_argument('--eps', default=0.1)

    parser.add_argument('--K', default=1.0)
    parser.add_argument('--delta', default=0.01)
    parser.add_argument('--chi0', default=1.e-14)
    parser.add_argument('--P0', default=300.0)
    parser.add_argument('--Cu', default=1.0)
    parser.add_argument('--Cn', default=1.0)

    parser.add_argument('--eta', default=0)
    parser.add_argument('--p_unique', default=0)

    parser.add_argument('--solver', choices=['mumps', 'gmres'], default='mumps')
    parser.add_argument('--test', choices=['2d', '3d'], default='2d')
    parser.add_argument('--initial_cond', default='three_tumors')
    parser.add_argument('--symmetric', default=0)

    # Params for the discrete scheme
    parser.add_argument('--nx', default=50)
    parser.add_argument('--dt', default=0.2)
    parser.add_argument('--tsteps', default=250)

    # Other parameters
    parser.add_argument('--verbosity', default=0, help="Extra information shown")
    parser.add_argument('--plot', default=10, help="Plot shown every number of time steps")
    parser.add_argument('--plot_mesh', default=0, help="Plot mesh")
    parser.add_argument('--vtk', default=0, help="vtk photogram saved to disk")
    parser.add_argument('--vtkfile', default="CHD_DG-UPW", help="Name of vtk file")
    parser.add_argument('--save', default=1, help="Figures and output saved")
    parser.add_argument('--savefile', default="CHD_DG-UPW", help="Name of output file")
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
    CHD = CHD_tumor_DG_UPW(parameters)
    CHD.load_initial_values()
    CHD.variational_problem()
    CHD.create_system()

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
        adx.write_mesh(CHD.mesh, f"{base_name_save}_mesh")

        with XDMFFile(comm, f"{base_name_save}.xdmf", "w") as xdmf:
            xdmf.write_mesh(CHD.mesh)

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
        topology, cell_types, geometry = plot.create_vtk_mesh(CHD.mesh, CHD.mesh.topology.dim)
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
    printMPI(f"h = {CHD.h_max}")

    #
    # Save max, min and energy
    #
    max_u_list = []
    min_u_list = []
    max_n_list = []
    min_n_list = []
    max_p1c_u_list = []
    min_p1c_u_list = []
    max_p_list = []
    min_p_list = []
    E = []
    dynam_u_list = []
    dynam_n_list = []
    dynam_p1c_u_list = []
    dynamics = 0

    #
    # Print info
    #
    printMPI("Time steps:\n  i t u_max p1c_u_max n_max p_max u_min p1c_u_min n_min p_min u_n_mass p1c_u_n_mass p_mass energy dynamics_u dynamics_p1c_u dynamics_n")

    #
    # VTK output files
    #
    do_vtk = (int(params.vtk) > 0)
    base_name_vtk = params.vtkfile
    if do_vtk:
        vtk_file = VTKFile(comm, f"{base_name_vtk}.vtu", "w")

    #
    # Time iterations
    #
    CHD_iterations = CHD.time_iterator(tsteps=int(params.tsteps))
    
    for t_step in CHD_iterations:

        i, t = t_step['step'], t_step['t']
        v, p, u, p1c_u, mu_u, p0d_mu_u, n = t_step['v'], t_step['p'], t_step['u'], t_step['p1c_u'], t_step['mu_u'], t_step['p0d_mu'], t_step['n']

        #
        # Print info
        #

        u_max, u_min = comm.allreduce(max(u.x.array), op=MPI.MAX), comm.allreduce(min(u.x.array), op=MPI.MIN)
        p1c_u_max, p1c_u_min = comm.allreduce(max(p1c_u.x.array), op=MPI.MAX), comm.allreduce(min(p1c_u.x.array), op=MPI.MIN)
        n_max, n_min = comm.allreduce(max(n.x.array), op=MPI.MAX), comm.allreduce(min(n.x.array), op=MPI.MIN)
        p_max, p_min = comm.allreduce(max(p.x.array), op=MPI.MAX), comm.allreduce(min(p.x.array), op=MPI.MIN)
        p_mass = assemble_scalar(form(p*dx))
        u_n_mass, p1c_u_n_mass = assemble_scalar(form((u + n)*dx)), assemble_scalar(form((p1c_u + n)*dx))
        energy = assemble_scalar(form(
            0.5 * pow(CHD.eps,2) * inner(grad(p1c_u), grad(p1c_u)) * dx \
            + CHD.Fwell(p1c_u) * dx \
            - CHD.chi0 * p1c_u * n * dx \
            + 0.5 * 1/CHD.delta * pow(n, 2) * dx
        ))
        if rank == 0:
            max_u_list.append(u_max)
            min_u_list.append(u_min)
            max_p1c_u_list.append(p1c_u_max)
            min_p1c_u_list.append(p1c_u_min)
            max_n_list.append(n_max)
            min_n_list.append(n_min)
            max_p_list.append(p_max)
            min_p_list.append(p_min)
            E.append(energy)

        if t>DOLFIN_EPS:
            if E[-1]>E[-2]: raise ValueError(f"Energy increased at step {i}: {E[-2]=} -> {E[-1]=}")

            u0, p1c_u0, n0 = t_step['u0'], t_step['p1c_u0'], t_step['n0']

            dynamics_u = comm.allreduce(max(np.abs(u.x.array - u0.x.array)), MPI.MAX) / comm.allreduce(max(np.abs(u0.x.array)), MPI.MAX)
            dynamics_p1c_u = comm.allreduce(max(np.abs(p1c_u.x.array - p1c_u0.x.array)), MPI.MAX) / comm.allreduce(max(np.abs(p1c_u0.x.array)), MPI.MAX)
            dynamics_n = comm.allreduce(max(np.abs(n.x.array - n0.x.array)), MPI.MAX) / comm.allreduce(max(np.abs(n0.x.array)), MPI.MAX)

            # print(dynamics_phi)

            if rank == 0:
                dynam_u_list.append(dynamics_u)
                dynam_p1c_u_list.append(dynamics_p1c_u)
                dynam_n_list.append(dynamics_n)

                dynamics = (dynamics_u, dynamics_p1c_u, dynamics_n)

        print_info(i, t,
                (u_max, u_min),
                (p1c_u_max, p1c_u_min), 
                (n_max, n_min),
                (u_n_mass, p1c_u_n_mass),
                (p_max, p_min, p_mass),
                energy,
                dynamics)
                
        #
        # Plot
        #
        if (do_plot and i % int(params.tsteps) % int(params.plot) == 0):

            # Properties of the scalar bar
            sargs_scalar = dict(height=0.6, vertical=True, position_x=0.8, position_y=0.2, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2f", font_family="arial")
            sargs_vector = dict(height=0.6, vertical=False, position_x=0.5, position_y=-1, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2f", font_family="arial")

            # Create a grid to attach the DoF values
            topology, cell_types, geometry = plot.vtk_mesh(CHD.P1cs)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            aux = p1c_u.x.array
            aux[abs(aux) < 1e-16] = 0.0
            aux[abs(aux - 1.0) < 1e-16] = 1.0
            grid.point_data["Pi1_u"] = aux
            
            aux = CHD.project(n, CHD.P1cs, mass_lumping=True).x.array
            aux[abs(aux) < 1e-16] = 0.0
            aux[abs(aux - 1.0) < 1e-16] = 1.0

            grid.point_data["Pi1_n"] = aux

            aux = CHD.project(p, CHD.P1cs).x.array
            grid.point_data["p"] = aux

            # Velocity field
            v_values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
            v_values[:, :len(v)] = CHD.project(v, CHD.P1cvecs).x.array.real.reshape((geometry.shape[0], len(v)))

            # Create a point cloud of glyphs
            v_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            v_grid["v"] = v_values
            max_v = np.max(np.linalg.norm(v_values, axis=1))
            glyphs = v_grid.glyph(orient="v", scale=False, factor=1.0, tolerance=5e-2, color_mode="vector")
            glyphs_clipped = glyphs.clip_box([-10,10,-10,10,-10,10], invert=False)

            # Plot u
            grid.set_active_scalars("Pi1_u")
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["viridis"], scalar_bar_args=sargs_scalar)
            if i > 0:
                plotter.add_mesh(glyphs_clipped, label="Velocity field", cmap=mpl.colormaps["binary"], show_scalar_bar=False, scalar_bar_args=sargs_vector)
            plotter.view_xy()

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                figure = plotter.screenshot(f"./{base_name_save}_Pi1_u_i-{i}.png", transparent_background=True)
                plotter.close()
                
                comm.Barrier()
                if rank == 0:
                    img = Image.open(f"./{base_name_save}_Pi1_u_i-{i}.png")
                    width, height = img.size
                    # Setting the points for cropped image
                    left = width/6
                    top = 0.08 * height
                    right = 0.96 * width
                    bottom = 0.92 * height
                    im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                    im_cropped.save(f"./{base_name_save}_Pi1_u_i-{i}.png")
                    img.close()
                comm.Barrier()
            else:
                plotter.show()
                plotter.close()

            # Plot n
            grid.set_active_scalars("Pi1_n")
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["Reds"], scalar_bar_args=sargs_scalar)
            if i > 0:
                plotter.add_mesh(glyphs_clipped, label="Velocity field", cmap=mpl.colormaps["binary"], show_scalar_bar=False, scalar_bar_args=sargs_vector)
            plotter.view_xy()

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                figure = plotter.screenshot(f"./{base_name_save}_Pi1_n_i-{i}.png", transparent_background=True)
                plotter.close()
                
                comm.Barrier()
                if rank == 0:
                    img = Image.open(f"./{base_name_save}_Pi1_n_i-{i}.png")
                    width, height = img.size
                    # Setting the points for cropped image
                    left = width/6
                    top = 0.08 * height
                    right = 0.96 * width
                    bottom = 0.92 * height
                    im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                    im_cropped.save(f"./{base_name_save}_Pi1_n_i-{i}.png")
                    img.close()
                comm.Barrier()
            else:
                plotter.show()
                plotter.close()

            # Plot n
            grid.set_active_scalars("p")
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["coolwarm"], scalar_bar_args=sargs_scalar)
            if i > 0:
                plotter.add_mesh(glyphs_clipped, label="Velocity field", cmap=mpl.colormaps["binary"], show_scalar_bar=False, scalar_bar_args=sargs_vector)
            plotter.view_xy()

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                figure = plotter.screenshot(f"./{base_name_save}_p_i-{i}.png", transparent_background=True)
                plotter.close()
                
                comm.Barrier()
                if rank == 0:
                    img = Image.open(f"./{base_name_save}_p_i-{i}.png")
                    width, height = img.size
                    # Setting the points for cropped image
                    left = width/6
                    top = 0.08 * height
                    right = 0.96 * width
                    bottom = 0.92 * height
                    im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                    im_cropped.save(f"./{base_name_save}_p_i-{i}.png")
                    img.close()
                comm.Barrier()
            else:
                plotter.show()
                plotter.close()

        #
        # Save functions to XDMF
        #
        if (savefunc and i % int(params.tsteps) % savefunc == 0):
            adx.write_function(u, f"{base_name_save}_u_i-{i}")
            adx.write_function(p1c_u, f"{base_name_save}_Pi1__u_i-{i}")
            adx.write_function(n, f"{base_name_save}_Pi1_n_i-{i}")
            adx.write_function(v, f"{base_name_save}_v_i-{i}")
            adx.write_function(p, f"{base_name_save}_p_i-{i}")

            with dolfinx.io.XDMFFile(comm, f"{base_name_save}.xdmf", "a") as xdmf:
                u.name = "u"
                p1c_u.name = "p1c_u"
                n.name = "n"
                v.name = "v"
                p.name = "p"
                xdmf.write_function(u, t=t)
                xdmf.write_function(p1c_u, t=t)
                xdmf.write_function(n, t=t)
                xdmf.write_function(v, t=t)
                xdmf.write_function(p, t=t)

        #
        # Save to VTK
        #
        if (do_vtk and i % int(params.tsteps) % int(params.vtk) == 0):
            p_aux = CHD.project(p, CHD.P1cs)
            p_aux.name = "p1c_p"
            v_aux = CHD.project(v, CHD.P1cvecs)
            v_aux.name = "p1c_v"
            p1c_u.name = "p1c_u"
            p1c_n = CHD.project(n, CHD.P1cs, mass_lumping=True)
            p1c_n.name = "p1c_n"
            vtk_file.write_function([p_aux, v_aux, p1c_u, p1c_n], t)

    if do_vtk:
        vtk_file.close()

    #
    # Plot
    #
    if do_plot:
        time_steps = np.linspace(0, t, int(params.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(params.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_u_list,'--',c='orange')
        axs[1].plot(time_steps,np.full(int(params.tsteps) + 1, -1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_u_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_u.png")
        else: plt.show()
        plt.close()

        time_steps = np.linspace(0, t, int(params.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(params.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_p1c_u_list,'--',c='orange')
        axs[1].plot(time_steps,np.full(int(params.tsteps) + 1, -1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_p1c_u_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_p1c_u.png")
        else: plt.show()
        plt.close()

        time_steps = np.linspace(0, t, int(params.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(params.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_n_list,'--',c='orange')
        axs[1].plot(time_steps,np.full(int(params.tsteps) + 1, -1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_n_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_n.png")
        else: plt.show()
        plt.close()

        plt.plot(time_steps, E, color='red')
        plt.title("Discrete energy")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        if do_save: plt.savefig(f"{base_name_save}_energy.png")
        else: plt.show()
        plt.close()

        plt.plot(np.linspace(0, t, int(params.tsteps)), dynam_u_list, color='darkblue')
        plt.title("Dynamics u")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_u.png")
        else: plt.show()
        plt.close()

        plt.plot(np.linspace(0, t, int(params.tsteps)), dynam_p1c_u_list, color='darkblue')
        plt.title("Dynamics p1c_u")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_p1c_u.png")
        else: plt.show()
        plt.close()

        plt.plot(np.linspace(0, t, int(params.tsteps)), dynam_n_list, color='darkblue')
        plt.title("Dynamics n")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_n.png")
        else: plt.show()
        plt.close()