#
# NSCH Test with coupled DG-UPW
# =============================
# FEniCSx version: 0.6
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

        NSCH.delta = float(params.delta)
        NSCH.p_unique = float(params.p_unique)

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

        #
        # Build DG, FE spaces and functions
        #
        NSCH.P0d = FiniteElement("DG", mesh.ufl_cell(), 0)
        NSCH.P1d = FiniteElement("DG", mesh.ufl_cell(), 1)
        NSCH.P1c = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        NSCH.P1cvec = VectorElement(FiniteElement("Lagrange", mesh.ufl_cell(), 1))
        NSCH.P2b = VectorElement(FiniteElement("Lagrange", mesh.ufl_cell(), 2) + FiniteElement("Bubble", mesh.ufl_cell(), 3))

        NSCH.P0ds = FunctionSpace(mesh, NSCH.P0d)
        NSCH.P1cs = FunctionSpace(mesh, NSCH.P1c)
        NSCH.P1cvecs = FunctionSpace(mesh, NSCH.P1cvec)
        NSCH.Uh = FunctionSpace(mesh, NSCH.P2b)
        NSCH.Ph = FunctionSpace(mesh, NSCH.P1d)
        NSCH.Wh = FunctionSpace(mesh, MixedElement([NSCH.P2b, NSCH.P1d, NSCH.P0d, NSCH.P1c, NSCH.P1c, NSCH.P0d]))


        NSCH.solvector, NSCH.testvector = Function(NSCH.Wh), TestFunction(NSCH.Wh)
        NSCH.solvector0 = Function(NSCH.Wh)

        NSCH.u, NSCH.p, NSCH.phi, NSCH.mu, NSCH.p1c_phi, NSCH.p0d_mu = split(NSCH.solvector)
        NSCH.ub, NSCH.pb, NSCH.phib, NSCH.mub, NSCH.p1c_phib, NSCH.p0d_mub = split(NSCH.testvector)
        NSCH.u0, NSCH.p0, NSCH.phi0, NSCH.mu0, NSCH.p1c_phi0, NSCH.p0d_mu0 = split(NSCH.solvector0)
        NSCH.p1c_grad_mu0 = Function(NSCH.P1cvecs)

        # Compute subspaces and maps from subspaces to main space in MixedElement space
        NSCH.num_subs = NSCH.Wh.num_sub_spaces
        NSCH.spaces = []
        NSCH.maps = []
        for i in range(NSCH.num_subs):
            space_i, map_i = NSCH.Wh.sub(i).collapse()
            NSCH.spaces.append(space_i)
            NSCH.maps.append(map_i)

        # Domain size
        aux = Function(NSCH.spaces[1])
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
        "Density of the mixture"
        NSCH = self
        return (NSCH.rho_avg + NSCH.rho_dif * phi)

    def pos(self, phi):
        return ((abs(phi) + phi) / 2.0)
    
    def neg(self, phi):
        return ((abs(phi) - phi) / 2.0)

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
            phi_init = Expression(tanh((0.2 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * eps)), NSCH.spaces[2].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                return vals
        elif NSCH.parameters.test == "rayleigh":
            phi_init = Expression(tanh((x[1] - (0.1 * exp(-(x[0]+0.2)**2/0.1)))/(sqrt(2.0) * eps)), NSCH.spaces[2].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                return vals
        elif NSCH.parameters.test == "circle":
            phi_init = Expression(2 * tanh((NSCH.pos(0.25 - sqrt(pow(x[0]-0.1, 2) + pow(x[1]-0.1, 2))) + NSCH.pos(0.15 - sqrt(pow(x[0]+0.15, 2) + pow(x[1]+0.15, 2))))/(sqrt(2.0) * eps)) - 1.0, NSCH.spaces[2].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                vals[0] = 100 * x[1] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                vals[1] = -100 * x[0] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                return vals
        elif NSCH.parameters.test == "order":
            phi_init = Expression(2 * tanh((NSCH.pos(0.25 - sqrt(pow(x[0]-0.1, 2) + pow(x[1]-0.1, 2))) + NSCH.pos(0.15 - sqrt(pow(x[0]+0.15, 2) + pow(x[1]+0.15, 2))))/(sqrt(2.0) * eps)) - 1.0, NSCH.spaces[2].element.interpolation_points())
            def u_init(x):
                vals = np.zeros((NSCH.mesh.geometry.dim, x.shape[1]))
                vals[0] = x[1] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                vals[1] = -x[0] * NSCH.pos(0.16 - (x[0]**2 + x[1]**2))
                return vals

        # Initial values
        NSCH.solvector0.sub(0).interpolate(u_init)
        NSCH.solvector0.x.scatter_forward()
        u0 = NSCH.u0 = NSCH.solvector0.sub(0)

        NSCH.solvector0.x.array[NSCH.maps[1]] = 0.0
        NSCH.solvector0.x.scatter_forward()
        p0 = NSCH.p0 = NSCH.solvector0.sub(1)

        NSCH.solvector0.sub(2).interpolate(phi_init)
        NSCH.solvector0.x.scatter_forward()
        phi0 = NSCH.phi0 = NSCH.solvector0.sub(2)

        NSCH.solvector0.x.array[NSCH.maps[4]] = NSCH.project(phi0, NSCH.spaces[4], mass_lumping=True).x.array
        NSCH.solvector0.x.scatter_forward()
        p1c_phi0 = NSCH.p1c_phi0 = NSCH.solvector0.sub(4)

        dF_phi0 = lambda phi0: (phi0**2 - 1) * phi0
        NSCH.solvector0.x.array[NSCH.maps[3]] = NSCH.project(lamb/eps * dF_phi0(phi0) - lamb * eps * div(grad(phi0)), NSCH.spaces[3]).x.array
        NSCH.solvector0.x.scatter_forward()
        mu0 = NSCH.mu0 = NSCH.solvector0.sub(3)

        NSCH.p1c_grad_mu0.x.array[:] = NSCH.project(grad(mu0), NSCH.P1cvecs).x.array
        NSCH.p1c_grad_mu0.x.scatter_forward()

        NSCH.solvector0.x.array[NSCH.maps[5]] = NSCH.project(mu0, NSCH.spaces[5]).x.array
        NSCH.solvector0.x.scatter_forward()
        p0d_mu0 = NSCH.p0d_mu0 = NSCH.solvector0.sub(5)

    def variational_problem(self):
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

        delta = NSCH.delta
        p_unique = NSCH.p_unique

        u, p, phi, mu, p1c_phi, p0d_mu = NSCH.u, NSCH.p, NSCH.phi, NSCH.mu, NSCH.p1c_phi, NSCH.p0d_mu
        ub, pb, phib, mub, p1c_phib, p0d_mub = NSCH.ub, NSCH.pb, NSCH.phib, NSCH.mub, NSCH.p1c_phib, NSCH.p0d_mub
        u0, p0, phi0, mu0, p1c_phi0, p0d_mu0 = NSCH.u0, NSCH.p0, NSCH.phi0, NSCH.mu0, NSCH.p1c_phi0, NSCH.p0d_mu0
        p1c_grad_mu0 = NSCH.p1c_grad_mu0

        pos = NSCH.pos
        neg = NSCH.neg

        def M(phi):
            """Mobility function"""
            return (1 - phi**2)
        def Mpos(phi):
            """Positive part of mobility function"""
            return pos(M(phi))
        def Mup(phi):
            """Increasing part of Mpos"""
            return Mpos(1.0 / 2.0 * (phi - abs(phi)))
        def Mdown(phi):
            """Decreasing part of Mpos"""
            return (Mpos(1.0 / 2.0 * (phi + abs(phi))) - Mpos(0))
        
        def f(p1c_phi, p1c_phi0):
            return (2 * p1c_phi + p1c_phi0**3 - 3 * p1c_phi0)
        
        rho = NSCH.rho

        def eta(phi):
            return NSCH.eta_val

        #
        # Variational problem
        #
        e_len = FacetArea(NSCH.mesh)
        n_e = FacetNormal(NSCH.mesh)
        l = 1.0/nx

        def aupw(u, phi, phib):
            # UPW bilinear form
            return (
                pos(inner(avg(u), n_e('+'))) * phi('+') * jump(phib) * dS \
                - neg(inner(avg(u), n_e('+'))) * phi('-') * jump(phib) * dS
            )

        def bupw(p0d_mu, phi, phib):
            # UPW bilinear form
            return (
                pos(jump(p0d_mu)/((2.0*pow(l,2))/(3.0*avg(e_len)))) * pos(Mup(phi('+')) + Mdown(phi('-'))) * jump(phib) * dS \
                - neg(jump(p0d_mu)/((2.0*pow(l,2))/(3.0*avg(e_len)))) * pos(Mup(phi('-')) + Mdown(phi('+'))) * jump(phib) * dS
            )
        
        def ch(phi, p0d_mu, ub):
            # Centered discretization
            return(
                inner(avg(ub), n_e('+')) * avg(phi) * jump(p0d_mu) * dS 
                + div(ub) * phi * p0d_mu * dx
            )

        def sh1(u, u0, p1c_phi, p1c_phi0, mu, ub):
            return(
                1/2 * (
                    inner((rho(p1c_phi) - rho(p1c_phi0))/dt, inner(u,  ub))
                    + inner(div(rho(p1c_phi0)*u0 - rho_dif*Mpos(p1c_phi0) * p1c_grad_mu0), inner(u, ub))
                ) * dx
            )
        
        def sh2(u, phi, p0d_mu, ub):
            return(1/2 * inner(avg(ub), n_e('+')) * sign(inner(avg(u), n_e('+'))) * jump(phi) * jump(p0d_mu) * dS)
        def sh2d(u, phi, p0d_mu, ub, delta = 1e-10):
            return(1/2 * inner(avg(ub), n_e('+')) * inner(avg(u), n_e('+'))/(abs(inner(avg(u), n_e('+'))) + delta) * jump(phi) * jump(p0d_mu) * dS)

        NSCH.a_u = a_u = inner(rho(p1c_phi0) * u, ub) * dx \
                - inner(rho(p1c_phi0) * u0, ub) * dx \
                + dt * inner(dot(rho(p1c_phi0) * u0 - rho_dif * Mpos(p1c_phi0) * p1c_grad_mu0, nabla_grad(u)), ub) * dx \
                + dt * 2 * eta(phi0) * inner(sym(grad(u)), sym(grad(ub))) * dx \
                - dt * inner(p, div(ub)) * dx \
                - dt * ch(phi, p0d_mu, ub) \
                + dt * sh1(u, u0, p1c_phi, p1c_phi0, mu0, ub) \
                - dt * sh2d(u, phi, p0d_mu, ub, delta)
        
        if params.dim == '3d':
            raise(Exception("Change how to deal with boundary conditions in 3D"))
        else:
            if NSCH.parameters.test == "rayleigh_std":
                def no_slip(x):
                    return (np.logical_or(np.isclose(x[0], 0), np.logical_or(np.isclose(x[0], 1), np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 4)))))
            else:
                def no_slip(x):
                    return (np.logical_or(np.isclose(np.abs(x[0]), 0.5), np.isclose(np.abs(x[1]), 0.5)))
            u_bc = Function(NSCH.spaces[0])
            u_bc.x.set(0)
            facets = dolfinx.mesh.locate_entities_boundary(NSCH.mesh, 1, no_slip)
            NSCH.bcu = bcu = dolfinx.fem.dirichletbc(
                u_bc, dolfinx.fem.locate_dofs_topological((NSCH.Wh.sub(0), NSCH.spaces[0]), 1, facets), NSCH.Wh.sub(0))

        if int(params.gravity) and params.dim=='2d':
            a_u += + rho(phi) *  inner(Constant(NSCH.mesh, PETSc.ScalarType((0.0, 1))), ub) * dx
        elif int(params.gravity) and params.dim=='3d':
            a_u += + rho(phi) *  inner(Constant(NSCH.mesh, PETSc.ScalarType((0.0, 0.0, 1))), ub) * dx
        
        NSCH.a_p = a_p = inner(div(u), pb) * dx \
                        + p_unique *  inner(p, pb) * dx

        NSCH.a_phi = a_phi = inner(phi, phib) * dx \
                - inner(phi0, phib) * dx \
                + dt * bupw(p0d_mu, phi, phib) \
                + dt * aupw(u, phi, phib)
        
        NSCH.a_mu = a_mu = lamb * eps * inner(grad(p1c_phi), grad(mub)) * dx \
                + lamb/eps * inner(f(p1c_phi, p1c_phi0), mub) * dx \
                - inner(mu, mub) * dx_ML
        
        NSCH.a_p1c_phi = a_p1c_phi = inner(phi, p1c_phib) * dx \
                    - inner(p1c_phi, p1c_phib) * dx_ML
        
        NSCH.a_p0d_mu = a_p0d_mu = inner(mu, p0d_mub) * dx \
                    - inner(p0d_mu, p0d_mub) * dx

        NSCH.F = a_u + a_p + a_phi + a_mu + a_p1c_phi + a_p0d_mu

    def create_system(self):
        #
        # Load variables from NSCH problem
        #
        NSCH = self
        params = NSCH.parameters

        #
        # Initialization
        #
        NSCH.solvector.x.array[:] = NSCH.solvector0.x.array
        NSCH.solvector.x.scatter_forward()

        #
        # Define problem
        #

        # Newton
        problem = NSCH.problem = NonlinearProblem(NSCH.F, NSCH.solvector, bcs=[NSCH.bcu])

        solver = NSCH.solver = NewtonSolver(MPI.COMM_WORLD, problem)

        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        # solver.atol = 1e-9
        solver.max_it = 100
        # solver.relaxation_parameter = 1e-3
        # solver.report=True

        # We can customize the linear solver used inside the NewtonSolver by
        # modifying the PETSc options
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        if int(params.verbosity):
            opts[f"{option_prefix}verbose"] = True
        ksp.setFromOptions()
    
    def time_iterator(self, tsteps=1, first_step=1, verbosity=0):
        """Time iterator"""
        NSCH = self
        params = NSCH.parameters

        solver = NSCH.solver

        #
        # Run time iterations
        #
        step = first_step - 1
        last_step = first_step + tsteps

        while step < last_step:
            if step == first_step - 1:
                u0, p0, phi0, mu0, p1c_phi0, p0d_mu0 = NSCH.solvector0.split()

                # --- Yield initial data
                yield {'step': step, 't': NSCH.t, 'u':u0.collapse(), 'p':p0.collapse(), 'phi': phi0.collapse(), 'mu':mu0.collapse(), 'p1c_phi':p1c_phi0.collapse(), 'p0d_mu':p0d_mu0.collapse()}

            else:
                NSCH.t += NSCH.dt

                # Solve

                # Newton
                solver.solve(NSCH.solvector)
                NSCH.solvector.x.scatter_forward()

                # --- Save solution (to be used in next iteration)
                u, p, phi, mu, p1c_phi, p0d_mu = NSCH.solvector.split()
                u0, p0, phi0, mu0, p1c_phi0, p0d_mu0 = NSCH.solvector0.split()

                # --- Yield data computed in current iteration
                yield {'step': step, 't': NSCH.t, 'u':u.collapse(), 'p':p.collapse(), 'phi': phi.collapse(), 'mu':mu.collapse(), 'p1c_phi':p1c_phi.collapse(), 'p0d_mu':p0d_mu.collapse(), 'u0':u0.collapse(), 'p0':p0.collapse(), 'phi0': phi0.collapse(), 'mu0':mu0.collapse(), 'p1c_phi0':p1c_phi0.collapse(), 'p0d_mu0':p0d_mu0.collapse()}

                # --- Update solution
                NSCH.solvector0.x.array[:] = NSCH.solvector.x.array
                NSCH.solvector0.x.scatter_forward()
                NSCH.p1c_grad_mu0.x.array[:] = NSCH.project(grad(mu), NSCH.P1cvecs).x.array
                NSCH.p1c_grad_mu0.x.scatter_forward()

            step = step + 1


# ---------------------------

def print_info(i, t, phi_data, p1c_phi_data, p_data, energy, dynamics=0):
# energy, dynamics = 0):
    phi_max, phi_min, phi_mass = phi_data
    p1c_phi_max, p1c_phi_min, p1c_phi_mass = p1c_phi_data
    p_max, p_min, p_mass = p_data
    s = f"{i:3} {t:.6e} {phi_max:.4e} {p1c_phi_max:.4e} {NSCH.rho(phi_max):.4e} {NSCH.rho(p1c_phi_max):.4e} {p_max:.4e}"
    s += f" {phi_min:.4e} {p1c_phi_min:.4e} {NSCH.rho(phi_min):.4e} {NSCH.rho(p1c_phi_min):.4e} {p_min:.4e}"
    s += f" {phi_mass:.4e} {p1c_phi_mass:.4e} {NSCH.rho_dif * phi_mass:.4e} {NSCH.rho_dif * p1c_phi_mass:.4e} {p_mass:.4e}"
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

    parser.add_argument('--delta', default=1e-6)
    parser.add_argument('--p_unique', default=1e-10)

    parser.add_argument('--dim', choices=['2d', '3d'], default='2d')
    parser.add_argument('--test', choices=['bubble', 'rayleigh', 'circle', 'order'], default='bubble')
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
    NSCH.variational_problem()
    NSCH.create_system()

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
        phi, p1c_phi, mu, p0d_mu, u, p = t_step['phi'], t_step['p1c_phi'], t_step['mu'], t_step['p0d_mu'], t_step['u'], t_step['p']

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
            topology, cell_types, geometry = plot.create_vtk_mesh(NSCH.P1cs)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            aux = p1c_phi.x.array
            aux[np.abs(aux + 1.0) < 1e-16] = -1.0
            aux[np.abs(aux - 1.0) < 1e-16] = 1.0
            grid.point_data["Pi1_phi"] = aux

            grid.set_active_scalars("Pi1_phi")

            # Velocity field
            u_values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
            u_values[:, :len(u)] = NSCH.project(u, NSCH.P1cvecs).x.array.real.reshape((geometry.shape[0], len(u)))

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
            adx.write_function(phi, f"{base_name_save}_phi_i-{i}")
            adx.write_function(p1c_phi, f"{base_name_save}_p1c_phi_i-{i}")
            adx.write_function(u, f"{base_name_save}_u_i-{i}")
            adx.write_function(p, f"{base_name_save}_p_i-{i}")

            with dolfinx.io.XDMFFile(comm, f"{base_name_save}.xdmf", "a") as xdmf:
                phi.name = "phi"
                p1c_phi.name = "p1c_phi"
                u.name = "u"
                p.name = "p"
                xdmf.write_function(phi, t=t)
                xdmf.write_function(p1c_phi, t=t)
                xdmf.write_function(u, t=t)
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
