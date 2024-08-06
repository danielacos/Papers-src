#
# Tumor Test
# ===================
# FEniCSx version: 0.6
#

import dolfinx
from dolfinx.fem import (
    Expression, Function, FunctionSpace,
    assemble_scalar, form, petsc
)
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import log
from ufl import(
     TestFunction, TrialFunction, FiniteElement, MixedElement,
     SpatialCoordinate,
     dx, dS, inner, grad, div, avg, jump,
     sqrt, tanh,
     split,
     FacetArea, # FacetNormal 
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
# dx_ML = dx

#
# Problem class
#
class Tumor_DG_UPW(object):
    
    def __init__(self, tumor_parameters):
        #
        # Load PDE and discretization parameters
        #
        tumor = self
        p = tumor.parameters = tumor_parameters
        tumor.eps = float(p.eps)
        tumor.delta = float(p.delta)
        tumor.chi0 = float(p.chi0)
        tumor.P0 = float(p.P0)
        tumor.Cu = float(p.Cu)
        tumor.Cn = float(p.Cn)
        tumor.tau = float(p.tau)

        file_path = os.path.dirname(os.path.abspath(__file__))
        printMPI(f"file_path = {file_path}")
        mesh_file = f"{file_path}/meshes/" + f"mesh_big_square_nx-{p.nx}.xdmf"
        printMPI(f"mesh_file = {mesh_file}")

        #
        # Read mesh
        #
        # mesh = tumor.mesh = Mesh(mesh_file)
        # tumor.mesh = Mesh()
        with XDMFFile(comm, mesh_file, 'r') as infile:
            mesh = tumor.mesh = infile.read_mesh()
        
        tumor.nx = int(p.nx)
        tumor.dt = float(p.dt)
        tumor.t = 0.

        #
        # Build DG, FE spaces and functions
        #
        tumor.P0d = FiniteElement("DG", mesh.ufl_cell(), 0)
        tumor.P1c = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        tumor.Uh = FunctionSpace(mesh, tumor.P0d)
        tumor.Vh1 = FunctionSpace(mesh, tumor.P1c)
        tumor.Wh = FunctionSpace(mesh, MixedElement([tumor.P0d, tumor.P1c, tumor.P1c, tumor.P0d, tumor.P0d]))
        tumor.solvector, tumor.testvector = Function(tumor.Wh), TestFunction(tumor.Wh)
        tumor.solvector0 = Function(tumor.Wh)

        # Compute subspaces and maps from subspaces to main space in MixedElement space
        tumor.num_subs = tumor.Wh.num_sub_spaces
        tumor.spaces = []
        tumor.maps = []
        for i in range(tumor.num_subs):
            space_i, map_i = tumor.Wh.sub(i).collapse()
            tumor.spaces.append(space_i)
            tumor.maps.append(map_i)

        #
        # Variables to store solution at two time steps
        #
        u, muu, wu, pi0muu, n = tumor.u, tumor.muu, tumor.wu, tumor.pi0muu, tumor.n = split(tumor.solvector)
        #
        # Test variables
        #
        ub, muub, wub, pi0muub, nb = tumor.ub, tumor.muub, tumor.wub, tumor.pi0muub, tumor.nb = split(tumor.testvector)

    def project(self, u, space, mass_lumping=False):
        tumor = self

        Piu_trial = TrialFunction(space)
        Piub = TestFunction(space)

        if mass_lumping:
            a = inner(Piu_trial, Piub) * dx_ML
        else:
            a = inner(Piu_trial, Piub) * dx

        L = inner(u, Piub) * dx

        problem = petsc.LinearProblem(a, L)
        return problem.solve()

    def load_initial_values(self):
        """Initialize variables"""
        tumor = self
        eps = tumor.eps

        #
        # Initial condition
        #
        x = SpatialCoordinate(tumor.mesh)
        u0_dict = {
            "three_tumors": (1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 2, 2) + pow(x[1] - 2, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 3, 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 1.0/2.0 * (tanh((1.73 - sqrt(pow(x[0] + 1.5,2) + pow(x[1] + 1.5, 2)))/(sqrt(2.0) * eps)) + 1.0)),
            # "single_tumor": (1.0/2.0 * (tanh((1.73 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * 0.1 * eps)) + 1.0))
            "single_tumor": (1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * eps)) + 1.0))

            }
        
        n0_dict = {
            "three_tumors": (1.0 -
                    1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 2, 2) + pow(x[1] - 2, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    - 1.0/2.0 * (tanh((1 - sqrt(pow(x[0] - 3, 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    - 1.0/2.0 * (tanh((1.73 - sqrt(pow(x[0] + 1.5,2) + pow(x[1] + 1.5, 2)))/(sqrt(2.0) * eps)) + 1.0)),
            # "single_tumor": (1.0/2.0 * (tanh((1.73 - sqrt(pow(x[0] - 3.5, 2) + pow(x[1] - 3.5, 2)))/(sqrt(2.0) * 0.1 * eps)) + 1.0)
            #         + 1.0/2.0 * (tanh((1.73 - sqrt(pow(x[0] + 3.5, 2) + pow(x[1] - 3.5, 2)))/(sqrt(2.0) * 0.1 * eps)) + 1.0)
            #         + 1.0/2.0 * (tanh((2.5 - sqrt(pow(x[0], 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * 0.1 * eps)) + 1.0))
            # "single_tumor": (0.5 * (1.0 - 1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * eps)) + 1.0))  + 0.5 * 1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0] - 3.0, 2) + pow(x[1] - 2.0, 2)))/(sqrt(2.0) * eps)) + 1.0)
            #         + 0.5 * 1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0] + 3.0, 2) + pow(x[1] - 2.0, 2)))/(sqrt(2.0) * eps)) + 1.0)
            #         + 0.5 * 1.0/2.0 * (tanh((2.5 - sqrt(pow(x[0], 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * eps)) + 1.0))
            "single_tumor": (0.5 * (1.0 - 1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0], 2) + pow(x[1], 2)))/(sqrt(2.0) * eps)) + 1.0))  + 0.5 * 1.0/2.0 * (tanh((1.0 - sqrt(pow(x[0] - 2.45, 2) + pow(x[1] - 1.45, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 0.5 * 1.0/2.0 * (tanh((1.75 - sqrt(pow(x[0] + 3.75, 2) + pow(x[1] - 1.0, 2)))/(sqrt(2.0) * eps)) + 1.0)
                    + 0.5 * 1.0/2.0 * (tanh((2.5 - sqrt(pow(x[0], 2) + pow(x[1] + 5, 2)))/(sqrt(2.0) * eps)) + 1.0))
            }
        

        print("u0 =", str(u0_dict[p.initial_cond]))
        print("n0 =", str(n0_dict[p.initial_cond]))

        u_init = Expression(u0_dict[p.initial_cond], tumor.Wh.sub(0).element.interpolation_points())
        n_init = Expression(n0_dict[p.initial_cond], tumor.Wh.sub(4).element.interpolation_points())

        # Initial values
        tumor.solvector0.sub(0).interpolate(u_init)
        tumor.solvector0.x.scatter_forward()
        u0 = tumor.solvector0.sub(0)

        tumor.solvector0.sub(4).interpolate(n_init)
        tumor.solvector0.x.scatter_forward()
        n0 = tumor.solvector0.sub(4)

        # Load w0 using mass-lumping
        tumor.solvector0.sub(2).x.array[tumor.maps[2]] = tumor.project(u0, tumor.spaces[2], mass_lumping=True).x.array
        tumor.solvector0.x.scatter_forward()
        wu0 = tumor.solvector0.sub(2)

        dF_wu = lambda wu: 2 * 0.25 * (wu * pow(wu - 1, 2) + pow(wu, 2) * (wu - 1))
        tumor.solvector0.sub(1).x.array[tumor.maps[1]] = tumor.project(dF_wu(wu0) - pow(tumor.eps, 2) * div(grad(wu0)) - tumor.chi0 * n0, tumor.spaces[1]).x.array
        tumor.solvector0.x.scatter_forward()
        muu0 = tumor.solvector0.sub(1)

        tumor.solvector0.sub(3).x.array[tumor.maps[3]] = tumor.project(muu0, tumor.spaces[3]).x.array
        tumor.solvector0.x.scatter_forward()
        pi0muu0 = tumor.solvector0.sub(3)

        tumor.u0, tumor.muu0, tumor.wu0, tumor.pi0muu0, tumor.n0 = split(tumor.solvector0)

    def variational_problem(self):
        """Build variational problem"""
        #
        # Load variables from tumor problem
        #
        tumor = self
        dt = tumor.dt
        nx = tumor.nx
        u, ub = tumor.u, tumor.ub
        muu, muub = tumor.muu, tumor.muub
        wu, wub = tumor.wu, tumor.wub
        pi0muu, pi0muub = tumor.pi0muu, tumor.pi0muub
        n, nb = tumor.n, tumor.nb
        u0, muu0, wu0, n0 = tumor.u0, tumor.muu0, tumor.wu0, tumor.n0
        eps, delta, chi0, Cu, Cn, P0, tau = tumor.eps, tumor.delta, tumor.chi0, tumor.Cu, tumor.Cn, tumor.P0, tumor.tau

        def pos(u):
            return ((abs(u) + u) / 2.0)
        tumor.pos = pos
        def neg(u):
            return ((abs(u) - u) / 2.0)

        symmetric = int(tumor.parameters.symmetric)
        if symmetric:
            def M(u):
                """Mobility function"""
                return (u * (1.0 - u))/(1/2 * (1 - 1/2))
            def Mpos(u):
                """Positive part of mobility function"""
                return pos(M(u))
            def Mup(u):
                """Increasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (u + 1.0 / 2.0 - abs(u - 1.0 / 2.0)))
            def Mdown(u):
                """Decreasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (u + 1.0 / 2.0 + abs(u - 1.0 / 2.0))) - Mpos(1.0 / 2.0)
            def P(u):
                return Mpos(u)
        else:
            def M(u):
                """Mobility function"""
                return (u**5 * (1.0 - u))/((5/6)**5 * (1.0 - 5/6))
            def Mpos(u):
                """Positive part of mobility function"""
                return pos(M(u))
            def Mup(u):
                """Increasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (u + 5.0 / 6.0 - abs(u - 5.0 / 6.0)))
            def Mdown(u):
                """Decreasing part of Mpos"""
                return Mpos(1.0 / 2.0 * (u + 5.0 / 6.0 + abs(u - 5.0 / 6.0))) - Mpos(5.0 / 6.0)
            def P(u):
                return pos(u * (1.0 - u)**3)/(1/4 * (1.0 - 1/4)**3)
    
        #
        # PDE functions
        #
        tumor.F_wu = lambda wu: 0.25 * (pow(wu,2)) * (pow(wu-1,2))
        f_im = tumor.f_im = 3.0 * 0.25 * wu
        f_ex = tumor.f_ex = 0.25 * (4.0*(pow(wu0,3)) - 6.0*(pow(wu0,2)) - wu0)
        
        # mun = tumor.mun = Function(tumor.Uh)
        # mun.x.array[:] = (1.0/delta * tumor.solvector.sub(4).x.array[tumor.maps[4]] - chi0*tumor.project(wu0, tumor.Uh).x.array)
        # mun.x.scatter_forward()

        # mun = tumor.mun = 1.0/delta * n
        
        # mun = tumor.mun = Function(tumor.Uh)
        # mun.x.array[:] = 1.0/delta * tumor.solvector.x.array[tumor.maps[4]]
        # mun.x.scatter_forward()

        def mun(n, wu0):
            return (1/delta * n - chi0*tumor.project(wu0, tumor.Uh))
        
        tumor.mun = mun

        #
        # Variational problem
        #
        e_len = FacetArea(tumor.mesh)
        l = 20.0/nx

        def aupw(muu, u, ub):
            # UPW bilinear form
            return (
                pos(jump(muu)/((2.0*pow(l,2))/(3.0*avg(e_len)))) * pos(Mup(u('+')) + Mdown(u('-'))) * jump(ub) * dS \
                - neg(jump(muu)/((2.0*pow(l,2))/(3.0*avg(e_len)))) * pos(Mup(u('-')) + Mdown(u('+'))) * jump(ub) * dS
            )

        a1 = inner(u, ub) * dx \
            + dt * Cu * aupw(pi0muu, u, ub) \
            - dt * delta * P0 * inner(P(u) * pos(n) * pos(mun(n, wu0) - pi0muu), ub) * dx
        L1 = inner(u0, ub) * dx

        a2 = inner(muu, muub) * dx_ML \
            - pow(eps, 2) * inner(grad(wu), grad(muub)) * dx \
             - inner(f_im, muub) * dx \
             + chi0 * inner(n, muub) * dx
        L2 = inner(f_ex, muub) * dx

        a3 = inner(wu, wub) * dx_ML \
             - inner(u, wub) * dx

        a4 = inner(pi0muu, pi0muub) * dx \
             - inner(muu, pi0muub) * dx

        a5 = tau * inner(n, nb) * dx \
            + dt * Cn * aupw(mun(n, wu0), n, nb) \
            + dt * delta * P0 * inner(P(u) * pos(n) * pos(mun(n, wu0) - pi0muu), nb) * dx
        L5 = tau * inner(n0, nb) * dx

        a = a1 + a2 + a3 + a4 + a5
        L = L1 + L2 + L5

        tumor.F = a - L

    def time_iterator(self, tsteps=1, first_step=1, verbosity=0):
        """Time iterator"""
        #
        # Load variables from tumor problem
        #
        tumor = self

        #
        # Run time iterations
        #
        step = first_step - 1

        #
        # Initialization
        #
        tumor.solvector.x.array[:] = tumor.solvector0.x.array
        tumor.solvector.x.scatter_forward()

        last_step = first_step + tsteps

        while step < last_step:
            if step == first_step - 1:
                u0, muu0, wu0, pi0muu0, n0 = tumor.solvector0.split()
                # --- Yield initial data
                yield {'step': step, 't': tumor.t, 'u': u0, 'muu': muu0, 'wu': wu0, 'pi0muu': pi0muu0, 'n': n0, 'mun': tumor.mun(n0,wu0)}

            else:
                tumor.t += tumor.dt

                # Create nonlinear problem and Newton solver
                problem = NonlinearProblem(tumor.F, tumor.solvector)
                solver = NewtonSolver(MPI.COMM_WORLD, problem)
                solver.convergence_criterion = "incremental"
                solver.rtol = 1e-6
                solver.max_it = 1000
                # solver.report=True

                # We can customize the linear solver used inside the NewtonSolver by
                # modifying the PETSc options
                ksp = solver.krylov_solver
                opts = PETSc.Options()
                option_prefix = ksp.getOptionsPrefix()
                if int(p.verbosity):
                    opts[f"{option_prefix}verbose"] = True
                # opts[f"{option_prefix}ksp_type"] = "preonly"
                # opts[f"{option_prefix}pc_type"] = "lu"
                # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
                # opts[f"{option_prefix}ksp_type"] = "gmres"
                # opts[f"{option_prefix}pc_type"] = "gamg"
                # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
                ksp.setFromOptions()

                solver.solve(tumor.solvector)
                tumor.solvector.x.scatter_forward()
# -

                # --- Save solution (to be used in next iteration)
                u, muu, wu, pi0muu, n = tumor.solvector.split()
                u0, muu0, wu0, pi0muu0, n0 = tumor.solvector0.split()

                # --- Yield data computed in current iteration
                yield {'step': step, 't': tumor.t, 'u': u, 'muu': muu, 'wu': wu, 'pi0muu': pi0muu, 'n': n, 'mun':mun, 'u0': u0, 'muu0': muu0, 'wu0': wu0, 'n0': n0}

                # --- Update solution
                tumor.solvector0.x.array[:] = tumor.solvector.x.array
                tumor.solvector0.x.scatter_forward()

                # tumor.mun.x.array[:] = (1/tumor.delta * tumor.solvector.sub(4).x.array[tumor.maps[4]] - tumor.chi0*tumor.project(wu0, tumor.Uh).x.array)
                # tumor.mun.x.scatter_forward()

            step = step + 1


# ---------------------------

def print_info(i, t, u_data, wu_data, n_data, energy, dynamics = 0):
    u_max, u_min, u_n_mass = u_data
    wu_max, wu_min, wu_n_mass = wu_data
    n_max, n_min = n_data
    s = f"{i:3} {t:.6e} {u_max:.4e} {wu_max:.4e} {n_max:.4e}"
    s += f" {u_min:.4e} {wu_min:.4e} {n_min:.4e}"
    s += f" {u_n_mass:.4e} {wu_n_mass:.4e}"
    s += f" {energy:.4e}"
    if dynamics:
        dynamics_u, dynamics_wu, dynamics_n = dynamics
        s += f" {dynamics_u:.4e} {dynamics_wu:.4e} {dynamics_n:.4e}"
    printMPI(s)


def define_parameters():

    parser = argparse.ArgumentParser()

    # Define remaining parameters
    parser.add_argument('--eps', default=0.1)
    parser.add_argument('--delta', default=0.01)
    parser.add_argument('--chi0', default=1.e-14)
    parser.add_argument('--P0', default=300.0)
    parser.add_argument('--Cu', default=1.0)
    parser.add_argument('--Cn', default=1.0)
    parser.add_argument('--tau', default=1.0)
    parser.add_argument('--initial_cond', default='three_tumors')
    parser.add_argument('--symmetric', default=0)

    parser.add_argument('--test', choices=['2d', '3d'], default='2d')
    # parser.add_argument('--mesh', default="mesh_big_square_nx-50.xdmf")

    # Params for the discrete stumoreme
    parser.add_argument('--nx', default=50)
    parser.add_argument('--dt', default=0.2)
    parser.add_argument('--tsteps', default=250)

    # Other parameters
    parser.add_argument('--verbosity', default=0, help="No extra information shown")
    parser.add_argument('--plot', default=0, help="Plot shown every number of time steps")
    parser.add_argument('--plot_mesh', default=0, help="Plot mesh")
    parser.add_argument('--vtk', default=0, help="No vtk photogram saved to disk")
    parser.add_argument('--vtkfile', default="tumor_DG-UPW", help="Name of vtk file")
    parser.add_argument('--save', default=0, help="No figures and output saved")
    parser.add_argument('--savefile', default="tumor_DG-UPW", help="Name of output file")
    parser.add_argument('--savefunc', default=0, help="No functions saved in HDF5 format")
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
    p = parameters = define_parameters()
    printMPI("Parameters:")
    for k, v in vars(parameters).items():
        printMPI(f"  {k} = {v}")
    
    if int(p.verbosity):
        log.set_log_level(log.LogLevel.INFO)

    #
    # Init problem
    #
    tumor = Tumor_DG_UPW(parameters)
    tumor.load_initial_values()
    tumor.variational_problem()

    #
    # More info
    #  
    printMPI("More info:")
    tdim = tumor.mesh.topology.dim
    num_cells = tumor.mesh.topology.index_map(tdim).size_local
    h = dolfinx.cpp.mesh.h(tumor.mesh, tdim, range(num_cells))
    printMPI("  Mesh h    = " + str(max(h)))

    #
    # Save max, min and energy
    #
    max_u_list = []
    min_u_list = []
    max_wu_list = []
    min_wu_list = []
    max_n_list = []
    min_n_list = []
    E = []
    dynam_u_list = []
    dynam_wu_list = []
    dynam_n_list = []

    #
    # Print info
    #
    printMPI("Time steps:\n  i t u_max wu_max n_max u_min wu_min n_min u_n_mass wu_n_mass energy dynam_u dynam_wu dynam_n")

    #
    # Time iterations
    #
    tumor_iterations = tumor.time_iterator(tsteps=int(p.tsteps), verbosity=bool(p.verbosity))
    
    for t_step in tumor_iterations:

        i, t = t_step['step'], t_step['t']
        u, wu, n = t_step['u'], t_step['wu'], t_step['n']
        pi0muu, mun = t_step['pi0muu'], t_step['mun']

        #
        # Save output
        #
        do_save = bool(p.save)
        server = bool(p.server)
        base_name_save = p.savefile

        #
        # Print info
        #
        u_max, u_min = max(u.x.array[tumor.maps[0]]), min(u.x.array[tumor.maps[0]])
        wu_max, wu_min = max(wu.x.array[tumor.maps[2]]), min(wu.x.array[tumor.maps[2]])
        n_max, n_min = max(n.x.array[tumor.maps[4]]), min(n.x.array[tumor.maps[4]])
        u_n_mass, wu_n_mass = assemble_scalar(form((u + n)*dx)), assemble_scalar(form((wu + n)*dx))
        energy = assemble_scalar(form(
            0.5 * pow(tumor.eps,2) * inner(grad(wu), grad(wu)) * dx \
            + tumor.F_wu(wu) * dx \
            - tumor.chi0 * wu * n * dx \
            + 0.5 * 1/tumor.delta * pow(n, 2) * dx
        ))
        max_u_list.append(u_max)
        min_u_list.append(u_min)
        max_wu_list.append(wu_max)
        min_wu_list.append(wu_min)
        max_n_list.append(n_max)
        min_n_list.append(n_min)
        E.append(energy)

        if t>DOLFIN_EPS:
            u0, wu0, n0 = t_step['u0'], t_step['wu0'], t_step['n0']

            dynamics_u = np.abs(u.x.array[tumor.maps[0]]-u0.x.array[tumor.maps[0]]).max()/np.abs(u0.x.array[tumor.maps[0]]).max()
            dynamics_wu = np.abs(wu.x.array[tumor.maps[2]]-wu0.x.array[tumor.maps[2]]).max()/np.abs(wu0.x.array[tumor.maps[2]]).max()
            dynamics_n = np.abs(n.x.array[tumor.maps[4]]-n0.x.array[tumor.maps[4]]).max()/np.abs(n0.x.array[tumor.maps[4]]).max()

            dynam_u_list.append(dynamics_u)
            dynam_wu_list.append(dynamics_wu)
            dynam_n_list.append(dynamics_n)

            print_info(i, t,
                    (u_max, u_min, u_n_mass),
                    (wu_max, wu_min, wu_n_mass),
                    (n_max, n_min),
                    energy,
                    (dynamics_u, dynamics_wu, dynamics_n))
        
        else:
            print_info(i, t,
                    (u_max, u_min, u_n_mass),
                    (wu_max, wu_min, wu_n_mass),
                    (n_max, n_min),
                    energy)

        #
        # Plot
        #
        if server:
            pyvista.start_xvfb()
        if do_save:
            pyvista.OFF_SCREEN = True

        do_plot = (int(p.plot) > 0)
        plot_mesh = (int(p.plot_mesh) > 0)
        pyvista.set_plot_theme("document")

        if plot_mesh: # Plot mesh
            topology, cell_types, geometry = plot.create_vtk_mesh(tumor.mesh, tumor.mesh.topology.dim)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=True, color="white")
            plotter.view_xy()
            if pyvista.OFF_SCREEN:
                plotter.screenshot("mesh.png", transparent_background=True)

                img = Image.open(f"mesh.png")
                width, height = img.size
                # Setting the points for cropped image
                left = width/6
                top = 0.08 * height
                right = 5 * width/6
                bottom = 0.92 * height
                im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                im_cropped.save(f"mesh.png")
            else:
                plotter.show()

        if (do_plot and i % int(p.tsteps) % int(p.plot) == 0):  # Plot some steps
            
            # pyvista.global_theme.window_size = [800, 650]

            # Properties of the scalar bar
            sargs = dict(height=0.6, vertical=True, position_x=0.8, position_y=0.2, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2f", font_family="arial")

            # Create a grid to attach the DoF values
            cells, types, x = plot.create_vtk_mesh(tumor.spaces[2])
            grid = pyvista.UnstructuredGrid(cells, types, x)
            aux = wu.x.array[tumor.maps[2]]
            aux[np.abs(aux) < 1e-16] = 0
            aux[np.abs(aux - 1.0) < 1e-16] = 1.0
            grid.point_data["Pi1_u"] = aux

            grid.set_active_scalars("Pi1_u")
            # warped = grid.warp_by_scalar()

            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["viridis"], scalar_bar_args=sargs)
            # plotter.camera.tight(padding=0.0)
            # plotter.camera.zoom(100.0)
            plotter.view_xy()

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                figure = plotter.screenshot(f"./{base_name_save}_Pi1_u_i-{i}.png", transparent_background=True)
                
                img = Image.open(f"./{base_name_save}_Pi1_u_i-{i}.png")
                width, height = img.size
                # Setting the points for cropped image
                left = width/6
                top = 0.08 * height
                right = 0.96 * width
                bottom = 0.92 * height
                im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                im_cropped.save(f"./{base_name_save}_Pi1_u_i-{i}.png")
            else:
                plotter.show()
            plotter.close()

            wn = tumor.project(n, tumor.Vh1, mass_lumping=True)
            aux = wn.x.array[:]
            aux[np.abs(aux) < 1e-16] = 0
            aux[np.abs(aux - 1.0) < 1e-16] = 1.0
            grid.point_data["Pi1_n"] = aux

            grid.set_active_scalars("Pi1_n")

            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["Reds"], scalar_bar_args=sargs)
            plotter.view_xy()

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                figure = plotter.screenshot(f"./{base_name_save}_Pi1_n_i-{i}.png", transparent_background=True)

                img = Image.open(f"./{base_name_save}_Pi1_n_i-{i}.png")
                width, height = img.size
                # Setting the points for cropped image
                left = width/6
                top = 0.08 * height
                right = 0.96 * width
                bottom = 0.92 * height
                im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                im_cropped.save(f"./{base_name_save}_Pi1_n_i-{i}.png")
            else:
                plotter.show()
            plotter.close()

            dif_pot = tumor.project(n * tumor.pos(mun - pi0muu), tumor.Vh1, mass_lumping=True)
            grid.point_data["Pi1_dif_pot"] = dif_pot.x.array[:]
            grid.set_active_scalars("Pi1_dif_pot")

            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, cmap=mpl.colormaps["cividis"], scalar_bar_args=sargs)
            plotter.view_xy()

            # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
            # otherwise create interactive plot
            if pyvista.OFF_SCREEN:
                figure = plotter.screenshot(f"./{base_name_save}_dif_pot_i-{i}.png", transparent_background=True)

                img = Image.open(f"./{base_name_save}_dif_pot_i-{i}.png")
                width, height = img.size
                # Setting the points for cropped image
                left = width/6
                top = 0.08 * height
                right = 0.96 * width
                bottom = 0.92 * height
                im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                im_cropped.save(f"./{base_name_save}_dif_pot_i-{i}.png")
            else:
                plotter.show()
            plotter.close()

    #
    # Plot
    #
    if do_plot:
        time_steps = np.linspace(0, t, int(p.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(p.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_u_list,'--',c='orange')
        axs[1].plot(time_steps,np.zeros(int(p.tsteps) + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_u_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_u.png")
        else: plt.show()
        plt.close()

        time_steps = np.linspace(0, t, int(p.tsteps) + 1)
        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(p.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_wu_list,'--',c='orange')
        axs[1].plot(time_steps,np.zeros(int(p.tsteps) + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[1].plot(time_steps,min_wu_list,'--',c='orange')
        plt.subplots_adjust(hspace=0.5, bottom=0.16)
        if do_save: plt.savefig(f"{base_name_save}_min-max_wu.png")
        else: plt.show()
        plt.close()

        fig, axs = plt.subplots(2)
        axs[0].plot(time_steps,np.full(int(p.tsteps) + 1, 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
        axs[0].plot(time_steps,max_n_list,'--',c='orange')
        axs[1].plot(time_steps,np.zeros(int(p.tsteps) + 1),'-',c='lightgray',linewidth=2,label='_nolegend_')
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

        plt.plot(np.linspace(0, t, int(p.tsteps)), dynam_wu_list, color='darkblue')
        plt.title("Dynamics u")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_u.png")
        else: plt.show()
        plt.close()

        plt.plot(np.linspace(0, t, int(p.tsteps)), dynam_wu_list, color='darkblue')
        plt.title("Dynamics w")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_wu.png")
        else: plt.show()
        plt.close()

        plt.plot(np.linspace(0, t, int(p.tsteps)), dynam_n_list, color='darkblue')
        plt.title("Dynamics n")
        plt.xlabel("Time")
        plt.ylabel("Dynamics")
        if do_save: plt.savefig(f"{base_name_save}_dynamics_n.png")
        else: plt.show()
        plt.close()
