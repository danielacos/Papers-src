#
# Soving steady Stokes equations
# Laminar flow cavity test
#

# FEniCS version: 2019.1.0

import dolfin as fe
import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from mshr import *

nu = 0.01  # Viscosity coefficient

# domain = Circle(Point(circle_outx,circle_outy),circle_outr) - Circle(Point(circle_inx,circle_iny),circle_inr)
# mesh = generate_mesh ( domain, 30 )
X0, X1 = 0, 2
Y0, Y1 = 0, 1
domain = Rectangle(Point(X0,Y0), Point(X1,Y1))
mesh = generate_mesh ( domain, 30 )
plot(mesh)
plt.show()

with XDMFFile("./mesh-stokes-cavity-test.xdmf") as outfile:
    outfile.write(mesh)

#
#  Declare Finite Element Spaces
#
P2 = VectorElement("P", triangle, 2)  # Taylor-Hood P2/P1 velocity/pressure
P1 = FiniteElement("P", triangle, 1)
TH = P2 * P1
Vh = VectorFunctionSpace(mesh, "P", 2)
Qh = FunctionSpace(mesh, "P", 1)
Wh = fe.FunctionSpace(mesh, TH)

#
#  Declare Finite Element Functions
#
(u, p) = TrialFunctions(Wh)
(v, q) = TestFunctions(Wh)
w = Function(Wh)

#
# Macros needed for weak formulation.
#
def diffusion(u, v):
  return inner(nabla_grad(u), nabla_grad(v))

#
#  Declare Boundary Conditions.
#

# Utop = 1
# Utop = Expression("(x[0]-X0)*(X1-x[0])",X0=X0,X1=X1,deg=2)
# top_velocity = Constant((Utop, 0.0))
top_velocity = Expression(("(x[0]-X0)*(X1-x[0])","0.0"),X0=X0,X1=X1,degree=2)
noslip = Constant((0.0, 0.0))

def top_boundary(x, on_boundary):
  tol = 1E-15
  return on_boundary and (abs(x[1]-Y1) < tol)

def bottom_boundary(x, on_boundary):
  tol = 1E-15
  return on_boundary and (abs(x[1]-Y0) < tol)

def left_boundary(x, on_boundary):
  tol = 1E-15
  return on_boundary and (abs(x[0]-X0) < tol)

def right_boundary(x, on_boundary):
  tol = 1E-15
  return on_boundary and (abs(x[0]-X1) < tol)

bc_top = DirichletBC(Wh.sub(0), top_velocity, top_boundary)

bc_bottom = DirichletBC(Wh.sub(0), noslip, bottom_boundary)
bc_right = DirichletBC(Wh.sub(0), noslip, right_boundary)
bc_left = DirichletBC(Wh.sub(0), noslip, left_boundary)

dirichlet_bcs = [bc_top, bc_bottom, bc_right, bc_left]

#
#  Generate Initial Conditions
#

#  Right Handside
f = Constant((0, 0))

#
#  Set up Stokes problem for IC
#

# AStokes(u,v) = LStokes(v)  forall v test function
LStokes = inner(f, v)*dx
AStokes = (inner(grad(u), grad(v)) - div(v)*p + q*div(u) + 1.e-14*p*q)*dx
solve(AStokes == LStokes, w, dirichlet_bcs)

usol = fe.Function(Vh)
psol = fe.Function(Qh)
fe.assign(usol, w.sub(0))
fe.assign(psol, w.sub(1))

pl = plot(psol)
plt.colorbar(pl)
plt.show()

with XDMFFile("./velocity-stokes-cavity-test.xdmf") as outfile:
    outfile.write_checkpoint(usol, "v", 0, append=False)

u_vtk = File("/tmp/uStokes.pvd")
p_vtk = File("/tmp/pStokes.pvd")
usol.rename("Velocity", "Velocity")
psol.rename("Pressure", "Pressure")
u_vtk << usol
p_vtk << psol
