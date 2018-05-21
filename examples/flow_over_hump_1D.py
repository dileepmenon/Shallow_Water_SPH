# Numpy
from numpy import ( ones_like, zeros, zeros_like, mgrid, pi,
                    arange, sqrt, concatenate, sin, cos, where )

# PySPH base
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator_step import IntegratorStep

# PySPH equations
from pysph.sph.equation import Group
from equation.SWE import *
from equation.SWE import get_particle_array_swe as gpa_swe

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
fluid_surf_hei = 0.4
hdx = 1.2
dx = 0.005
dim = 1


class Flow_Over_Hump(Application):
    def create_particles(self):
        # 1D
        # Bed
        dxb = dx
        xb = arange(0, 1+1e-4, dxb)
        cond = (0.25<xb) & (xb<0.75)
        b = where(cond, 0.05*(1+sin(pi*(4*xb+0.5))), 0)

        Vb = ones_like(xb) * dxb
        hb = ones_like(xb) * hdx * dxb

        bed = gpa_swe(name='bed', x=xb, V=Vb, b=b, h=hb)

        # Fluid
        x = arange(0, 1+1e-4, dx)
        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        fluid = gpa_swe(x=x, h=h, h0=h0, name='fluid')
        compute_fluid_elevation([fluid, bed])

        dw = fluid_surf_hei - fluid.b
        rho = dw * rho_w
        rho0 = dw * rho_w
        m = rho * dx

        fluid.m = m
        fluid.rho = rho
        fluid.rho0 = rho0
        fluid.dw = dw

        compute_initial_props([fluid])

        # Boundary
        x_cb = np.array([-2*dx, -dx, 1+dx, 1+2*dx])
        m_cb = ones_like(x_cb) * rho_w * dx * fluid_surf_hei
        h_cb = ones_like(x_cb) * hdx * dx
        rho_cb = ones_like(x_cb) * rho_w * fluid_surf_hei
        dw_cb = ones_like(x_cb) * fluid_surf_hei
        cs_cb = sqrt(9.8 * dw_cb)
        alpha_cb = dim * rho_cb

        boundary = gpa_swe(name='boundary', x=x_cb, m=m_cb,
                           h=h_cb, rho=rho_cb, dw=dw_cb, cs=cs_cb,
                           alpha=alpha_cb)

        return [fluid, bed, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=1)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 5
        solver = Solver(
            kernel=kernel,
            dim=1,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            GatherDensityEvalNextIteration(
                                dest='fluid', sources=['fluid', 'boundary']
                                ),
                            ]
                        ),
                    Group(
                        equations=[
                            NonDimensionalDensityResidual(dest='fluid')
                            ]
                        ),
                    Group(
                        equations=[
                            UpdateSmoothingLength(dim, dest='fluid')
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            CheckConvergenceDensityResidual(dest='fluid')
                            ],
                    )], iterate=True, max_iterations=10
            ),
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(
                        dest='fluid', sources=['fluid', 'boundary']),
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ]
                ),
            Group(
                equations=[
                    FluidBottomElevation(dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    FluidBottomGradient(dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(dim, dest='fluid', sources=['fluid',
                                         'boundary'], u_only=True),
                    ],
                ),
            ]
        return equations


def compute_fluid_elevation(particles):
    one_time_equations = [
       Group(
            equations=[
                FluidBottomElevation(dest='fluid', sources=['bed'])
                ]
            ),
       Group(
            equations=[
                BedGradient(dest='bed', sources=['bed']),
                ]
            ),
    ]
    kernel = CubicSpline(dim=1)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=1,
                            kernel=kernel)
    sph_eval.evaluate()


def compute_initial_props(particles):
    one_time_equations = [
        Group(
            equations=[
                SWEOS(dest='fluid')
                    ]
            ),
    ]
    kernel = CubicSpline(dim=1)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=1,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = Flow_Over_Hump()
    app.run()
