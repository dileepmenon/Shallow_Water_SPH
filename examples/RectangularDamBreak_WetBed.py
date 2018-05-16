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
hdx = 1.2
d1 = 10.0
d2 = 5.0
dx1 = 5.0
dx2 = 10.0
tot_fluid_len = 2000.0
fluid_col1_len = 1000.0
dim = 1


class RectangularDamBreak(Application):
    def create_particles(self):
        x = concatenate((arange(0, fluid_col1_len, dx1),
                         arange(fluid_col1_len, tot_fluid_len+1e-4, dx2)),
                         axis=0)

        m = ones_like(x)
        h = ones_like(x)
        h0 = ones_like(x)
        rho = ones_like(x)
        rho0 = ones_like(x)

        # Setting first fluid column properties
        idx_fluid_col_1 = where(x < 1000.0)[0]
        m[idx_fluid_col_1] *= dx1 * rho_w * d1
        h[idx_fluid_col_1] *= hdx * dx1
        h0[idx_fluid_col_1] *= hdx * dx1
        rho[idx_fluid_col_1] *= rho_w * d1
        rho0[idx_fluid_col_1] *= rho_w * d1

        # Setting second fluid column properties
        idx_fluid_col_2 = where(x >= 1000.0)[0]
        m[idx_fluid_col_2] *= dx2 * rho_w * d2
        h[idx_fluid_col_2] *= hdx * dx2
        h0[idx_fluid_col_2] *= hdx * dx2
        rho[idx_fluid_col_2] *= rho_w * d2
        rho0[idx_fluid_col_2] *= rho_w * d2


        fluid = gpa_swe(x=x, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                        name='fluid')

        # Closed Boundary
        x = concatenate((arange(-2*dx1, fluid_col1_len, dx1),
                         arange(fluid_col1_len, tot_fluid_len+2*dx2+1e-4, dx2))
                         , axis=0)
        idx_cb_left = where((x < 0))[0]
        idx_cb_right = where((x > 2000))[0]

        m_cb = ones_like(x)
        h_cb = ones_like(x)
        rho_cb = ones_like(x)
        dw_cb = ones_like(x)
        cs_cb = ones_like(x)
        alpha_cb = ones_like(x)

        m_cb[idx_cb_left] *=  dx1 * rho_w * d1
        h_cb[idx_cb_left] *= hdx * dx1
        rho_cb[idx_cb_left] *= rho_w * d1
        dw_cb[idx_cb_left] *= d1
        cs_cb[idx_cb_left] *= sqrt(9.8 * dw_cb[idx_cb_left])
        alpha_cb[idx_cb_left] *= dim * rho_cb[idx_cb_left]

        m_cb[idx_cb_right] *=  dx2 * rho_w * d2
        h_cb[idx_cb_right] *= hdx * dx2
        rho_cb[idx_cb_right] *= rho_w * d2
        dw_cb[idx_cb_right] *= d2
        cs_cb[idx_cb_right] *= sqrt(9.8 * dw_cb[idx_cb_right])
        alpha_cb[idx_cb_right] *= dim * rho_cb[idx_cb_right]

        boundary = gpa_swe(name='boundary', x=x, m=m_cb,
                           h=h_cb, rho=rho_cb, dw=dw_cb, cs=cs_cb,
                           alpha=alpha_cb)

        idx_to_remove = where((x >= 0) & (x <= 2000))[0]
        boundary.remove_particles(idx_to_remove)

        compute_initial_props([fluid, boundary])
        return [fluid, boundary]

    def create_solver(self):
        kernel = CubicSpline(dim=1)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 60
        solver = Solver(
            kernel=kernel,
            dim=1,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=[10, 20, 30, 40, 50, 60],
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
                    ParticleAccelerations(dim, dest='fluid', sources=['fluid',
                                          'boundary'], u_only=True),
                    ],
                ),
            ]
        return equations


def compute_initial_props(particles):
    one_time_equations = [
        Group(
            equations=[
                SWEOS(dest='fluid'),
                ],
            )
    ]
    kernel = CubicSpline(dim=1)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=1,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = RectangularDamBreak()
    app.run()
