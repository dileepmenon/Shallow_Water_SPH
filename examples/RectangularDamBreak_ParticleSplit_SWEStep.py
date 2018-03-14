# Numpy
from numpy import ( ones_like, zeros, zeros_like, mgrid, pi,
                    arange, sqrt, concatenate, sin, cos, where )

# PySPH base
from pysph.base.utils import get_particle_array as gpa
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator_step import IntegratorStep

# PySPH equations
from pysph.sph.equation import Group
from equation.SWE import *

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
hdx = 2.3
d = 10.0
dx = 20.
h0 = hdx * dx
dim = 2
A_max = 1.5 * dx**2
h_max = 2 * h0


class RectangularDamBreak(Application):
    def create_particles(self):
        """Create the Rectangular patch of fluid."""
        x, y = mgrid[-1000:1000+1e-4:dx, -500:500+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x) * dx * dx * rho_w * d
        h = ones_like(x) * h0

        rho = ones_like(x) * rho_w
        rho0 = ones_like(x) * rho_w
        rho_prev_iter = ones_like(x) * rho_w

        A = m / rho

        dw = ones_like(x) * d
        cs = ones_like(x) * sqrt(9.8*d)
        p = ones_like(x) * 0.5 * rho_w * g * d**2
        alpha = ones_like(x)

        pa_to_split = zeros_like(x)
        sum_Ak = zeros_like(x)
        psi = zeros_like(x)

        tv = zeros_like(x)
        tu = zeros_like(x)

        u = zeros_like(x)
        u_parent = zeros_like(x)
        uh = zeros_like(x)
        u_prev_iter = zeros_like(x)
        v = zeros_like(x)
        v_parent = zeros_like(x)
        vh = zeros_like(x)
        v_prev_iter = zeros_like(x)

        dt_cfl = ones_like(x)

        au = zeros_like(x)
        av = zeros_like(x)

        Sfx = zeros_like(x)
        Sfy = zeros_like(x)

        consts = {'tmp_comp': [0.0, 0.0]}

        pa = gpa(x=x, y=y, m=m, rho0=rho0, rho=rho, A=A, psi=psi, cs=cs,
                 alpha=alpha, h=h, u=u, v=v, uh=uh, vh=vh, Sfx=Sfx, Sfy=Sfy,
                 u_prev_iter=u_prev_iter, sum_Ak=sum_Ak, u_parent=u_parent,
                 v_parent=v_parent, pa_to_split=pa_to_split, dt_cfl=dt_cfl,
                 v_prev_iter=v_prev_iter, au=au, av=av, tv=tv, tu=tu, p=p,
                 rho_prev_iter=rho_prev_iter, dw=dw, name='fluid',
                 constants=consts)

        pa.add_property('parent_idx', type='int')

        props = ['parent_idx', 'A', 'u', 'v', 'm', 'tu', 'tv', 'dw', 'au',
                'av', 'h', 'rho', 'p', 'pa_to_split']
        pa.add_output_arrays(props)

        compute_initial_props([pa])
        return [pa]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 40
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.1,
            adaptive_timestep=True,
            output_at_times=[10, 20, 30, 40, 50],
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    Group(
                        equations=[
                            ScatterDensityEvalNextIteration(dest='fluid',
                                                            sources=['fluid',]),
                            ]
                        ),
                    Group(
                        equations=[
                            NonDimensionalDensityResidual(dest='fluid')
                            ]
                        ),
                    Group(
                        equations=[
                            UpdateSmoothingLength(h0, dim, dest='fluid')
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            CheckConvergenceDensityResidual(dest='fluid')
                            ],
                    )], iterate=True, max_iterations=100
            ),
            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                            sources=['fluid',]),
            ]),
            Group(
                equations=[
                    DaughterVelocityEval(rho_w, dest='fluid',
                                         sources=['fluid',]),
                    ]
                ),
            Group(
                equations=[
                    SWEOS(dest='fluid'),
                    ]
                ),
            Group(
                equations=[
                    ParticleAccelerations(dim, dest='fluid', sources=['fluid',],
                                          u_only=True),
                    ]
                ),
            Group(
                equations=[
                    CheckForParticlesToSplit(dest='fluid', h_max=h_max,
                                             A_max=A_max)
                    ],
                ),
            ]
        return equations

    def pre_step(self, solver):
        for pa in self.particles:
            ps = ParticleSplit(pa)
            ps.do_particle_split()
        self.nnps.update()


def compute_initial_props(particles):
    one_time_equations = [
                Group(
                    equations=[
                        InitialTimeScatterSummationDensity(dest='fluid',
                                                           sources=['fluid',]),
                        ]
                    ),
                Group(
                    equations=[
                        SWEOS(dest='fluid'),
                        ]
                    ),
                Group(
                    equations=[
                        CheckForParticlesToSplit(dest='fluid', h_max=h_max,
                                                 A_max=A_max),
                            ]
                    )
            ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = RectangularDamBreak()
    app.run()
