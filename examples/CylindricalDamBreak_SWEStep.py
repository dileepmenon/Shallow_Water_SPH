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
from pysph.sph.SWE.basic_equations import *


# Constants
rho_w = 1000.0
g = 9.81
hdx = 2.3
d = 1.0
n = 50
dr = 0.5 / n
dx = 0.01
h0 = hdx * dx
dim = 2


class CircularDamBreak(Application):
    def create_particles(self):
        """Create the circular patch of fluid."""
        x = zeros(0)
        y = zeros(0)
        m = zeros(0)

        # Create circular patch in a radial grid
        rad = 0.0
        for j in range(1, n+1):
                npnts = 4 * j
                dtheta = (2*pi) / npnts

                theta = arange(0, 2*pi-1e-10, dtheta)
                rad = rad + dr

                _x = rad * cos(theta)
                _y = rad * sin(theta)

                x = concatenate( (x, _x) )
                y = concatenate( (y, _y) )

        m = ones_like(x) * dx * dx * rho_w * (2.5 * 2.5/4)
        h = ones_like(x) * h0

        rho = ones_like(x) * rho_w
        rho0 = ones_like(x) * rho_w
        rho_prev_iter = ones_like(x)
        rho_residual= zeros_like(x)
        positive_rho_residual = zeros_like(x)
        summation_rho = zeros_like(x)

        dw = ones_like(x) * d
        cs = ones_like(x) * sqrt(9.8*d)
        p = ones_like(x) * 0.5 * rho_w * g * d**2
        A = m / rho
        alpha = ones_like(x)
        exp_lambda = zeros_like(x)

        tv = zeros_like(x)
        tu = zeros_like(x)

        u = zeros_like(x)
        uh = zeros_like(x)
        u_prev_iter = zeros_like(x)
        v = zeros_like(x)
        vh = zeros_like(x)
        v_prev_iter = zeros_like(x)

        dt_cfl = zeros_like(x)

        au = zeros_like(x)
        av = zeros_like(x)

        pa = gpa(x=x, y=y, m=m, rho0=rho0, rho=rho, exp_lambda=exp_lambda,
                rho_prev_iter=rho_prev_iter, rho_residual=rho_residual,
                positive_rho_residual=positive_rho_residual, cs=cs,
                summation_rho=summation_rho, alpha=alpha, h=h, u=u, v=v, uh=uh,
                vh=vh, u_prev_iter=u_prev_iter, v_prev_iter=v_prev_iter, au=au,
                A=A, av=av, tv=tv, tu=tu, p=p, dw=dw, dt_cfl=dt_cfl,
                name='fluid')

        props = ['u', 'v', 'm', 'tu', 'tv', 'dw', 'au', 'av', 'h', 'rho', 'p']
        pa.add_output_arrays(props)
        print ( "Cylindrical Dam break :: %d particles"
                %(pa.get_number_of_particles()) )

        return [pa]

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    InitialGuessDensity(dim, dest='fluid', sources=['fluid',],),
                    UpdateSmoothingLength(h0, dim, dest='fluid')
                    ], update_nnps=True
                ),
            Group(
                equations=[
                    Group(
                        equations=[
                            CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                                    sources=['fluid',]),
                            SummationDensity(dest='fluid', sources=['fluid',]),
                            DensityResidual('fluid'),
                            DensityNewtonRaphsonIteration(dim, dest='fluid')
                            ]),
                    Group(
                        equations=[
                            UpdateSmoothingLength(h0, dim, dest='fluid')
                            ], update_nnps=True
                        ),
                    Group(
                        equations=[
                            SummationDensity(dest='fluid', sources=['fluid',]),
                            ],
                        ),
                    Group(
                        equations=[
                            DensityResidual(dest='fluid'),
                            CheckConvergence(dest='fluid')
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
                    SWEOS(dest='fluid'),
                    ParticleAccelerations(dim, dest='fluid', sources=['fluid',]),
                    ],
                ),
            ]
        return equations


if __name__ == '__main__':
    app = CircularDamBreak()
    one_time_equations = [
        Group(
            equations=[
                CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                        sources=['fluid',]),
                InitialTimeSummationDensity(dest='fluid', sources=['fluid',]),
                SWEOS(dest='fluid'),
                ], update_nnps=False
            )
    ]

    from pysph.sph.acceleration_eval import AccelerationEval
    from pysph.sph.sph_compiler import SPHCompiler
    kernel = CubicSpline(dim=2)
    integrator = SWEIntegrator(fluid=SWEStep())
    tf = 6
    solver = Solver(
        kernel=kernel,
        dim=2,
        integrator=integrator,
        cfl=0.3,
        adaptive_timestep=True,
        tf=tf
        )
    part_arr = app.create_particles
    eqns = app.create_equations()
    app.setup(solver=solver, equations=eqns,
          particle_factory=part_arr)
    a_eval = AccelerationEval(
        solver.particles, equations=one_time_equations, kernel=CubicSpline(dim=2))
    compiler = SPHCompiler(a_eval, None)
    compiler.compile()
    a_eval.set_nnps(app.nnps)
    app.run()
