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
dx = 0.025
h0 = hdx * dx
dim = 2


class RectangularDamBreak(Application):
    def create_particles(self):
        """Create the Rectangular patch of fluid."""
        x, y = mgrid[-2:2+1e-4:dx, -0.5:0.5+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x) * dx * dx * rho_w
        h = ones_like(x) * h0

        rho = ones_like(x) * rho_w
        rho0 = ones_like(x) * rho_w
        rho_prev_iter = ones_like(x) #This was zeros_like and hence creating lot of problems when dividing with rho_residual!!!
        rho_residual= zeros_like(x)
        positive_rho_residual = zeros_like(x)
        summation_rho = zeros_like(x)

        dw = ones_like(x) * d
        cs = ones_like(x) * sqrt(9.8*d)
        p = ones_like(x) * 0.5 * rho_w * g * d**2
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

        dt_cfl = ones_like(x)

        au = zeros_like(x)
        av = zeros_like(x)

        pa = gpa(x=x, y=y, m=m, rho0=rho0, rho=rho, exp_lambda=exp_lambda,
                rho_prev_iter=rho_prev_iter, rho_residual=rho_residual,
                positive_rho_residual=positive_rho_residual, cs=cs,
                summation_rho=summation_rho, alpha=alpha, h=h, u=u, v=v, uh=uh,
                vh=vh, u_prev_iter=u_prev_iter, v_prev_iter=v_prev_iter, au=au,
                av=av, tv=tv, tu=tu, p=p, dw=dw, dt_cfl=dt_cfl, name='fluid')

        props = ['u', 'v', 'm', 'tu', 'tv', 'dw', 'au', 'av', 'h', 'rho', 'p']
        pa.add_output_arrays(props)
        print ( "Rectangular Dam break :: %d particles"
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
                    ParticleAccelerations(dim, dest='fluid', sources=['fluid',],
                                          u_only=True),
                    ],
                ),
            ]
        return equations


if __name__ == '__main__':
    app = RectangularDamBreak()
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
    tf = 2
    solver = Solver(
        kernel=kernel,
        dim=2,
        integrator=integrator,
        cfl=0.1,
        adaptive_timestep=True,
        output_at_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
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
    a_eval.compute(0.0, 1e-4)

    app.run()
