# Numpy
from numpy import ( sqrt, ones_like, zeros, zeros_like, mgrid, pi,
                    arange, concatenate, sin, cos )

# PySPH base
from pysph.base.utils import get_particle_array as gpa
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator
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
dr = 0.25 / n
dx = dr
h0 = hdx * dx
dim = 2


class ParticleAccelerations(Equation):
    # This contains equation governing the flat bed case with bed gradient = 0
    def __init__(self, dim, dest, sources):
        super(ParticleAccelerations, self).__init__(dest, sources)
        self.g = 9.81
        self.rhow = 1000.0
        self.ct = self.g/(2*self.rhow)
        self.dim = dim

    def initialize(self, d_idx, d_tu, d_tv):
        d_tu[d_idx] = 0.0
        d_tv[d_idx] = 0.0

    def loop(self, d_x, d_y, t, dt, d_rho , d_idx , s_m , s_idx , s_rho , d_m ,
            DWI ,DWJ , d_au, d_av , s_alpha , d_alpha , s_p , d_p , d_tu ,
            s_tu, d_tv , s_tv):
        tmp1 = (s_rho[s_idx]*self.dim) / s_alpha[s_idx]
        tmp2 = (d_rho[d_idx]*self.dim) / d_alpha[d_idx]
        d_tu[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[0] + tmp2*DWI[0])
        d_tv[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[1] + tmp2*DWI[1])

    def post_loop(self,t, dt, d_x ,d_y,d_idx ,d_u ,d_v ,d_tu ,d_tv ,d_au, d_av):
        dhxi = -0.839
        dhyi = d_y[d_idx] / 1.1
        dhxxi = 0
        dhxyi = 0
        dhyyi = 1.0 / 1.1
        vikivi = d_u[d_idx]*d_u[d_idx]*dhxxi + 2*d_u[d_idx]*d_v[d_idx]*dhxyi + \
                 d_v[d_idx]*d_v[d_idx]*dhyyi
        tidotdhi = d_tu[d_idx]*dhxi + d_tv[d_idx]*dhyi
        dhidotdhi = dhxi**2 + dhyi**2
        temp3 = self.g + vikivi - tidotdhi
        temp4 = 1 +  dhidotdhi
        d_au[d_idx] = -(temp3/temp4)*dhxi - d_tu[d_idx]
        d_av[d_idx] = -(temp3/temp4)*dhyi - d_tv[d_idx]


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

        m = ones_like(x) * 0.025/16. * 0.025 * rho_w * .25
        h = ones_like(x) * hdx * dx

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
        u_prev_iter = zeros_like(x)
        v = zeros_like(x)
        v_prev_iter = zeros_like(x)

        dt_cfl = zeros_like(x)

        au = zeros_like(x)
        av = zeros_like(x)

        pa = gpa(x=x, y=y, m=m, rho0=rho0, rho=rho, exp_lambda=exp_lambda,
                rho_prev_iter=rho_prev_iter, rho_residual=rho_residual,
                positive_rho_residual=positive_rho_residual,
                summation_rho=summation_rho, alpha=alpha, h=h, u=u, v=v,
                u_prev_iter=u_prev_iter, v_prev_iter=v_prev_iter, au=au, av=av,
                A=A, cs=cs, dt_cfl=dt_cfl, tv=tv, tu=tu, p=p, dw=dw,
                name='fluid')

        props = ['u', 'v', 'tu', 'tv', 'dw', 'au', 'av', 'h', 'rho', 'p']
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
    integrator = EulerIntegrator(fluid=EulerStep())
    dt = 1e-4; tf = 2
    solver = Solver(
        kernel=kernel,
        dim=2,
        integrator=integrator,
        dt=dt,
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
    a_eval.compute(0.0, dt)

    app.run()

