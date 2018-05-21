# Numpy
from numpy import ( sqrt, ones_like, zeros, zeros_like, mgrid, pi,
                    arange, concatenate, sin, cos )

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
hdx = 1.3
d = 0.25
n = 50
dr = 0.25 / n
dx = dr
dim = 2


#class ParticleAccelerations(Equation):
#    # This contains equation governing the flat bed case with bed gradient = 0
#    def __init__(self, dim, dest, sources):
#        super(ParticleAccelerations, self).__init__(dest, sources)
#        self.g = 9.81
#        self.rhow = 1000.0
#        self.ct = self.g/(2*self.rhow)
#        self.dim = dim
#
#    def initialize(self, d_idx, d_tu, d_tv):
#        d_tu[d_idx] = 0.0
#        d_tv[d_idx] = 0.0
#
#    def loop(self, d_x, d_y, t, dt, d_rho , d_idx , s_m , s_idx , s_rho , d_m ,
#            DWI ,DWJ , d_au, d_av , s_alpha , d_alpha , s_p , d_p , d_tu ,
#            s_tu, d_tv , s_tv):
#        tmp1 = (s_rho[s_idx]*self.dim) / s_alpha[s_idx]
#        tmp2 = (d_rho[d_idx]*self.dim) / d_alpha[d_idx]
#        d_tu[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[0] + tmp2*DWI[0])
#        d_tv[d_idx] += s_m[s_idx] * self.ct * (tmp1*DWJ[1] + tmp2*DWI[1])
#
#    def post_loop(self,t, dt, d_x ,d_y,d_idx ,d_u ,d_v ,d_tu ,d_tv ,d_au, d_av):
#        dhxi = -0.839
#        dhyi = 0.909 * d_y[d_idx]
#        dhxxi = 0
#        dhxyi = 0
#        dhyyi = 0.909
#        vikivi = d_u[d_idx]*d_u[d_idx]*dhxxi + 2*d_u[d_idx]*d_v[d_idx]*dhxyi + \
#                 d_v[d_idx]*d_v[d_idx]*dhyyi
#        tidotdhi = d_tu[d_idx]*dhxi + d_tv[d_idx]*dhyi
#        dhidotdhi = dhxi**2 + dhyi**2
#        temp3 = self.g + vikivi - tidotdhi
#        temp4 = 1 +  dhidotdhi
#        d_au[d_idx] = -(temp3/temp4)*dhxi - d_tu[d_idx]
#        d_av[d_idx] = -(temp3/temp4)*dhyi - d_tv[d_idx]


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

        #m = ones_like(x) * 0.025/16. * 0.025 * rho_w * .25
        m = ones_like(x) * (1.56*dr*dr) * rho_w * d

        rho = ones_like(x) * rho_w * d
        rho0 = ones_like(x) * rho_w * d

        h = ones_like(x) * hdx * dx
        h0 = ones_like(x) * hdx * dx

        fluid = gpa_swe(x=x, y=y, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                        name='fluid')


        # Bed props
        dxb = dx
        left_edge_bed = -1
        right_edge_bed = 8
        top_edge_bed = 2
        bottom_edge_bed = -2
        xb, yb = mgrid[left_edge_bed:+right_edge_bed+1e-4:dxb,
                       bottom_edge_bed:+top_edge_bed+1e-4:dxb]
        xb = xb.ravel()
        yb = yb.ravel()

        b = (yb**2/2.2) + 40*(pi/180.0) * xb

        Vb = ones_like(xb) * dxb * dxb
        hb = ones_like(xb) * hdx * dxb

        bed = gpa_swe(name='bed', x=xb, y=yb, V=Vb, b=b, h=hb)

        compute_initial_props([fluid, bed])
        return [fluid, bed]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 4.0
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.3,
            adaptive_timestep=True,
            output_at_times=(0.5, 2.0, 4.0),
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    InitialGuessDensity(dim, dest='fluid', sources=['fluid',],),
                    ]
                ),
            Group(
                equations=[
                    UpdateSmoothingLength(dim, dest='fluid'),
                    ], update_nnps=True
                ),
            Group(
                equations=[
                    Group(
                        equations=[
                            CorrectionFactorVariableSmoothingLength(
                                dest='fluid', sources=['fluid',]
                                ),
                            SummationDensity(dest='fluid', sources=['fluid',]),
                            ]
                        ),
                    Group(
                        equations=[
                            DensityResidual('fluid'),
                            DensityNewtonRaphsonIteration(dim, dest='fluid')
                            ]
                        ),
                    Group(
                        equations=[
                            UpdateSmoothingLength(dim, dest='fluid')
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
                    ]
                ),
            Group(
                equations=[
                    FluidBottomElevation(dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    FluidBottomGradient(dest='fluid', sources=['bed']),
                    FluidBottomCurvature(dest='fluid', sources=['bed'])
                    ]
                ),
            Group(
                equations=[
                    ParticleAcceleration(dim, dest='fluid', sources=['fluid',]),
                    ],
                ),
            ]
        return equations


def compute_initial_props(particles):
    one_time_equations = [
       Group(
            equations=[
                FluidBottomElevation(dest='fluid', sources=['bed'])
                    ]
            ),
       Group(
            equations=[
                BedGradient(dest='bed', sources=['bed']),
                BedCurvature(dest='bed', sources=['bed'])
                    ]
            ),
        Group(
            equations=[
                CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                        sources=['fluid',]),
                ]
            ),
        Group(
            equations=[
                SWEOS(dest='fluid'),
                ],
            )
    ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = CircularDamBreak()
    app.run()
