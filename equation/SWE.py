from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.integrator import Integrator
from pysph.base.utils import get_particle_array
from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array
from pysph.base.cython_generator import declare

from numpy import sqrt, cos, sin, ones, zeros, pi
import numpy as np


def get_particle_array_swe(constants=None, **props):
    """Return a particle array for the Shallow Water formulation.

    This sets the default properties to be::

        ['x', 'y', 'z', 'u', 'v', 'w', 'h', 'rho', 'arho' 'm', 'p', 'V', 'A',
        'cs', 'n', 'rho0', 'rho_prev_iter', 'rho_residual',
        'positive_rho_residual', 'summation_rho', 'dw', 'alpha', 'exp_lambda',
        'tv', 'tu', 'au', 'av', 'u_prev_step', 'v_prev_step', 'uh', 'vh',
        'dt_cfl', 'pa_to_split', 'Sfx', 'Sfy', 'psi', 'sum_Ak', 'u_parent',
        v_parent', 'uh_parent', 'vh_parent', 'parent_idx', 'b', 'bx', 'by',
        'bxx', 'bxy', byy', 'closest_idx', 'merge', 'dw_inner_reimann',
        'u_inner_reimann', 'v_inner_reimann', 'shep_corr']


    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """

    swe_props = ['A', 'cs', 'n', 'rho0', 'rho_prev_iter', 'rho_residual',
                 'positive_rho_residual', 'summation_rho', 'dw', 'alpha',
                 'exp_lambda', 'tv', 'tu', 'u_prev_step', 'v_prev_step',
                 'uh', 'vh', 'dt_cfl', 'pa_to_split', 'Sfx', 'Sfy', 'V',
                 'psi', 'sum_Ak', 'arho', 'u_parent', 'v_parent', 'uh_parent',
                 'vh_parent', 'b', 'bx', 'by', 'bxx', 'bxy', 'byy', 'merge',
                 'dw_inner_reimann', 'u_inner_reimann', 'v_inner_reimann',
                 'shep_corr']

    consts = {'tmp_comp': [0.0, 0.0]}
    if constants:
        for key, val in constants.items():
            consts[key] = val

    pa = get_particle_array(
        constants=consts, additional_props=swe_props, **props
    )
    pa.add_property('parent_idx', type='int')
    pa.add_property('closest_idx', type='int')

    # default property arrays to save out.
    props = ['x', 'y', 'h', 'rho', 'p', 'A', 'uh', 'vh', 'u', 'v', 'm', 'tu',
             'tv', 'dw', 'alpha', 'au', 'av', 'Sfx', 'Sfy', 'n', 'cs', 'pid',
             'gid', 'tag']
    pa.set_output_arrays(props)

    return pa


class CheckForParticlesToSplit(Equation):
    def __init__(self, dest, h_max=1e9, A_max=1e9, x_min=-1e9, x_max=1e9,
                 y_min=-1e9, y_max=1e9):
        self.A_max = A_max
        self.h_max = h_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        super(CheckForParticlesToSplit, self).__init__(dest, None)

    def initialize(self, d_pa_to_split, d_idx):
        d_pa_to_split[d_idx] = 0.0

    def post_loop(self, d_idx, d_rho0, d_A, d_h, d_x, d_y, d_pa_to_split):
        if (d_A[d_idx] > self.A_max and d_h[d_idx] < self.h_max
           and (self.x_min < d_x[d_idx] < self.x_max)
           and (self.y_min < d_y[d_idx] < self.y_max)):
            d_pa_to_split[d_idx] = 1


class ParticleSplit(object):
    def __init__(self, pa_arr):
        self.pa_arr = pa_arr
        self.center_pa_mass_frac = 0.178705766141917
        self.edge_pa_mass_frac = 0.136882287617319
        self.pa_h_ratio = 0.9
        self.center_and_edge_pa_separation_frac = 0.4
        self.props = self.pa_arr.properties.keys() # Change
        self.idx_pa_to_split = self._get_idx_of_particles_to_split()
        self.num_edge_pa_after_single_split = 6

    def do_particle_split(self, solver=None):
        if not self.idx_pa_to_split.size:
            return
        else:
            # Parent particle properties
            h_parent = self.pa_arr.h[self.idx_pa_to_split]
            h0_parent = self.pa_arr.h0[self.idx_pa_to_split]
            m_parent = self.pa_arr.m[self.idx_pa_to_split]
            x_parent = self.pa_arr.x[self.idx_pa_to_split]
            y_parent = self.pa_arr.y[self.idx_pa_to_split]
            u_parent = self.pa_arr.u[self.idx_pa_to_split]
            v_parent = self.pa_arr.v[self.idx_pa_to_split]
            u_prev_step_parent = self.pa_arr.u_prev_step[self.idx_pa_to_split]
            v_prev_step_parent = self.pa_arr.v_prev_step[self.idx_pa_to_split]
            rho_parent = self.pa_arr.rho[self.idx_pa_to_split]
            rho0_parent = self.pa_arr.rho0[self.idx_pa_to_split]
            alpha_parent = self.pa_arr.alpha[self.idx_pa_to_split]

            # Edge daughter particle properties update
            n = self.num_edge_pa_after_single_split
            h_edge_pa  = self.pa_h_ratio * np.repeat(h_parent, n)
            h0_edge_pa = self.pa_h_ratio * np.repeat(h_parent, n)
            u_prev_step_edge_pa = np.repeat(u_prev_step_parent, n)
            v_prev_step_edge_pa = np.repeat(v_prev_step_parent, n)
            m_edge_pa  = self.edge_pa_mass_frac * np.repeat(m_parent, n)
            edge_pa_pos = self._get_edge_pa_positions(h_parent, u_parent,
                                                      v_parent)
            x_edge_pa  = edge_pa_pos[0] + np.repeat(x_parent, n)
            y_edge_pa  = edge_pa_pos[1] + np.repeat(y_parent, n)

            total_num_edge_pa = x_edge_pa.size
            rho0_edge_pa = np.repeat(rho0_parent, n)
            rho_edge_pa = np.repeat(rho_parent, n)
            alpha_edge_pa = np.repeat(alpha_parent, n)
            parent_idx_edge_pa = np.repeat(self.idx_pa_to_split, n)

            # Center daughter particle properties update
            for idx in self.idx_pa_to_split:
                self.pa_arr.m[idx] *= self.center_pa_mass_frac
                self.pa_arr.h[idx] *= self.pa_h_ratio
                self.pa_arr.h0[idx] *= self.pa_h_ratio
                self.pa_arr.parent_idx[idx] = int(idx)

            self._add_edge_pa_prop(h0_edge_pa, h_edge_pa, m_edge_pa, x_edge_pa,
                                   y_edge_pa, rho0_edge_pa, rho_edge_pa,
                                   u_prev_step_edge_pa, v_prev_step_edge_pa,
                                   alpha_edge_pa, parent_idx_edge_pa)

    def _get_idx_of_particles_to_split(self):
        idx_pa_to_split = []
        for idx, val in enumerate(self.pa_arr.pa_to_split):
            if val:
                idx_pa_to_split.append(idx)
        return np.array(idx_pa_to_split)

    def _get_edge_pa_positions(self, h_parent, u_parent, v_parent):
        num_of_pa_to_split = len(self.idx_pa_to_split)
        n = self.num_edge_pa_after_single_split
        theta_edge_pa = zeros(n)
        r = self.center_and_edge_pa_separation_frac

        for i, theta in enumerate(range(0, 360, 60)):
            theta_edge_pa[i] = (pi/180)*theta

        angle_vel =  np.where((np.abs(u_parent) > 1e-3) | (np.abs(v_parent) >
                              1e-3), np.arctan2(v_parent, u_parent), 0)
        angle_actual = np.tile(theta_edge_pa, num_of_pa_to_split) \
                       + np.repeat(angle_vel, n)

        x = r * np.cos(angle_actual) * np.repeat(h_parent, n)
        y = r * np.sin(angle_actual) * np.repeat(h_parent, n)
        return x.copy(), y.copy()

    def _add_edge_pa_prop(self, h0_edge_pa, h_edge_pa, m_edge_pa, x_edge_pa,
                          y_edge_pa, rho0_edge_pa, rho_edge_pa,
                          u_prev_step_edge_pa, v_prev_step_edge_pa,
                          alpha_edge_pa, parent_idx_edge_pa):
        add_prop = {}
        for prop in self.props:
            if prop == 'm':
                add_prop[prop] = m_edge_pa
            elif prop == 'h':
                add_prop[prop] = h_edge_pa
            elif prop == 'h0':
                add_prop[prop] = h0_edge_pa
            elif prop == 'x':
                add_prop[prop] = x_edge_pa
            elif prop == 'y':
                add_prop[prop] = y_edge_pa
            elif prop == 'u_prev_step':
                add_prop[prop] = u_prev_step_edge_pa
            elif prop == 'v_prev_step':
                add_prop[prop] = v_prev_step_edge_pa
            elif prop == 'rho0':
                add_prop[prop] = rho0_edge_pa
            elif prop == 'rho':
                add_prop[prop] = rho_edge_pa
            elif prop == 'alpha':
                add_prop[prop] = alpha_edge_pa
            elif prop == 'parent_idx':
                add_prop[prop] = parent_idx_edge_pa.astype(int)
            else:
                pass
        self.pa_arr.add_particles(**add_prop)


class DaughterVelocityEval(Equation):
    def __init__(self, rhow, dest, sources):
        self.rhow = rhow
        super(DaughterVelocityEval, self).__init__(dest, sources)

    def initialize(self, d_sum_Ak, d_idx, d_m, d_rho, d_u, d_v, d_uh,
                   d_vh, d_u_parent, d_v_parent, d_uh_parent, d_vh_parent,
                   d_parent_idx):
        d_sum_Ak[d_idx] = 0.0
        d_u_parent[d_idx] = d_u[d_parent_idx[d_idx]]
        d_uh_parent[d_idx] = d_uh[d_parent_idx[d_idx]]
        d_v_parent[d_idx] = d_v[d_parent_idx[d_idx]]
        d_vh_parent[d_idx] = d_vh[d_parent_idx[d_idx]]

    def loop_all(self, d_sum_Ak, d_pa_to_split, d_parent_idx, d_idx, s_m, s_rho,
                 s_parent_idx, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        if d_pa_to_split[d_idx]:
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                if s_parent_idx[s_idx] == d_parent_idx[d_idx]:
                    d_sum_Ak[d_idx] += s_m[s_idx] / s_rho[s_idx]

    def post_loop(self, d_idx, d_parent_idx, d_A, d_sum_Ak, d_dw, d_rho, d_u,
                  d_uh, d_vh, d_v, d_u_parent, d_v_parent, d_uh_parent,
                  d_vh_parent, t):
        if d_parent_idx[d_idx]:
            cv = d_A[d_parent_idx[d_idx]] / d_sum_Ak[d_parent_idx[d_idx]]
            dw_ratio = d_dw[d_parent_idx[d_idx]] / (d_rho[d_idx]/self.rhow)
            d_u[d_idx] = cv * dw_ratio * d_u_parent[d_idx]
            d_uh[d_idx] = cv * dw_ratio * d_uh_parent[d_idx]
            d_v[d_idx] = cv * dw_ratio * d_v_parent[d_idx]
            d_vh[d_idx] = cv * dw_ratio * d_vh_parent[d_idx]
            d_parent_idx[d_idx] = 0


class FindMergeable(Equation):
    def __init__(self, dest, sources, A_min, x_min=-1e9, x_max=1e9, y_min=-1e9,
                 y_max=1e9):
        self.A_min = A_min
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        super(FindMergeable, self).__init__(dest, sources)

    def loop_all(self, d_idx, d_merge, d_closest_idx, d_x, d_y, d_h, d_A,
                 s_x, s_y, s_A, NBRS, N_NBRS):
        i, closest = declare('int', 2)
        s_idx = declare('unsigned int')
        d_merge[d_idx] = 0
        xi = d_x[d_idx]
        yi = d_y[d_idx]
        rmin = d_h[d_idx] * 10.0
        closest = -1
        if (d_A[d_idx] < self.A_min
            and ((self.x_min < d_x[d_idx] < self.x_max)
                  and (self.y_min < d_y[d_idx] < self.y_max))):
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                if s_idx == d_idx:
                    continue
                xij = xi - s_x[s_idx]
                yij = yi - s_y[s_idx]
                rij = sqrt(xij*xij + yij*yij)
                if rij < rmin:
                    closest = s_idx
                    rmin = rij
        d_closest_idx[d_idx] = closest

    def post_loop(self, d_idx, d_m, d_u, d_v, d_h, d_uh, d_vh, d_closest_idx,
                  d_merge, d_x, d_y, KERNEL):
        idx = declare('int')
        xma = declare('matrix(3)')
        xmb = declare('matrix(3)')
        idx = d_closest_idx[d_idx]
        if idx > -1:
            if d_idx == d_closest_idx[idx]:
                if d_idx < idx:
                    m_merged = d_m[d_idx] + d_m[idx]
                    x_merged = (d_m[d_idx]*d_x[d_idx] + d_m[idx]*d_x[idx]) \
                                / m_merged
                    y_merged = (d_m[d_idx]*d_y[d_idx] + d_m[idx]*d_y[idx]) \
                                / m_merged
                    xma[0] = x_merged - d_x[d_idx]
                    xma[1] = y_merged - d_y[d_idx]
                    xmb[0] = x_merged - d_x[idx]
                    xmb[1] = y_merged - d_y[idx]
                    rma = sqrt(xma[0]*xma[0] + xma[1]*xma[1])
                    rmb = sqrt(xmb[0]*xmb[0] + xmb[1]*xmb[1])
                    d_u[d_idx] = (d_m[d_idx]*d_u[d_idx] + d_m[idx]*d_u[idx]) \
                                  / m_merged
                    d_uh[d_idx] = (d_m[d_idx]*d_uh[d_idx] + d_m[idx]*d_uh[idx]) \
                                  / m_merged
                    d_v[d_idx] = (d_m[d_idx]*d_v[d_idx] + d_m[idx]*d_v[idx]) \
                                  / m_merged
                    d_vh[d_idx] = (d_m[d_idx]*d_vh[d_idx] + d_m[idx]*d_vh[idx]) \
                                  / m_merged
                    const1 = d_m[d_idx] * KERNEL.kernel(xma, rma, d_h[d_idx])
                    const2 = d_m[idx] * KERNEL.kernel(xmb, rmb, d_h[idx])
                    d_h[d_idx] = sqrt((7*pi/10.) * (m_merged/(const1+const2)))
                    d_m[d_idx] = m_merged
                else:
                    d_merge[d_idx] = 1

    def reduce(self, dst, t, dt):
        indices = declare('object')
        indices = numpy.where(dst.merge > 0)[0]
        if numpy.any(indices):
            dst.remove_particles(indices)


class InitialDensityEvalAfterMerge(Equation):
    def loop_all(self, d_rho, d_idx, d_merge, d_closest_idx, d_x, d_y, s_h,
                 s_m, s_x, s_y, KERNEL, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix(3)')
        if d_merge[d_closest_idx[d_idx]] == 1:
            d_rho[d_idx] = 0.0
            rij = 0.0
            rho_sum = 0.0
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                xij[0] = d_x[d_idx] - s_x[s_idx]
                xij[1] = d_y[d_idx] - s_y[s_idx]
                rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1])
                rho_sum += s_m[s_idx] * KERNEL.kernel(xij, rij, s_h[s_idx])
            d_rho[d_idx] += rho_sum


class EulerStep(IntegratorStep):
    """Fast but inaccurate integrator. Use this for testing"""
    def initialize(self, d_u, d_v, d_u_prev_step, d_v_prev_step, d_idx):
        d_u_prev_step[d_idx] = d_u[d_idx]
        d_v_prev_step[d_idx] = d_v[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_au, d_av, d_x, d_y, dt):
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]


class SWEStep(IntegratorStep):
    """Predictor corrector Integrator for Shallow Water problems"""
    def initialize(self, t, d_u, d_v, d_uh, d_vh, d_u_prev_step, d_v_prev_step,
                   d_idx):
        d_u_prev_step[d_idx] = d_u[d_idx]
        d_v_prev_step[d_idx] = d_v[d_idx]
        #if d_uh[d_idx] == 0.0 and d_vh[d_idx] == 0.0:
        #    d_uh[d_idx] = d_u[d_idx]
        #    d_vh[d_idx] = d_v[d_idx]

    def stage1(self, d_uh, d_vh, d_idx, d_au, d_av, dt):
        d_uh[d_idx] +=  dt * d_au[d_idx]
        d_vh[d_idx] +=  dt * d_av[d_idx]

    def stage2(self, d_u, d_v, d_uh, d_vh, d_idx, d_au, d_av, d_x, d_y, dt):
        d_x[d_idx] += dt * d_uh[d_idx]
        d_y[d_idx] += dt * d_vh[d_idx]
        d_u[d_idx] = d_uh[d_idx] + dt/2.*d_au[d_idx]
        d_v[d_idx] = d_vh[d_idx] + dt/2.*d_av[d_idx]


class SWEIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations()

        self.initialize()

        # Predict
        self.stage1()

        # Call any post-stage functions.
        self.do_post_stage(0.5*dt, 1)

        # Correct
        self.stage2()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)


class GatherDensityEvalNextIteration(Equation):
    def initialize(self, d_rho, d_idx, d_rho_prev_iter):
        d_rho_prev_iter[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0

    def loop(self, d_rho, d_idx, s_m, s_idx, WI):
        d_rho[d_idx] += s_m[s_idx] * WI


class ScatterDensityEvalNextIteration(Equation):
    def initialize(self, t, d_rho, d_idx, d_rho_prev_iter):
        d_rho_prev_iter[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0

    def loop(self, d_rho, d_idx, s_m, s_idx, WJ):
        d_rho[d_idx] += s_m[s_idx] * WJ


class NonDimensionalDensityResidual(Equation):
    def __init__(self, dest, sources=None):
        super(NonDimensionalDensityResidual, self).__init__(dest, sources)

    def post_loop(self, d_psi, d_rho, d_rho_prev_iter, d_idx):
        d_psi[d_idx] = abs(d_rho[d_idx]-d_rho_prev_iter[d_idx]) \
                       / d_rho_prev_iter[d_idx]


#class CheckConvergenceDensityResidual(Equation):
#    def __init__(self, dest, sources=None):
#        super(CheckConvergenceDensityResidual, self).__init__(dest, sources)
#        self.eqn_has_converged = 0
#
#    def initialize(self):
#        self.eqn_has_converged = 0
#
#    def reduce(self, dst, t, dt):
#        dst.tmp_comp[0] = serial_reduce_array(dst.psi >= 0.0, 'sum')
#        dst.tmp_comp[1] = serial_reduce_array(dst.psi**2, 'sum')
#        dst.get_carray('tmp_comp').set_data(parallel_reduce_array(dst.tmp_comp, 
#                                                                  'sum'))
#        epsilon = sqrt(dst.tmp_comp[1] / dst.tmp_comp[0])
#        if epsilon <= 1e-3:
#            self.eqn_has_converged = 1
#
#    def converged(self):
#        return self.eqn_has_converged


class CheckConvergenceDensityResidual(Equation):
    def __init__(self, dest, sources=None):
        super(CheckConvergenceDensityResidual, self).__init__(dest, sources)
        self.eqn_has_converged = 0

    def initialize(self):
        self.eqn_has_converged = 0

    def reduce(self, dst, t, dt):
        epsilon = max(dst.psi)
        if epsilon <= 1e-3:
            self.eqn_has_converged = 1

    def converged(self):
        return self.eqn_has_converged


class InitialTimeGatherSummationDensity(Equation):
    def initialize(self, d_rho0, d_idx):
        d_rho0[d_idx] = 0.0

    def loop(self, d_rho0, d_idx, s_m, s_idx, WI):
        d_rho0[d_idx] += s_m[s_idx] * WI

    def post_loop(self, d_rho0, d_rho, d_idx):
        d_rho[d_idx] = d_rho0[d_idx]


class InitialTimeScatterSummationDensity(Equation):
    def initialize(self, d_rho0, d_idx):
        d_rho0[d_idx] = 0.0

    def loop(self, d_rho0, d_idx, s_m, s_idx, WJ):
        d_rho0[d_idx] += s_m[s_idx] * WJ

    def post_loop(self, d_rho0, d_rho, d_idx):
        d_rho[d_idx] = d_rho0[d_idx]


class CorrectionFactorVariableSmoothingLength(Equation):
    def initialize(self, d_idx, d_alpha):
         d_alpha[d_idx] = 0.0

    def loop(self, d_alpha, d_idx, DWIJ, XIJ, s_idx, s_m):
        d_alpha[d_idx] += -s_m[s_idx] * (DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1])


class SummationDensity(Equation):
    def initialize(self, d_summation_rho, d_idx):
        d_summation_rho[d_idx] = 0.0

    def loop(self, d_summation_rho, d_idx, s_m, s_idx, WI):
        d_summation_rho[d_idx] += s_m[s_idx] * WI


class InitialGuessDensityAndSmoothingLengthVacondio(Equation):
    def __init__(self, dim, dest, sources):
        super(InitialGuessDensityAndSmoothingLengthVacondio,
              self).__init__(dest, sources)
        self.dim = dim

    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0

    def loop(self, d_arho, d_idx, s_m, s_rho, s_idx, d_u_prev_step,
             d_v_prev_step, s_u_prev_step, s_v_prev_step, DWI):
        tmp1 = (d_u_prev_step[s_idx]-s_u_prev_step[d_idx]) * DWI[0]
        tmp2 = (d_v_prev_step[s_idx]-s_v_prev_step[d_idx]) * DWI[1]
        d_arho[d_idx] += (s_m[s_idx]/s_rho[s_idx]) * (tmp1 + tmp2)

    def post_loop(self, d_rho, d_h, dt, d_arho, d_idx):
        d_rho[d_idx] = d_rho[d_idx] + dt*d_rho[d_idx]*d_arho[d_idx]
        d_h[d_idx] = d_h[d_idx] - (dt/self.dim)*d_h[d_idx]*d_arho[d_idx]


class InitialGuessDensity(Equation):
    def __init__(self, dim, dest, sources):
        super(InitialGuessDensity, self).__init__(dest, sources)
        self.dim = dim

    def initialize(self, d_exp_lambda, d_idx):
        d_exp_lambda[d_idx] = 0.0

    def loop(self, d_exp_lambda, d_u_prev_step, d_v_prev_step, d_alpha, d_idx,
             s_m, s_u_prev_step, s_v_prev_step, s_idx, DWI, dt, t):
        a1 = (s_u_prev_step[s_idx]-d_u_prev_step[d_idx]) * DWI[0]
        a2 = (s_v_prev_step[s_idx]-d_v_prev_step[d_idx]) * DWI[1]
        const = (self.dim*dt) / d_alpha[d_idx]
        d_exp_lambda[d_idx] +=  const * (s_m[s_idx]*(a1+a2))

    def post_loop(self, t, d_rho, d_exp_lambda, d_idx):
        d_rho[d_idx] = d_rho[d_idx] * e**(d_exp_lambda[d_idx])


#class UpdateSmoothingLength(Equation):
#    def __init__(self, h0, dim, dest, sources=None):
#        self.h0 = h0
#        self.dim = dim
#        super(UpdateSmoothingLength, self).__init__(dest, sources)
#
#    def post_loop(self, d_h, d_rho0, d_rho, d_idx):
#        d_h[d_idx] = self.h0 * (d_rho0[d_idx]/d_rho[d_idx])**(1./self.dim)


class UpdateSmoothingLength(Equation):
    def __init__(self, dim, dest):
        self.dim = dim
        super(UpdateSmoothingLength, self).__init__(dest, None)

    def post_loop(self, d_h, d_h0, d_rho0, d_rho, d_idx):
        d_h[d_idx] = d_h0[d_idx] * (d_rho0[d_idx]/d_rho[d_idx])**(1./self.dim)


class DensityResidual(Equation):
    def __init__(self, dest, sources=None):
        super(DensityResidual, self).__init__(dest, sources)

    def post_loop(self, d_rho, d_idx, d_rho_residual, d_summation_rho, t):
        d_rho_residual[d_idx] = d_rho[d_idx] - d_summation_rho[d_idx]


class DensityNewtonRaphsonIteration(Equation):
    def __init__(self, dim, dest, sources=None):
        self.dim = dim
        super(DensityNewtonRaphsonIteration, self).__init__(dest, sources)

    def initialize(self, d_rho, d_rho_prev_iter, d_idx):
        d_rho_prev_iter[d_idx] = d_rho[d_idx]

    def post_loop(self, d_rho, d_idx, d_alpha, d_rho_residual):
        a1 = d_rho_residual[d_idx] * self.dim
        a2 = a1 + d_alpha[d_idx]
        const = 1 - (a1/a2)
        d_rho[d_idx] = d_rho[d_idx] * const


class CheckConvergence(Equation):
    def __init__(self, dest, sources=None):
        super(CheckConvergence, self).__init__(dest, sources)
        self.eqn_has_converged = 0

    def initialize(self):
        self.eqn_has_converged = 0

    def post_loop(self, d_positive_rho_residual, d_rho_residual,
                  d_rho_prev_iter,  d_idx, t):
        d_positive_rho_residual[d_idx] = abs(d_rho_residual[d_idx])

    def reduce(self, dst, t, dt):
        max_epsilon = max(dst.positive_rho_residual / dst.rho_prev_iter)
        if max_epsilon <= 1e-15:
            self.eqn_has_converged = 1

    def converged(self):
        return self.eqn_has_converged


class SWEOS(Equation):
    def __init__(self, dest, sources=None, g=9.81, rhow=1000.0):
        self.rhow = rhow
        self.g = g
        self.fac = 0.5 * (g/rhow)
        super(SWEOS, self).__init__(dest, sources)

    def post_loop(self, d_rho, d_cs, d_u, d_v, d_idx, d_p, d_dw, d_dt_cfl,
                 d_m, d_A, d_alpha):
        d_p[d_idx] = self.fac * (d_rho[d_idx])**2
        d_cs[d_idx] = sqrt(self.g * d_rho[d_idx]/self.rhow)
        d_A[d_idx] = d_m[d_idx] / d_rho[d_idx]
        d_dw[d_idx] = d_rho[d_idx] / self.rhow
        d_dt_cfl[d_idx] = d_cs[d_idx] + (d_u[d_idx]**2 + d_v[d_idx]**2)**0.5




def mu_calc(hi=1.0, hj=1.0, rhoi=1.0, rhoj=1.0, csi=1.0, csj=1.0,
            velij_dot_rij=1.0, rij2=1.0):
    h_bar = (hi+hj) / 2.0
    rho_bar = (rhoi+rhoj) / 2.0
    cs_bar = (csi+csj) / 2.0
    eta2 = 0.01 * hi**2
    muij = (h_bar*velij_dot_rij) / (rij2+eta2)
    return muij


def artificial_visc(alpha=1.0, rij2=1.0,  hi=1.0, hj=1.0, rhoi=1.0, rhoj=1.0,
                    csi=1.0, csj=1.0, muij=1.0):
    # Artificial Viscosity (Monoghan)
    cs_bar = (csi+csj) / 2.0
    rho_bar = (rhoi+rhoj) / 2.0
    pi_visc = -(alpha*cs_bar*muij) / rho_bar
    return pi_visc


def viscosity_LF(alpha=1.0, rij2=1.0, hi=1.0, hj=1.0, rhoi=1.0, rhoj=1.0,
                 csi=1.0, csj=1.0, muij=1.0):
    # Viscosity (Ata and Soulaimani)
    cs_bar = (csi+csj) / 2.0
    rho_bar = (rhoi+rhoj) / 2.0
    eta2 = 0.01 * hi**2
    h_bar = (hi+hj) / 2.0
    tmp = (muij*(rij2+eta2)**0.5) / h_bar
    pi_visc = -(cs_bar*tmp) / rho_bar
    return pi_visc


class ParticleAccelerations(Equation):
    def __init__(self, dim, dest, sources, bx=0, by=0, bxx=0,
                 bxy=0, byy=0, u_only=False, v_only=False,
                 alpha=0, visc_option=2):
        super(ParticleAccelerations, self).__init__(dest, sources)
        self.g = 9.81
        self.rhow = 1000.0
        self.ct = self.g / (2*self.rhow)
        self.dim = dim
        self.bx = bx
        self.by = by
        self.bxx = bxx
        self.bxy = bxy
        self.byy = byy
        self.u_only = u_only
        self.v_only = v_only
        self.alpha = 0.0
        if visc_option == 1:
            self.viscous_func = artificial_visc
        else:
            self.viscous_func = viscosity_LF

    def initialize(self, d_idx, d_tu, d_tv):
        d_tu[d_idx] = 0.0
        d_tv[d_idx] = 0.0

    def loop(self, d_x, d_y, s_x, s_y, d_rho, d_idx, s_m, s_idx, s_rho, d_m,
             DWI, DWJ, d_au, d_av, s_alpha, d_alpha, s_p, d_p, d_tu, s_dw, d_dw,
             t, s_tu, d_tv, s_tv, d_h, s_h, d_u, s_u, d_v, s_v, d_cs, s_cs):
        #tmp1 = (s_rho[s_idx]*self.dim) / s_alpha[s_idx]
        #tmp2 = (d_rho[d_idx]*self.dim) / d_alpha[d_idx]
        tmp1 = (s_dw[s_idx]*self.rhow*self.dim) / s_alpha[s_idx]
        tmp2 = (d_dw[d_idx]*self.rhow*self.dim) / d_alpha[d_idx]

        uij = d_u[d_idx] - s_u[s_idx]
        vij = d_v[d_idx] - s_v[s_idx]
        xij = d_x[d_idx] - s_x[s_idx]
        yij = d_y[d_idx] - s_y[s_idx]
        rij2 = xij**2 + yij**2
        uij_dot_xij = uij * xij
        vij_dot_yij = vij * yij
        velij_dot_rij = uij_dot_xij + vij_dot_yij

        muij = mu_calc(d_h[d_idx], s_h[s_idx], d_rho[d_idx], s_rho[s_idx],
                       d_cs[d_idx], s_cs[s_idx], velij_dot_rij, rij2)


        if velij_dot_rij < 0:
            pi_visc = self.viscous_func(self.alpha, rij2, d_h[d_idx],
                                        s_h[s_idx], d_rho[d_idx], s_rho[s_idx],
                                        d_cs[d_idx], s_cs[s_idx], muij)
        else:
            pi_visc = 0


        d_tu[d_idx] += s_m[s_idx] * ((self.ct*tmp1 + 0.5*pi_visc)*DWJ[0] +
                                     (self.ct*tmp2 + 0.5*pi_visc)*DWI[0])

        d_tv[d_idx] += s_m[s_idx] * ((self.ct*tmp1 + 0.5*pi_visc)*DWJ[1] +
                                     (self.ct*tmp2 + 0.5*pi_visc)*DWI[1])

    def _get_helpers_(self):
        return [mu_calc, artificial_visc, viscosity_LF]

    def post_loop(self, d_idx, d_u, d_v, d_tu, d_tv, d_au, d_av, d_Sfx, d_Sfy):
        bx =  self.bx; by = self.by
        bxx = self.bxx; bxy = self.bxy
        byy = self.byy
        vikivi = d_u[d_idx]*d_u[d_idx]*bxx + 2*d_u[d_idx]*d_v[d_idx]*bxy + \
                 d_v[d_idx]*d_v[d_idx]*byy
        tidotgradbi = d_tu[d_idx]*bx + d_tv[d_idx]*by
        gradbidotgradbi = bx**2 + by**2
        temp3 = self.g + vikivi - tidotgradbi
        temp4 = 1 + gradbidotgradbi
        if not self.v_only:
            d_au[d_idx] = -(temp3/temp4)*bx - d_tu[d_idx] - d_Sfx[d_idx]
        if not self.u_only:
            d_av[d_idx] = -(temp3/temp4)*by - d_tv[d_idx] - d_Sfy[d_idx]


class BedElevation(Equation):
    def initialize(self, d_b, d_idx):
        d_b[d_idx] = 0.0

    def loop(self, d_b, d_idx, s_b, s_idx, WJ, s_V, RIJ):
        if RIJ > 1e-6:
            d_b[d_idx] += s_b[s_idx] * WJ * s_V[s_idx]


class BedGradient(Equation):
    def initialize(self, d_bx, d_by, d_idx):
        d_bx[d_idx] = 0.0
        d_by[d_idx] = 0.0

    def loop(self, d_bx, d_by, d_idx, s_b, s_idx, DWJ, s_V, RIJ):
        if RIJ > 1e-6:
            d_bx[d_idx] += s_b[s_idx] * DWJ[0] * s_V[s_idx]
            d_by[d_idx] += s_b[s_idx] * DWJ[1] * s_V[s_idx]


class BedCurvature(Equation):
    def initialize(self, d_bxx, d_bxy, d_byy, d_idx):
        d_bxx[d_idx] = 0.0
        d_bxy[d_idx] = 0.0
        d_byy[d_idx] = 0.0

    def loop(self, d_bxx, d_bxy, d_byy, d_b, d_idx, s_h, s_b, s_idx, XIJ, RIJ,
             DWJ, s_V):
        if RIJ > 1e-6:
            eta = 0.01 * s_h[s_idx]
            temp1 = (d_b[d_idx]-s_b[s_idx]) / (RIJ**2+eta**2)
            temp2 = XIJ[0]*DWJ[0] + XIJ[1]*DWJ[1]
            temp_bxx = ((4*XIJ[0]**2/RIJ**2)-1) * temp1
            temp_bxy = ((4*XIJ[0]*XIJ[1]/RIJ**2)-1) * temp1
            temp_byy = ((4*XIJ[1]**2/RIJ**2)-1) * temp1
            d_bxx += temp_bxx * temp2 * s_V[s_idx]
            d_bxy += temp_bxy * temp2 * s_V[s_idx]
            d_byy += temp_byy * temp2 * s_V[s_idx]


class BedFrictionSourceEval(Equation):
    def __init__(self, dest, sources):
        self.g = 9.8
        super(BedFrictionSourceEval, self).__init__(dest, sources)

    def initialize(self, d_n, d_idx):
        d_n[d_idx] = 0.0

    def loop(self, d_n, d_idx, s_n, s_idx, WJ, s_V, RIJ):
        if RIJ > 1e-6:
            d_n[d_idx] += s_n[s_idx] * WJ * s_V[s_idx]

    def post_loop(self, d_idx, d_Sfx, d_Sfy, d_u, d_v, d_n, d_dw):
        vmag = sqrt(d_u[d_idx]**2 + d_v[d_idx]**2)
        temp = (self.g*d_n[d_idx]**2*vmag) / d_dw[d_idx]**(4.0/3.0)
        d_Sfx[d_idx] = d_u[d_idx] * temp
        d_Sfy[d_idx] = d_v[d_idx] * temp


class BoundaryInnerReimannStateEval(Equation):
    def initialize(self, d_u_inner_reimann, d_v_inner_reimann,
                   d_dw_inner_reimann, d_idx):
        d_u_inner_reimann[d_idx] = 0.0
        d_v_inner_reimann[d_idx] = 0.0
        d_dw_inner_reimann[d_idx] = 0.0

    def loop_all(self, d_shep_corr, d_x, d_y, d_idx, s_x, s_y, s_m, s_rho,
                 s_idx, d_h, KERNEL, NBRS, N_NBRS):
        i = declare('int')
        xij = declare('matrix(3)')
        rij = 0.0
        corr_sum = 0.0
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1])
            corr_sum += (s_m[s_idx]/s_rho[s_idx]) * KERNEL.kernel(xij, rij,
                                                                  d_h[d_idx])
        d_shep_corr[d_idx] = corr_sum

    def loop(self, d_u_inner_reimann, d_v_inner_reimann, d_dw_inner_reimann,
             d_shep_corr, d_idx, WI, s_m, s_u, s_v, s_rho, s_dw, s_idx):
        tmp = (WI/d_shep_corr[d_idx]) * (s_m[s_idx]/s_rho[s_idx])
        d_u_inner_reimann[d_idx] += s_u[s_idx] * tmp
        d_v_inner_reimann[d_idx] += s_v[s_idx] * tmp
        d_dw_inner_reimann[d_idx] += s_dw[s_idx] * tmp


class SubCriticalInFlow(Equation):
    def __init__(self, dest):
        self.g = 9.8
        super(SubCriticalInFlow, self).__init__(dest, None)

    def post_loop(self, d_dw, d_dw_inner_reimann, d_u, d_u_inner_reimann,
                  d_idx):
        const = 1. / (2.*sqrt(self.g))
        d_dw[d_idx] = (const*(d_u[d_idx] - d_u_inner_reimann[d_idx])
                       + sqrt(d_dw_inner_reimann[d_idx]))**2


class SubCriticalOutFlow(Equation):
    def __init__(self, dest):
        self.g = 9.8
        super(SubCriticalOutFlow, self).__init__(dest, None)

    def post_loop(self, d_dw, d_dw_inner_reimann, d_u, d_u_inner_reimann,
                  d_v, d_v_inner_reimann, d_idx):
        const = 2. * sqrt(self.g)
        d_u[d_idx] = d_u_inner_reimann[d_idx] \
                     + const*(sqrt(d_dw_inner_reimann[d_idx])
                              - sqrt(d_dw[d_idx]))
        d_v[d_idx] = d_v_inner_reimann[d_idx]


class SuperCriticalOutFlow(Equation):
    def post_loop(self, d_dw, d_dw_inner_reimann, d_u, d_u_inner_reimann,
                  d_v, d_v_inner_reimann, d_idx):
        d_u[d_idx] = d_u_inner_reimann[d_idx]
        d_v[d_idx] = d_v_inner_reimann[d_idx]
        d_dw[d_idx] = d_dw_inner_reimann[d_idx]



#class InletDensityEvalFromDepth(Equation):
#    def __init__(self, dest, rhow=1000.0):
#        self.rhow = rhow
#        super(InletDensityEvalFromDepth, self).__init__(dest, None)
#
#    def post_loop(self, d_rho, d_idx, d_dw):
#        d_rho[d_idx] = d_dw[d_idx] * self.rhow


