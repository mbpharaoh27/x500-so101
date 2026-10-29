"""Finite shaft motors -> actual hinge speed -> rotor thrust and air drag.

Normal run: ground idle; W/S, arrows, Q/E, 1..6 and Shift teleoperation.
--test: deterministic input validation; --window-test: visible GLFW smoke test.
No real aircraft connection. The +/-2.26 Nm motor limit is a user-approved
simulation assumption, including active braking, not a hardware motor rating.
"""
from pathlib import Path
from itertools import product
import argparse
import csv
import hashlib
import json
import math
import struct
import time
import xml.etree.ElementTree as ET
import zlib

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
XML = HERE/'PX4_X500_SO101_keyboard.xml'
SOURCE = HERE/'PX4_X500_SO101_arm_sequence.xml'
ARM_NAMES = ('so101_shoulder_pan', 'so101_shoulder_lift', 'so101_elbow_flex',
             'so101_wrist_flex', 'so101_wrist_roll', 'so101_gripper')


def load():
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key('keyboard_ready').id)
    mujoco.mj_forward(m, d)
    return m, d


def smoothstep(s):
    s = float(np.clip(s, 0.0, 1.0))
    return s**3*(10.0-15.0*s+6.0*s*s)


def allocate(a, b, cap):
    answer = np.linalg.solve(a, b)
    if np.all(answer >= 0.0) and np.all(answer <= cap):
        return answer, False
    weights = np.array([1.0, 6.0, 6.0, 10.0])
    a, b = weights[:, None]*a, weights*b
    best = np.clip(answer, 0.0, cap)
    cost = np.sum((a@best-b)**2)
    for states in product((-1, 0, 1), repeat=4):
        states = np.array(states)
        trial = np.where(states == 1, cap, 0.0)
        free = states == -1
        if free.any():
            trial[free] = np.linalg.lstsq(a[:, free], b-a@trial, rcond=None)[0]
        if np.any(trial < -1e-9) or np.any(trial > cap+1e-9):
            continue
        new_cost = np.sum((a@trial-b)**2)
        if new_cost < cost:
            best, cost = np.clip(trial, 0.0, cap), new_cost
    return best, True


class MeasuredRotorFlight:
    def __init__(self, m, d):
        self.m = m
        self.base = m.body('base_link').id
        self.free = m.joint('x500_free').id
        self.vadr = int(m.jnt_dofadr[self.free])
        self.rotors = np.array([m.body(f'rotor_{i}').id for i in range(4)])
        self.joints = np.array([m.joint(f'rotor_{i}_joint').id for i in range(4)])
        self.rotor_vadr = m.jnt_dofadr[self.joints]
        self.rotor_qadr = m.jnt_qposadr[self.joints]
        self.shaft_ids = np.array([m.actuator(f'x500_rotor_{i}_shaft_motor').id for i in range(4)])
        self.aero_ids = np.array([m.actuator(f'x500_rotor_{i}_aerodynamics').id for i in range(4)])
        self.arm_joints = np.array([m.joint(n).id for n in ARM_NAMES])
        self.arm_qadr = m.jnt_qposadr[self.arm_joints]
        self.arm_ids = np.array([m.actuator(n+'_hold').id for n in ARM_NAMES])
        self.arm_target = d.ctrl[self.arm_ids].copy()
        self.parameters = np.array([m.numeric(f'x500_rotor_{i}_motor_parameters').data
                                    for i in range(4)])
        self.kf, self.km, self.max_speed, self.tau_up, self.tau_down, self.spin = self.parameters.T
        self.cap = self.kf*self.max_speed**2
        self.torque_cap = float(m.numeric('approved_shaft_torque_limit_Nm').data[0])
        self.period = float(m.numeric('outer_control_period_s').data[0])
        self.dt = float(m.opt.timestep)
        self.start = d.xpos[self.base].copy()
        self.ground = m.geom('ground').id
        self.ground_z = float(d.geom_xpos[self.ground, 2])
        self.target_z = self.ground_z+2.0
        self.mass = float(m.body_subtreemass[self.base])
        self.g = -float(m.opt.gravity[2])
        r = d.xmat[self.base].reshape(3, 3)
        self.yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        self.initial_com = r.T@(d.subtree_com[self.base]-self.start)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full)
        c = self.initial_com
        self.inertia = (full[self.vadr+3:self.vadr+6, self.vadr+3:self.vadr+6]
                        - self.mass*(np.dot(c, c)*np.eye(3)-np.outer(c, c)))
        self.axial_inertia = []
        for joint, body in zip(self.joints, self.rotors):
            axis = d.xmat[body].reshape(3, 3)@m.jnt_axis[joint]
            principal = d.ximat[body].reshape(3, 3)
            self.axial_inertia.append(float(axis@(principal@np.diag(m.body_inertia[body])
                                                   @principal.T)@axis))
        self.axial_inertia = np.array(self.axial_inertia)
        self.integral = np.zeros(3)
        self.command = np.zeros(4)
        self.target_speed = np.zeros(4)
        self.saturated = False
        self.next_outer = 0.0
        self.reference_z = self.start[2]
        self.measured_speed = d.qvel[self.rotor_vadr].copy()
        self.force = np.zeros(4)
        self.shaft_request = np.zeros(4)
        if self.mass*self.g >= self.cap.sum():
            raise ValueError('Insufficient source thrust capacity')
        if not np.allclose(m.jnt_axis[self.joints], [0, 0, 1]):
            raise ValueError('Expected source +Z rotor axes')



    def apply_inner(self, d):
        # This is the actual engine-integrated JOINT velocity. There is no
        # independent simulated motor-speed state and no assignment to qvel.
        self.measured_speed = d.qvel[self.rotor_vadr].copy()
        w = self.measured_speed
        # Same signed-thrust convention as the source Gazebo model. Correct
        # spin gives +Z thrust; back-driving gives reverse thrust. The XML
        # limits the absolute aerodynamic force to the original source cap.
        self.force = self.spin*self.kf*w*np.abs(w)
        drag_compensation = self.km*self.kf*w*np.abs(w)
        accelerating = self.spin*self.target_speed > self.spin*w
        tau = np.where(accelerating, self.tau_up, self.tau_down)
        self.shaft_request = self.axial_inertia*(self.target_speed-w)/tau+drag_compensation
        d.ctrl[self.shaft_ids] = np.clip(self.shaft_request, -self.torque_cap, self.torque_cap)
        d.ctrl[self.aero_ids] = self.force
        d.ctrl[self.arm_ids] = self.arm_target
        # Each aero site is on its ROTOR: Fz=force, Tz=-spin*km*force.
        # Shaft motor torque already includes its reaction on the parent.
        # Never add an extra yaw torque directly to base_link.





def write_png(path, rgb):
    height, width = rgb.shape[:2]
    def chunk(kind, payload):
        return (struct.pack('!I', len(payload))+kind+payload
                + struct.pack('!I', zlib.crc32(kind+payload) & 0xffffffff))
    raw = b''.join(b'\0'+row.tobytes() for row in np.asarray(rgb, dtype=np.uint8))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'
                     + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b''))


def lowest(m, d, geom):
    if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        return float(d.geom_xpos[geom, 2]
                     - np.abs(d.geom_xmat[geom].reshape(3, 3)[2])@m.geom_size[geom])
    if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[geom]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        return float((m.mesh_vert[a:a+n]@d.geom_xmat[geom].reshape(3, 3).T
                      + d.geom_xpos[geom])[:, 2].min())
    raise ValueError('Unsupported clearance geometry')


import glfw

FOLDED = np.deg2rad([70., -179., 178., 66., 100., 0.])
ALIGNED = np.deg2rad([0., -179., 178., 66., 100., 0.])
DOWN = np.deg2rad([0., -90., 0., 0., 100., 0.])
XY_SPEED, Z_SPEED, YAW_SPEED = .3, .2, math.radians(15.)
# Command headroom for the measured attitude/motor response, below the agreed
# operating ceilings. These do NOT change any source actuator limits.
XY_COMMAND, Z_COMMAND, YAW_COMMAND = .20, .18, math.radians(12.)
XY_ACCEL, Z_ACCEL, YAW_ACCEL = .15, .15, math.radians(20.)
ARM_SPEED, ARM_ACCEL = math.radians(10.), math.radians(30.)
MAX_HEIGHT, MAX_TILT = 2., math.radians(10.)


def approach(value, target, delta):
    return value+np.clip(target-value, -delta, delta)


def quintic(start, end, elapsed, duration):
    s = float(np.clip(elapsed/duration, 0., 1.))
    delta = end-start
    return (start+delta*smoothstep(s), delta*30*s*s*(1-s)**2/duration,
            delta*60*s*(1-s)*(1-2*s)/duration**2)


class Keys:
    """Physical GLFW key codes, no text/auto-repeat or global keyboard hook."""
    def __init__(self):
        self.down = set()
        self.edges = set()
        self.focused = True
        self.paused = False
        self.reset = False

    def clear(self):
        self.down.clear()
        self.edges.clear()

    def focus(self, window, focused):
        self.focused = bool(focused)
        self.clear()

    def key(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            self.down.discard(key)
        elif action == glfw.PRESS and self.focused:
            if key == glfw.KEY_SPACE:
                self.paused = not self.paused
                self.clear()
            elif key == glfw.KEY_R:
                self.reset = True
                self.paused = False
                self.clear()
            elif key == glfw.KEY_ESCAPE:
                if window is not None:
                    glfw.set_window_should_close(window, True)
            elif not self.paused:
                self.down.add(key)
                self.edges.add(key)
        # REPEAT never changes a control reference.

    def axis(self, positive, negative):
        return float(positive in self.down)-float(negative in self.down)

    @property
    def shift(self):
        return glfw.KEY_LEFT_SHIFT in self.down or glfw.KEY_RIGHT_SHIFT in self.down


class PoseGuard:
    """Independent planning data only. Never edits the running simulation state."""
    def __init__(self, m, d, controller):
        self.m, self.c = m, controller
        self.probe = mujoco.MjData(m)
        self.arm_geoms = [g for g in range(m.ngeom) if m.geom(g).name.startswith('so101_')
                          and m.geom_group[g] == 3]
        self.corners = {}
        self.rotor_bounds = []
        for g in self.arm_geoms:
            mesh = m.geom_dataid[g]
            a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            points = m.mesh_vert[a:a+n]
            lo, hi = points.min(0), points.max(0)
            self.corners[g] = np.array(list(product(*zip(lo, hi))))
        for i, body in enumerate(controller.rotors):
            g = m.geom(f'x500_rotor_{i}_visual').id
            mesh = m.geom_dataid[g]
            a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, m.geom_quat[g])
            points = m.mesh_vert[a:a+n]@matrix.reshape(3, 3).T+m.geom_pos[g]
            self.rotor_bounds.append((float(np.linalg.norm(points[:, :2], axis=1).max()),
                                      float(points[:, 2].min()), float(points[:, 2].max())))
        collision_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        excluded = set(int(x) for x in m.exclude_signature)
        self.pairs = []
        for a in self.arm_geoms:
            for b in collision_geoms:
                if b == controller.ground or a >= b and b in self.arm_geoms:
                    continue
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                wa, wb = int(m.body_weldid[ba]), int(m.body_weldid[bb])
                pa = int(m.body_weldid[m.body_parentid[wa]])
                pb = int(m.body_weldid[m.body_parentid[wb]])
                if wa == wb or wa == pb or wb == pa:
                    continue
                if (min(wa, wb)<<16)+max(wa, wb) in excluded:
                    continue
                if not ((m.geom_contype[a]&m.geom_conaffinity[b]) or
                        (m.geom_contype[b]&m.geom_conaffinity[a])):
                    continue
                self.pairs.append((a, b))
        self.last_reason = ''

    def check(self, data, q, clearance=.0008, torque_fraction=.85):
        m, p, c = self.m, self.probe, self.c
        limits = c.arm_limits
        if np.any(q < limits[:, 0]-1e-8) or np.any(q > limits[:, 1]+1e-8):
            return False, 'joint limit / braking margin'
        # Offline copies for prediction, not animation or actual joint actuation.
        p.qpos[:] = data.qpos
        p.qpos[c.arm_qadr] = q
        p.qvel[:] = 0.
        mujoco.mj_forward(m, p)
        for a, b in self.pairs:
            if np.linalg.norm(p.geom_xpos[a]-p.geom_xpos[b]) > m.geom_rbound[a]+m.geom_rbound[b]+clearance:
                continue
            distance = mujoco.mj_geomDistance(m, p, a, b, clearance, None)
            if distance < clearance-1e-9:
                return False, f'clearance: {m.geom(a).name} / {m.geom(b).name}'
        for g, local in self.corners.items():
            world = local@p.geom_xmat[g].reshape(3, 3).T+p.geom_xpos[g]
            if world[:, 2].min() < c.ground_z+.10:
                return False, 'arm ground clearance below 0.10 m'
            # Conservative arm AABB vs complete propeller swept cylinders.
            for body, (radius, zlo, zhi) in zip(c.rotors, self.rotor_bounds):
                rotor_local = (world-p.xpos[body])@p.xmat[body].reshape(3, 3)
                lo, hi = rotor_local.min(0), rotor_local.max(0)
                xy = np.maximum(np.maximum(lo[:2], -hi[:2]), 0.)
                if hi[2] > zlo-clearance and lo[2] < zhi+clearance and np.linalg.norm(xy) < radius+clearance:
                    return False, 'propeller swept volume'
        cap = m.actuator_forcerange[c.arm_ids, 1]
        if np.any(np.abs(p.qfrc_bias[c.arm_vadr]) > torque_fraction*cap):
            return False, 'predicted gravity torque reserve insufficient'
        return True, ''


class KeyboardFlight(MeasuredRotorFlight):
    def __init__(self, m, d):
        super().__init__(m, d)
        self.arm_vadr = m.jnt_dofadr[self.arm_joints]
        self.arm_limits = m.jnt_range[self.arm_joints].copy()
        self.arm_limits[:, 0] += math.radians(1.)
        self.arm_limits[:, 1] -= math.radians(1.)
        # Original gripper rest is 0deg, approved open pose 99deg. Mechanical
        # source limits remain untouched; teleoperation uses this inner range.
        self.arm_limits[-1] = [0., math.radians(99.)]
        self.kp = m.actuator_gainprm[self.arm_ids, 0].copy()
        self.kv = -m.actuator_biasprm[self.arm_ids, 2].copy()
        self.arm_velocity, self.arm_acceleration = np.zeros(6), np.zeros(6)
        self.arm_feedforward = np.zeros(6)
        self.full = np.zeros((m.nv, m.nv))
        self.p_ref, self.v_ref, self.a_ref = self.start.copy(), np.zeros(3), np.zeros(3)
        self.yaw_rate = self.yaw_acceleration = 0.
        self.state = 'GROUND'
        self.state_time = float(d.time)
        self.spool = 0.
        self.stop_spool = 0.
        self.message = 'Hold W to start; rotor joints are at rest.'
        self.fault = None
        self.feet = {m.geom(f'x500_base_link_collision_{i}').id for i in (3, 4)}
        self.grounded_since = None
        self.last_foot_contact = {g: -float('inf') for g in self.feet}
        self.initial_height = float(self.start[2])
        self.guard = PoseGuard(m, d, self)
        self.next_guard = 0.
        self.arm_blocked = False
        self.path = []
        self.path_index = 0
        self.path_epoch = None
        self.path_kind = None
        self.path_elapsed = 0.
        self.path_rate = 1.
        self.path_abort = False
        self.land_after_fold = False
        self.events = []
        self.ground_force = 0.
        self.integral[:] = 0.
        self.last_input = Keys()

    def transition(self, state, d, message):
        if state != self.state:
            self.events.append({'time_s': float(d.time), 'state': state, 'message': message})
        self.state, self.state_time, self.message = state, float(d.time), message

    def support(self, d):
        force, touching = 0., set()
        for i, contact in enumerate(d.contact[:d.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ground in pair and pair & self.feet:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.m, d, i, wrench)
                force += max(0., float(wrench[0]))
                touching |= pair & self.feet
                if wrench[0] > .0001:
                    for foot in pair & self.feet:
                        self.last_foot_contact[foot] = float(d.time)
        self.ground_force = force
        return len(touching) == 2, force

    def folded(self, d):
        return np.max(np.abs(d.qpos[self.arm_qadr]-FOLDED)) < math.radians(2.)

    def stop_motion(self):
        # Reference velocities decelerate in update_reference; no state writes.
        self.last_input.clear()

    def plan(self, d, waypoints, kind, durations=None):
        start = self.arm_target.copy()
        segments = []
        for i, end in enumerate(waypoints):
            end = np.asarray(end, dtype=float)
            if np.max(np.abs(end-start)) < 1e-6:
                continue
            duration = (float(durations[i]) if durations is not None else
                        max(1., 1.875*float(np.max(np.abs(end-start)))/math.radians(30.),
                            math.sqrt(5.8*float(np.max(np.abs(end-start)))/math.radians(60.))))
            for s in np.linspace(0., 1., 51):
                q, _, _ = quintic(start, end, s*duration, duration)
                ok, reason = self.guard.check(d, q)
                if not ok:
                    self.message = 'Path refused: '+reason
                    return False
            segments.append((start.copy(), end.copy(), duration))
            start = end
        self.path, self.path_index, self.path_epoch, self.path_kind = segments, 0, float(d.time), kind
        self.path_elapsed, self.path_rate, self.path_abort = 0., 1., False
        return True

    def request_fold(self, d, landing=False):
        if self.folded(d) and np.max(np.abs(self.arm_target-FOLDED)) < math.radians(.1):
            self.message = 'Arm already folded.'
            if landing:
                self.transition('LANDING', d, 'Arm already folded; beginning descent.')
            return True
        if np.max(np.abs(self.arm_velocity)) > .01:
            self.message = 'Release joint keys and wait before F/L.'
            return False
        q = self.arm_target.copy()
        reset = q.copy(); reset[4:] = FOLDED[4:]
        pan_down = DOWN.copy(); pan_down[0] = q[0]
        pan_zero = reset.copy(); pan_zero[0] = 0.
        options = [[reset, ALIGNED, FOLDED],
                   [reset, pan_down, DOWN, ALIGNED, FOLDED],
                   [reset, pan_zero, DOWN, ALIGNED, FOLDED]]
        for path in options:
            if self.plan(d, path, 'fold'):
                self.land_after_fold = landing
                self.message = 'Verified fold path; holding flight position.'
                return True
        self.message = 'Cannot verify fold path. Landing refused; adjust joints first.'
        return False

    def update_reference(self, d, keys):
        dt = self.period
        self.last_input = keys
        both_feet, support = self.support(d)
        rising = keys.axis(glfw.KEY_W, glfw.KEY_S)
        if self.state == 'GROUND' and rising > 0.:
            self.p_ref = d.xpos[self.base].copy()
            self.integral[:] = 0.
            self.transition('STARTING', d, 'Hold W: motors spool for 2 s before climbing.')
        if self.state == 'STARTING':
            self.spool = smoothstep((d.time-self.state_time)/2.)
            if rising <= 0. and both_feet:
                self.stop_spool = self.spool
                self.transition('STOPPING', d, 'W released before liftoff: cancel start.')
            elif d.time-self.state_time >= 2.:
                self.spool = 1.
                self.transition('FLIGHT', d, 'Flight: release keys to brake and hold.')
        if self.state == 'STOPPING':
            self.spool = self.stop_spool*(1.-smoothstep((d.time-self.state_time)/1.5))
            if d.time-self.state_time >= 1.5:
                self.spool = 0.
                self.integral[:] = 0.
                self.transition('GROUND', d, 'Landed; motors off. Hold W to start again.')
                self.p_ref = d.xpos[self.base].copy()
                self.v_ref[:] = 0.
        edges = keys.edges.copy()
        keys.edges.clear()
        if self.state == 'LANDING' and not keys.focused:
            self.transition('FLIGHT', d, 'Focus lost: cancel descent and brake to hold.')
        if not keys.focused and self.path:
            self.path_abort = True
        if self.state == 'FLIGHT':
            if glfw.KEY_U in edges and not self.path:
                if d.xpos[self.base, 2]-self.ground_z < 1.:
                    self.message = 'U refused: base_link must be above 1 m.'
                elif np.max(np.abs(self.arm_target-FOLDED)) < math.radians(2.):
                    self.plan(d, [ALIGNED, DOWN], 'unfold')
                else:
                    self.message = 'U requires folded pose. Use F first.'
            if glfw.KEY_F in edges and not self.path:
                self.request_fold(d)
            if glfw.KEY_L in edges:
                if self.folded(d) and not self.path:
                    self.transition('LANDING', d, 'Landing: reduce descent speed near skids.')
                elif not self.path:
                    self.request_fold(d, landing=True)
                else:
                    self.message = 'Arm helper busy. Wait for completion, then press L again.'
        command = np.zeros(3)
        yaw_command = 0.
        if self.state == 'FLIGHT' and not self.land_after_fold:
            local = np.array([keys.axis(glfw.KEY_UP, glfw.KEY_DOWN),
                              keys.axis(glfw.KEY_LEFT, glfw.KEY_RIGHT)])
            local /= max(1., float(np.linalg.norm(local)))
            matrix = d.xmat[self.base].reshape(3, 3)
            heading = math.atan2(matrix[1, 0], matrix[0, 0])
            cs, sn = math.cos(heading), math.sin(heading)
            command[:2] = XY_COMMAND*np.array([[cs, -sn], [sn, cs]])@local
            command[2] = Z_COMMAND*rising
            yaw_command = YAW_COMMAND*keys.axis(glfw.KEY_Q, glfw.KEY_E)
            if command[2] > 0.:
                distance = max(0., self.ground_z+MAX_HEIGHT-self.p_ref[2])
                command[2] = min(command[2], .95*math.sqrt(2.*Z_ACCEL*distance))
                if distance < .02:
                    self.message = '2 m operating ceiling: braking.'
            if command[2] < 0.:
                if not self.folded(d):
                    # Current conservative arm bounds plus 15cm ground buffer.
                    minimum = min(lowest(self.m, d, g) for g in self.guard.arm_geoms)
                    if minimum-self.ground_z < .20:
                        command[2] = 0.
                        self.message = 'Descent limited by arm clearance; F to fold.'
                elif d.xpos[self.base, 2] < self.initial_height+.06:
                    self.transition('LANDING', d, 'S reached landing zone: complete touchdown.')
        if self.state == 'LANDING':
            height = d.xpos[self.base, 2]-self.initial_height
            command[2] = -min(.15, max(.025, .6*max(0., height)))
            r = d.xmat[self.base].reshape(3, 3)
            # Tiny contact forces can alternate between corners at the source
            # skid margin. Require measured positive contact on BOTH skids
            # within 80ms, then dwell 0.5s with low speed; never infer landing
            # merely from altitude. Full load support is checked after unload.
            recent_feet = all(d.time-stamp < .08 for stamp in self.last_foot_contact.values())
            stable = (recent_feet and support > .05 and abs(d.qvel[2]) < .04
                      and np.linalg.norm(d.qvel[:2]) < .04 and r[2, 2] > math.cos(math.radians(3.)))
            self.grounded_since = (float(d.time) if self.grounded_since is None else self.grounded_since) if stable else None
            if self.grounded_since is not None and d.time-self.grounded_since >= .5:
                self.stop_spool = self.spool
                self.transition('STOPPING', d, 'Two skids in stable contact; unloading motors.')
        old_v = self.v_ref.copy()
        delta = command[:2]-self.v_ref[:2]
        delta *= min(1., XY_ACCEL*dt/max(1e-12, float(np.linalg.norm(delta))))
        self.v_ref[:2] += delta
        self.v_ref[2] = approach(self.v_ref[2], command[2], Z_ACCEL*dt)
        if self.state not in ('FLIGHT', 'LANDING'):
            self.v_ref[:] = approach(self.v_ref, np.zeros(3), Z_ACCEL*dt)
        self.a_ref = (self.v_ref-old_v)/dt
        self.p_ref += .5*(self.v_ref+old_v)*dt
        # Braking-distance ceiling: start slowing before the operating boundary.
        if self.p_ref[2] > self.ground_z+MAX_HEIGHT:
            self.p_ref[2] = self.ground_z+MAX_HEIGHT
            self.v_ref[2] = min(0., self.v_ref[2])
        self.p_ref[2] = max(self.p_ref[2], self.initial_height-.004)
        old_rate = self.yaw_rate
        self.yaw_rate = float(approach(self.yaw_rate, yaw_command, YAW_ACCEL*dt))
        self.yaw_acceleration = (self.yaw_rate-old_rate)/dt
        self.yaw += .5*(old_rate+self.yaw_rate)*dt
        self.update_arm(d, keys)

    def update_arm(self, d, keys):
        dt = self.period
        if self.path:
            begin, end, duration = self.path[self.path_index]
            old_rate = self.path_rate
            self.path_rate = float(approach(old_rate, 0. if self.path_abort else 1., dt))
            rate_accel = (self.path_rate-old_rate)/dt
            self.path_elapsed += .5*(old_rate+self.path_rate)*dt
            q, v, a = quintic(begin, end, self.path_elapsed, duration)
            self.arm_target = q
            self.arm_velocity = v*self.path_rate
            self.arm_acceleration = a*self.path_rate**2+v*rate_accel
            if self.path_abort and self.path_rate < 1e-9:
                self.path = []
                self.land_after_fold = False
                self.message = 'Focus loss: helper stopped on checked path; press U/F/L to request again.'
            elif self.path_elapsed >= duration:
                self.path_index += 1
                self.path_epoch = float(d.time)
                self.path_elapsed = 0.
                if self.path_index == len(self.path):
                    self.path = []
                    self.message = 'Arm path complete.'
                    if self.land_after_fold:
                        self.land_after_fold = False
                        self.transition('LANDING', d, 'Arm folded; beginning descent.')
            return
        desired = np.array([float(glfw.KEY_1+i in keys.down or glfw.KEY_KP_1+i in keys.down)
                            for i in range(6)])*ARM_SPEED*(-1. if keys.shift else 1.)
        if self.state != 'FLIGHT' or d.xpos[self.base, 2]-self.ground_z < 1.:
            if np.any(desired):
                self.message = 'Joint motion requires flight above 1 m.'
            desired[:] = 0.
        old_v = self.arm_velocity.copy()
        new_v = approach(old_v, desired, ARM_ACCEL*dt)
        candidate = self.arm_target+.5*(old_v+new_v)*dt
        if np.any(np.abs(new_v) > 1e-9):
            stop = candidate+new_v*np.abs(new_v)/(2.*ARM_ACCEL)+new_v*.04
            if d.time >= self.next_guard or np.any(stop < self.arm_limits[:, 0]) or np.any(stop > self.arm_limits[:, 1]):
                self.next_guard = float(d.time)+.02
                ok, reason = True, ''
                for fraction in (0., .5, 1.):
                    ok, reason = self.guard.check(d, candidate+fraction*(stop-candidate))
                    if not ok:
                        break
                self.arm_blocked = not ok
                if not ok:
                    self.message = 'Joint command limited: '+reason
            if self.arm_blocked:
                new_v = approach(old_v, np.zeros(6), ARM_ACCEL*dt)
                candidate = self.arm_target+.5*(old_v+new_v)*dt
        else:
            self.arm_blocked = False
        clipped = np.clip(candidate, self.arm_limits[:, 0], self.arm_limits[:, 1])
        new_v[np.abs(clipped-candidate) > 1e-12] = 0.
        self.arm_target = clipped
        self.arm_velocity = new_v
        self.arm_acceleration = (new_v-old_v)/dt

    def outer(self, d):
        m, adr = self.m, self.vadr
        mujoco.mj_fullM(m, d, self.full)
        r = d.xmat[self.base].reshape(3, 3)
        error = self.p_ref-d.xpos[self.base]
        if self.state == 'FLIGHT' and not self.saturated:
            self.integral = np.clip(self.integral+self.period*error, -.25, .25)
        a = (self.a_ref+np.array([2.25, 2.25, 4.])*error
             +np.array([3., 3., 4.])*(self.v_ref-d.qvel[adr:adr+3])
             +np.array([.15, .15, .5])*self.integral)
        a = np.clip(a, [-1.2, -1.2, -1.], [1.2, 1.2, 1.])
        qacc = np.zeros(m.nv)
        qacc[adr:adr+3], qacc[self.arm_vadr] = a, self.arm_acceleration
        required = self.full@qacc+d.qfrc_bias
        force = required[adr:adr+3].copy()
        horizontal = np.linalg.norm(force[:2])
        max_horizontal = max(0., force[2])*math.tan(MAX_TILT*.8)
        force[:2] *= min(1., max_horizontal/max(1e-12, horizontal))
        z = force/np.linalg.norm(force)
        y = np.cross(z, [math.cos(self.yaw), math.sin(self.yaw), 0.])
        y /= np.linalg.norm(y)
        desired = np.column_stack((np.cross(y, z), y, z))
        skew = .5*(desired.T@r-r.T@desired)
        e = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        omega = d.qvel[adr+3:adr+6]
        desired_omega = r.T@np.array([0., 0., self.yaw_rate])
        alpha = (-np.array([64., 64., 36.])*e-np.array([16., 16., 12.])*(omega-desired_omega)
                 +r.T@np.array([0., 0., self.yaw_acceleration]))
        qacc[adr+3:adr+6] = alpha
        required = self.full@qacc+d.qfrc_bias
        lever = m.body_pos[self.rotors]
        allocation = np.vstack((np.ones(4), lever[:, 1], -lever[:, 0], -self.spin*self.km))
        wanted = self.spool*np.r_[np.dot(required[adr:adr+3], r[:, 2]), required[adr+3:adr+6]]
        self.command, self.saturated = allocate(allocation, wanted, self.cap)
        self.target_speed = self.spin*np.sqrt(self.command/self.kf)
        self.arm_feedforward = required[self.arm_vadr].copy()
        if self.state == 'GROUND':
            self.command[:] = 0.
            self.target_speed[:] = 0.

    def apply_inner(self, d):
        super().apply_inner(d)
        self.arm_ctrl_requested = self.arm_target+(self.kv*self.arm_velocity+self.arm_feedforward)/self.kp
        limits = self.m.actuator_ctrlrange[self.arm_ids]
        d.ctrl[self.arm_ids] = np.clip(self.arm_ctrl_requested, limits[:, 0], limits[:, 1])

    def step(self, d, keys=None):
        keys = self.last_input if keys is None else keys
        if d.time+1e-10 >= self.next_outer:
            self.update_reference(d, keys)
            self.outer(d)
            self.next_outer = float(d.time)+self.period
        self.apply_inner(d)

    def safety(self, d):
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            return 'non-finite state'
        r = d.xmat[self.base].reshape(3, 3)
        if r[2, 2] < math.cos(MAX_TILT):
            return 'actual tilt exceeds 10 degrees'
        for contact in d.contact[:d.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if contact.dist < -1e-5 and not (self.ground in pair and pair & self.feet):
                return 'unexpected collision'
        if np.max(np.abs(d.qpos[self.arm_qadr]-self.arm_target)) > math.radians(12.):
            return 'arm tracking error exceeds 12 degrees'
        return None


def model_audit(m, d):
    old = mujoco.MjModel.from_xml_path(str(SOURCE))
    od = mujoco.MjData(old)
    mujoco.mj_resetDataKeyframe(old, od, old.key('arm_sequence_ready').id)
    for field in ('body_parentid', 'body_mass', 'body_pos', 'body_quat', 'body_ipos',
                  'body_inertia', 'body_iquat', 'jnt_type', 'jnt_axis', 'jnt_range',
                  'jnt_pos', 'dof_damping', 'dof_frictionloss', 'geom_pos', 'geom_quat',
                  'geom_size', 'geom_margin', 'geom_solref', 'geom_solimp',
                  'geom_contype', 'geom_conaffinity', 'exclude_signature', 'qpos0',
                  'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange',
                  'actuator_forcerange', 'actuator_gear', 'actuator_trnid'):
        assert np.array_equal(getattr(m, field), getattr(old, field)), field
    assert np.array_equal(d.qpos, od.qpos) and np.array_equal(d.qvel, od.qvel)
    assert np.array_equal(d.ctrl, od.ctrl)
    assert (m.nq, m.nv, m.nu, m.neq) == (17, 16, 14, 0)
    assert np.all(m.body_mocapid < 0) and np.array_equal(m.opt.gravity, [0, 0, -9.81])
    return {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'new_xml_sha256': hashlib.sha256(XML.read_bytes()).hexdigest(),
            'all_original_physics_actuators_and_contacts_preserved': True,
            'initial_state_and_zero_velocities_preserved': True}


def metrics(m, d, c):
    r = d.xmat[c.base].reshape(3, 3)
    return {'time_s': float(d.time), 'state': c.state,
            'base_x_m': float(d.xpos[c.base, 0]), 'base_y_m': float(d.xpos[c.base, 1]),
            'base_z_m': float(d.xpos[c.base, 2]), 'target_x_m': float(c.p_ref[0]),
            'target_y_m': float(c.p_ref[1]), 'target_z_m': float(c.p_ref[2]),
            'horizontal_error_m': float(np.linalg.norm(d.xpos[c.base, :2]-c.p_ref[:2])),
            'height_error_m': float(d.xpos[c.base, 2]-c.p_ref[2]),
            'horizontal_speed_m_s': float(np.linalg.norm(d.qvel[:2])),
            'vertical_speed_m_s': float(d.qvel[2]),
            'tilt_deg': float(np.rad2deg(np.arccos(np.clip(r[2, 2], -1., 1.)))),
            'yaw_deg': float(np.rad2deg(np.arctan2(r[1, 0], r[0, 0]))),
            'target_yaw_deg': float(np.rad2deg(c.yaw)),
            'actual_yaw_rate_deg_s': float(np.rad2deg((r@d.qvel[c.vadr+3:c.vadr+6])[2])),
            'arm_max_error_deg': float(np.max(np.abs(np.rad2deg(d.qpos[c.arm_qadr]-c.arm_target))))}


class Window:
    def __init__(self, m, d, c, visible=True):
        if not glfw.init():
            raise RuntimeError('Existing GLFW could not initialize; no packages changed')
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.window = glfw.create_window(1280, 900, 'X500 + SO101 Keyboard Control', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Cannot create existing GLFW OpenGL window')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.keys = Keys()
        glfw.set_key_callback(self.window, self.keys.key)
        glfw.set_window_focus_callback(self.window, self.keys.focus)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0., 0., .8]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.4, 140., -18.
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0
        self.opt.sitegroup[4] = 0
        self.scene = mujoco.MjvScene(m, maxgeom=2000)
        self.context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.mouse = [0., 0.]
        self.drag = False
        self.last_print = 0.
        def button(window, button, action, mods):
            self.drag = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            self.mouse[:] = glfw.get_cursor_pos(window)
        def cursor(window, x, y):
            dx, dy = x-self.mouse[0], y-self.mouse[1]
            self.mouse[:] = [x, y]
            if self.drag:
                height = max(1, glfw.get_window_size(window)[1])
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
                mujoco.mjv_moveCamera(m, action, dx/height, dy/height, self.scene, self.cam)
        def scroll(window, x, y):
            mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0., -.05*y, self.scene, self.cam)
        glfw.set_mouse_button_callback(self.window, button)
        glfw.set_cursor_pos_callback(self.window, cursor)
        glfw.set_scroll_callback(self.window, scroll)

    def draw(self, m, d, c):
        width, height = glfw.get_framebuffer_size(self.window)
        if not width or not height:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        self.cam.lookat[:] = d.xpos[c.base]+[0., 0., -.15]
        mujoco.mjv_updateScene(m, d, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        mujoco.mjr_render(viewport, self.scene, self.context)
        state = metrics(m, d, c)
        title = f'{c.state}'+(' | PAUSED' if self.keys.paused else '')
        left = ('State\nHeight / target (m)\nXY speed (m/s)\nYaw / target (deg)\n'
                'Arm target (deg)\nArm actual (deg)\nRotor speed (rad/s)\nSaturation\nStatus')
        actual = np.rad2deg(d.qpos[c.arm_qadr])
        target = np.rad2deg(c.arm_target)
        sat = np.any(np.abs(d.actuator_force[c.arm_ids]) >= m.actuator_forcerange[c.arm_ids, 1]-.00001)
        text = (f'{title}\n{state["base_z_m"]-c.ground_z:.3f} / {c.p_ref[2]-c.ground_z:.3f}\n'
                f'{state["horizontal_speed_m_s"]:.3f}\n{state["yaw_deg"]:.1f} / {np.rad2deg(c.yaw):.1f}\n'
                f'{np.round(target, 1).tolist()}\n{np.round(actual, 1).tolist()}\n'
                f'{np.round(d.qvel[c.rotor_vadr], 1).tolist()}\n'
                f'arm={bool(sat)} allocation={c.saturated}\n{c.fault or c.message}')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, left, text, self.context)
        help_text = ('Hold W/S: up/down | Arrows: forward/back/left/right (body heading)\n'
                     'Hold Q/E: yaw left/right | 1..6: joint + | Shift+1..6: joint -\n'
                     'U: verified unfold | F: request verified fold | L: fold + land\n'
                     'Space: physics pause/resume | R: ground reset | Esc: close\n'
                     'Release keys: brake and hold | Mouse left drag: rotate | Wheel: zoom')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport, help_text, '', self.context)
        glfw.swap_buffers(self.window)

    def close(self):
        self.context.free()
        # MjvScene owns its native allocation through the Python destructor.
        # Unlike MjrContext, this installed binding has no .free() method.
        del self.scene
        glfw.destroy_window(self.window)
        glfw.terminate()


def run_window(window_test=False):
    m, d = load()
    c = KeyboardFlight(m, d)
    ui = Window(m, d, c)
    frames, first = 0, time.perf_counter()
    window_result = None
    print('Loaded:', XML, flush=True)
    print('Ground idle. Hold W to start; arrow keys translate; 1..6 / Shift control joints.', flush=True)
    print('Click this window to focus. L: verified fold then landing. Space: pause. R: reset.', flush=True)
    try:
        while not glfw.window_should_close(ui.window):
            wall = time.perf_counter()
            glfw.poll_events()
            if ui.keys.reset:
                mujoco.mj_resetDataKeyframe(m, d, m.key('keyboard_ready').id)
                mujoco.mj_forward(m, d)
                c = KeyboardFlight(m, d)
                ui.keys.reset = False
                ui.keys.clear()
            if not ui.keys.paused:
                for _ in range(round(1./(60.*m.opt.timestep))):
                    c.step(d, ui.keys)
                    mujoco.mj_step(m, d)
                    mujoco.mj_forward(m, d)
                    fault = c.safety(d)
                    if fault:
                        c.fault = fault
                        c.transition('FAULT', d, fault)
                        ui.keys.paused = True
                        ui.keys.clear()
                        print('FAULT: physics paused:', fault, flush=True)
                        break
            ui.draw(m, d, c)
            frames += 1
            if window_test and frames >= 90:
                break
            delay = 1./60.-(time.perf_counter()-wall)
            if delay > 0.:
                time.sleep(delay)
        if window_test:
            width, height = glfw.get_framebuffer_size(ui.window)
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), ui.context)
            write_png(HERE/'keyboard_window_preview.png', rgb[::-1])
            window_result = {'passed': frames >= 90 and c.fault is None,
                      'visible_glfw_window_created_and_rendered': True, 'frames': frames,
                      'wall_time_s': time.perf_counter()-first, 'physics_time_s': float(d.time),
                      'framebuffer_px': [width, height], 'state': c.state,
                      'note': 'Rendering smoke test only; no real human keyboard test claimed.'}
    finally:
        ui.close()
    if window_result is not None:
        window_result['window_and_renderer_clean_shutdown'] = True
        (HERE/'keyboard_window_validation.json').write_text(json.dumps(window_result, indent=2), encoding='utf-8')
        print(json.dumps(window_result, indent=2), flush=True)


def run_test():
    m, d = load()
    audit = model_audit(m, d)
    c, keys = KeyboardFlight(m, d), Keys()
    rows, phases = [], {}
    peak_arm, peak_speed, peak_shaft, peak_force = np.zeros(6), np.zeros(4), np.zeros(4), np.zeros(4)
    arm_saturated, shaft_saturated = np.zeros(6, dtype=int), np.zeros(4, dtype=int)
    clipping = allocation_sat = 0
    max_formula_error = max_limit_excess = 0.
    peak_xy_speed = peak_z_speed = peak_yaw_rate = peak_tilt = 0.
    hover_stable_since = None
    stable_hover_duration = 0.
    fault = None
    snapshots = []
    phase, phase_start = 'idle', 0.
    phase_events = []
    tested_joints, digit = [], 0
    extra = {}
    started = time.perf_counter()

    def switch(label):
        nonlocal phase, phase_start
        phase, phase_start = label, float(d.time)
        keys.clear()
        print(f'keyboard: t={d.time:.3f} phase={label} state={c.state} z={d.xpos[c.base,2]:.4f}', flush=True)
        phase_events.append({'time_s': float(d.time), 'phase': label})
        snapshots.append({'phase': label, 'time_s': float(d.time), 'qpos': d.qpos.tolist()})

    def press(*values):
        for key in values:
            keys.key(None, key, 0, glfw.PRESS, 0)

    for step in range(round(180./m.opt.timestep)):
        elapsed = float(d.time)-phase_start
        if step % 16 == 0:
            if phase == 'idle' and elapsed >= 2.:
                extra['idle_rotor_speed_rad_s'] = np.abs(d.qvel[c.rotor_vadr]).tolist()
                extra['idle_thrust_N'] = c.force.tolist()
                switch('cancel_start'); press(glfw.KEY_W)
            elif phase == 'cancel_start' and elapsed >= .6:
                switch('cancel_wait')
            elif phase == 'cancel_wait' and elapsed >= 2.:
                extra['cancelled_start_state'] = c.state
                switch('climb'); press(glfw.KEY_W)
            elif phase == 'climb' and d.xpos[c.base, 2] > 1.5:
                switch('hover_10s')
            elif phase == 'hover_10s' and elapsed >= 14.:
                switch('descend_manual'); press(glfw.KEY_S)
            elif phase == 'descend_manual' and elapsed >= 2.:
                switch('descent_hold')
            elif phase == 'descent_hold' and elapsed >= 14.:
                switch('forward'); press(glfw.KEY_UP)
            elif phase == 'forward' and elapsed >= 2.:
                switch('forward_brake')
            elif phase == 'forward_brake' and elapsed >= 3.:
                switch('backward'); press(glfw.KEY_DOWN)
            elif phase == 'backward' and elapsed >= 3.:
                switch('left'); keys.clear(); press(glfw.KEY_LEFT)
            elif phase == 'left' and elapsed >= 3.:
                switch('right'); press(glfw.KEY_RIGHT)
            elif phase == 'right' and elapsed >= 4.:
                switch('yaw_left'); press(glfw.KEY_Q)
            elif phase == 'yaw_left' and elapsed >= 2.:
                switch('forward_after_yaw'); press(glfw.KEY_UP)
            elif phase == 'forward_after_yaw' and elapsed >= 2.:
                switch('yaw_right'); press(glfw.KEY_E)
            elif phase == 'yaw_right' and elapsed >= 2.:
                switch('combined'); press(glfw.KEY_W, glfw.KEY_UP, glfw.KEY_LEFT)
            elif phase == 'combined' and elapsed >= 1.:
                switch('opposites'); press(glfw.KEY_W, glfw.KEY_S, glfw.KEY_LEFT, glfw.KEY_RIGHT,
                                           glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_Q, glfw.KEY_E)
            elif phase == 'opposites' and elapsed >= 2.:
                switch('focus_loss'); press(glfw.KEY_UP)
            elif phase == 'focus_loss' and elapsed >= .5:
                keys.focus(None, False)
                extra['focus_loss_cleared_keys'] = len(keys.down) == 0
                switch('focus_hold')
            elif phase == 'focus_hold' and elapsed >= 3.:
                keys.focus(None, True)
                switch('unfold'); press(glfw.KEY_U)
            elif phase == 'unfold' and elapsed >= .1 and not c.path:
                extra['unfold_target_deg'] = np.rad2deg(c.arm_target).tolist()
                if np.max(np.abs(c.arm_target-DOWN)) > .01:
                    fault = 'Unfold path refused: '+c.message
                    break
                switch('joint_plus'); press(glfw.KEY_1+digit)
            elif phase == 'joint_plus' and elapsed >= (11. if digit == 5 else .7):
                switch('joint_hold')
            elif phase == 'joint_hold' and elapsed >= .7:
                tested_joints.append({'joint': ARM_NAMES[digit], 'plus_target_deg': float(np.rad2deg(c.arm_target[digit]))})
                switch('joint_minus'); press(glfw.KEY_LEFT_SHIFT, glfw.KEY_1+digit)
            elif phase == 'joint_minus' and elapsed >= (11. if digit == 5 else .7):
                switch('joint_minus_hold')
            elif phase == 'joint_minus_hold' and elapsed >= .7:
                tested_joints[-1]['minus_target_deg'] = float(np.rad2deg(c.arm_target[digit]))
                digit += 1
                if digit < 6:
                    switch('joint_plus'); press(glfw.KEY_1+digit)
                else:
                    switch('land_request'); press(glfw.KEY_L)
            elif phase == 'land_request' and elapsed >= .1:
                switch('landing_wait')
            elif phase == 'landing_wait' and c.state == 'GROUND':
                switch('landed_5s')
            elif phase == 'landed_5s' and elapsed >= 5.:
                break
        c.step(d, keys)
        mujoco.mj_forward(m, d)
        actual_speed = d.qvel[c.rotor_vadr]
        aero = d.actuator_force[c.aero_ids]
        shaft = d.actuator_force[c.shaft_ids]
        arm = d.actuator_force[c.arm_ids]
        peak_arm, peak_speed = np.maximum(peak_arm, np.abs(arm)), np.maximum(peak_speed, np.abs(actual_speed))
        peak_shaft, peak_force = np.maximum(peak_shaft, np.abs(shaft)), np.maximum(peak_force, np.abs(aero))
        arm_saturated += np.abs(arm) >= m.actuator_forcerange[c.arm_ids, 1]-1e-8
        shaft_saturated += np.abs(shaft) >= c.torque_cap-1e-8
        allocation_sat += int(c.saturated)
        clipping += int(np.max(np.abs(c.arm_ctrl_requested-d.ctrl[c.arm_ids])) > 1e-8)
        current = metrics(m, d, c)
        peak_xy_speed = max(peak_xy_speed, current['horizontal_speed_m_s'])
        peak_z_speed = max(peak_z_speed, abs(current['vertical_speed_m_s']))
        peak_yaw_rate = max(peak_yaw_rate, abs(current['actual_yaw_rate_deg_s']))
        peak_tilt = max(peak_tilt, current['tilt_deg'])
        stable = (current['horizontal_error_m'] < .03 and abs(current['height_error_m']) < .03
                  and current['horizontal_speed_m_s'] < .03 and abs(current['vertical_speed_m_s']) < .03
                  and current['tilt_deg'] < 3. and d.ncon == 0)
        if phase in ('hover_10s', 'descent_hold'):
            hover_stable_since = (float(d.time) if hover_stable_since is None else hover_stable_since) if stable else None
            stable_hover_duration = 0. if hover_stable_since is None else float(d.time)-hover_stable_since
            extra[phase+'_continuous_stable_s'] = stable_hover_duration
        else:
            hover_stable_since = None
        max_formula_error = max(max_formula_error, float(np.max(np.abs(aero-c.spin*c.kf*actual_speed*np.abs(actual_speed)))))
        limits = m.jnt_range[c.arm_joints]
        max_limit_excess = max(max_limit_excess, float(np.max(limits[:,0]-d.qpos[c.arm_qadr])),
                              float(np.max(d.qpos[c.arm_qadr]-limits[:,1])))
        assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        fault = c.safety(d)
        if np.any(np.abs(actual_speed) > c.max_speed+1e-6):
            fault = 'Source physical rotor speed cap exceeded'
        if fault:
            break
        if step % 40 == 0:
            row = metrics(m, d, c)
            row['phase'] = phase
            row['contacts'] = int(d.ncon)
            row['skid_support_N'] = c.ground_force
            row['reference_horizontal_speed_m_s'] = float(np.linalg.norm(c.v_ref[:2]))
            row['reference_vertical_speed_m_s'] = float(c.v_ref[2])
            row['reference_yaw_rate_deg_s'] = float(np.rad2deg(c.yaw_rate))
            row['message'] = c.message
            for i, name in enumerate(ARM_NAMES):
                row[name+'_target_deg'] = float(np.rad2deg(c.arm_target[i]))
                row[name+'_actual_deg'] = float(np.rad2deg(d.qpos[c.arm_qadr[i]]))
                row[name+'_torque_Nm'] = float(arm[i])
            for i in range(4):
                row[f'rotor_{i}_actual_speed_rad_s'] = float(actual_speed[i])
                row[f'rotor_{i}_target_speed_rad_s'] = float(c.target_speed[i])
                row[f'rotor_{i}_shaft_torque_Nm'] = float(shaft[i])
                row[f'rotor_{i}_thrust_N'] = float(aero[i])
            rows.append(row)
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
    for label in sorted(set(row['phase'] for row in rows)):
        section = [row for row in rows if row['phase'] == label]
        phases[label] = {'start_s': section[0]['time_s'], 'end_s': section[-1]['time_s'],
                         'max_height_error_m': max(abs(row['height_error_m']) for row in section),
                         'max_horizontal_error_m': max(row['horizontal_error_m'] for row in section),
                         'max_horizontal_speed_m_s': max(row['horizontal_speed_m_s'] for row in section),
                         'max_tilt_deg': max(row['tilt_deg'] for row in section),
                         'max_arm_error_deg': max(row['arm_max_error_deg'] for row in section),
                         'first_position_m': [section[0]['base_x_m'], section[0]['base_y_m'], section[0]['base_z_m']],
                         'last_position_m': [section[-1]['base_x_m'], section[-1]['base_y_m'], section[-1]['base_z_m']]}
    warnings = {str(mujoco.mjtWarning(i)): int(d.warning[i].number) for i in range(len(d.warning))}
    passed = (fault is None and phase == 'landed_5s' and d.time-phase_start >= 5.
              and not any(warnings.values()) and max_limit_excess < 1e-6 and max_formula_error < 1e-10
              and not np.any(arm_saturated) and not np.any(shaft_saturated) and allocation_sat == 0
              and len(tested_joints) == 6 and np.max(np.abs(c.force)) < 1e-8
              and peak_xy_speed <= XY_SPEED+1e-6 and peak_z_speed <= Z_SPEED+1e-6
              and peak_yaw_rate <= math.degrees(YAW_SPEED)+1e-6 and peak_tilt <= math.degrees(MAX_TILT)
              and extra.get('hover_10s_continuous_stable_s',0.) >= 10.
              and extra.get('descent_hold_continuous_stable_s',0.) >= 10.
              and c.ground_force > .95*c.mass*c.g)
    result = {'passed': bool(passed), 'fault': fault, 'mujoco_version': mujoco.__version__,
              'duration_s': float(d.time), 'wall_time_s': time.perf_counter()-started, 'audit': audit,
              'phase_events': phase_events, 'phases': phases, 'states': c.events, 'extra_checks': extra,
              'joints': tested_joints, 'peak_arm_torque_Nm': peak_arm.tolist(),
              'peak_rotor_speed_rad_s': peak_speed.tolist(), 'peak_shaft_torque_Nm': peak_shaft.tolist(),
              'peak_thrust_N': peak_force.tolist(), 'arm_saturation_steps': arm_saturated.tolist(),
              'shaft_saturation_steps': shaft_saturated.tolist(), 'allocation_saturation_steps': allocation_sat,
              'ctrl_clipping_steps': clipping, 'actual_speed_force_formula_error_N': max_formula_error,
              'peak_actual_horizontal_speed_m_s': peak_xy_speed,
              'peak_actual_vertical_speed_m_s': peak_z_speed,
              'peak_actual_yaw_rate_deg_s': peak_yaw_rate, 'peak_actual_tilt_deg': peak_tilt,
              'max_joint_limit_excess_deg': float(np.rad2deg(max_limit_excess)), 'warnings': warnings,
              'final_actual_speed_rad_s': d.qvel[c.rotor_vadr].tolist(), 'final_thrust_N': c.force.tolist(),
              'final_skid_support_N': c.ground_force, 'weight_N': c.mass*c.g,
              'final_arm_deg': np.rad2deg(d.qpos[c.arm_qadr]).tolist(), 'snapshots': snapshots,
              'keyboard_test_scope': 'Deterministic physical key-code events sent to the same input handler; not a real keyboard device test.'}
    with (HERE/'keyboard_control_timeseries.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (HERE/'keyboard_control_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('snapshots','phases')}, indent=2), flush=True)
    return bool(passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--window-test', action='store_true')
    args = parser.parse_args()
    if args.test:
        if not run_test():
            raise SystemExit('Keyboard control validation did not pass; see report')
    else:
        run_window(args.window_test)


if __name__ == '__main__':
    main()

