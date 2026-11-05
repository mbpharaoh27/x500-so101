"""Native MuJoCo viewer; Windows foreground-only chord polling, no keyboard hook.

Normal run: click 3D, release keys, then Ctrl+W/S, Ctrl+I/K, Ctrl+E/R.
No strafe bindings. Ctrl/Alt+1..6: positive/negative arm joint velocity.
Ctrl+U/F: unfold/fold; Ctrl+Enter: land; Ctrl+F10: pause; Ctrl+F11: reset.
Uses the original V07 flight/shaft controller with RAM-only teaching references.
Ctrl+T/Y/B/X record/finish/play/abort. No route files are written.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, deque
import ctypes
from ctypes import wintypes as wt
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

import keyboard_control_x500_so101 as core

HERE = Path(__file__).resolve().parent
VK_CTRL, VK_ALT, VK_SHIFT = 0x11, 0x12, 0x10
VK_LBUTTON, VK_RBUTTON, VK_MBUTTON = 1, 2, 4
VK_ENTER, VK_F10, VK_F11 = 0x0D, 0x79, 0x7A
FLIGHT = {ord('W'): core.glfw.KEY_W, ord('S'): core.glfw.KEY_S,
          ord('I'): core.glfw.KEY_UP, ord('K'): core.glfw.KEY_DOWN,
          ord('E'): core.glfw.KEY_Q, ord('R'): core.glfw.KEY_E}
SINGLE = {ord('U'): 'unfold', ord('F'): 'fold', VK_ENTER: 'land',
          VK_F10: 'pause', VK_F11: 'reset', ord('T'): 'record',
          ord('Y'): 'finish', ord('B'): 'replay', ord('X'): 'cancel'}
WATCH = set(FLIGHT) | set(SINGLE) | set(range(ord('1'), ord('6')+1))
MODIFIERS = {VK_CTRL, VK_ALT, VK_SHIFT, 0x5B, 0x5C}
GLFW_SINGLE = {core.glfw.KEY_U: 'unfold', core.glfw.KEY_F: 'fold',
               core.glfw.KEY_ENTER: 'land', core.glfw.KEY_F10: 'pause',
               core.glfw.KEY_F11: 'reset', core.glfw.KEY_T: 'record',
               core.glfw.KEY_Y: 'finish', core.glfw.KEY_B: 'replay',
               core.glfw.KEY_X: 'cancel'}
ACTION_KEYS = {'unfold': core.glfw.KEY_U, 'fold': core.glfw.KEY_F,
               'land': core.glfw.KEY_L}
RECOGNIZED_GLFW = set(FLIGHT) | set(GLFW_SINGLE) | set(range(ord('1'), ord('6')+1))


class Chords:
    """Pure state translation, tested without a GUI or OS keyboard injection."""
    def __init__(self):
        self.keys = core.Keys()
        self.keys.focused = False
        self.armed = False
        self.wait_release = True
        self.previous = set()
        self.message = 'Click the 3D view, then release all control keys.'
        self.last_chord = '-'
        self.accepted = Counter()

    def disarm(self, reason):
        self.armed = False
        self.wait_release = True
        self.previous.clear()
        self.keys.focused = False
        self.keys.clear()
        self.message = reason

    def click(self, inside):
        self.disarm('Release control keys to enable.' if inside else
                    'MENU MODE: click 3D again to enable motion.')
        self.armed = bool(inside)

    def update(self, down, foreground=True, paused=False, events=()):
        commands = set()
        if not foreground:
            self.disarm('Focus lost: braking to hold. Click 3D to re-enable.')
            return commands
        if not self.armed:
            self.keys.focused = False
            self.keys.clear()
            return commands
        if self.wait_release:
            self.keys.clear()
            if not down & (WATCH | MODIFIERS):
                self.wait_release = False
                self.message = 'CONTROL ACTIVE'
            self.previous = down.copy()
            return commands
        ctrl = VK_CTRL in down and not down & {VK_ALT, VK_SHIFT, 0x5B, 0x5C}
        alt = VK_ALT in down and not down & {VK_CTRL, VK_SHIFT, 0x5B, 0x5C}
        # Discrete native callback events supplement polling for brief taps.
        # They never latch a continuous motion or replace release polling.
        commands.update(events)
        new = down-self.previous
        if ctrl:
            commands.update(SINGLE[v] for v in new if v in SINGLE)
        self.previous = down.copy()
        mapped = set()
        if not paused:
            if ctrl:
                mapped.update(value for key, value in FLIGHT.items() if key in down)
            if ctrl or alt:
                mapped.update(core.glfw.KEY_1+i for i in range(6) if ord('1')+i in down)
                if alt and mapped:
                    mapped.add(core.glfw.KEY_LEFT_SHIFT)
        self.keys.down = mapped
        self.keys.focused = not paused
        if not paused:
            self.keys.edges.update(ACTION_KEYS[a] for a in commands if a in ACTION_KEYS)
        labels = []
        if ctrl or alt:
            prefix = 'Ctrl+' if ctrl else 'Alt+'
            for v in sorted(down & WATCH):
                label = prefix+({VK_ENTER: 'Enter', VK_F10: 'F10', VK_F11: 'F11'}.get(v) or chr(v))
                labels.append(label)
                if v in new:
                    self.accepted[label] += 1
        if labels:
            self.last_chord = ', '.join(labels)
        for action in commands:
            self.accepted['action:'+action] += 1
        return commands


class ForegroundKeys:
    """Documented Win32 HWND/APIs only; never a private GLFW viewer handle."""
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('This native-viewer input adapter requires Windows.')
        self.u = ctypes.WinDLL('user32', use_last_error=True)
        self.u.GetForegroundWindow.argtypes = []
        self.u.GetForegroundWindow.restype = wt.HWND
        self.u.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        self.u.GetWindowThreadProcessId.restype = wt.DWORD
        self.u.IsWindowVisible.argtypes = [wt.HWND]
        self.u.IsWindowVisible.restype = wt.BOOL
        self.u.IsWindow.argtypes = [wt.HWND]
        self.u.IsWindow.restype = wt.BOOL
        self.u.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        self.u.GetWindowTextW.restype = ctypes.c_int
        self.u.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
        self.u.GetClientRect.restype = wt.BOOL
        self.u.GetCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
        self.u.GetCursorPos.restype = wt.BOOL
        self.u.ScreenToClient.argtypes = [wt.HWND, ctypes.POINTER(wt.POINT)]
        self.u.ScreenToClient.restype = wt.BOOL
        self.u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.u.GetAsyncKeyState.restype = ctypes.c_short
        self.u.GetKeyState.argtypes = [ctypes.c_int]
        self.u.GetKeyState.restype = ctypes.c_short
        self.enum_type = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        self.u.EnumWindows.argtypes = [self.enum_type, wt.LPARAM]
        self.u.EnumWindows.restype = wt.BOOL
        self.pid, self.hwnd = os.getpid(), None
        self.mouse_was_down = False

    def discover(self):
        found = []
        @self.enum_type
        def visit(hwnd, _):
            pid = wt.DWORD()
            self.u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == self.pid and self.u.IsWindowVisible(hwnd):
                title = ctypes.create_unicode_buffer(512)
                self.u.GetWindowTextW(hwnd, title, len(title))
                if title.value.startswith('MuJoCo'):
                    found.append(int(hwnd))
            return True
        self.u.EnumWindows(visit, 0)
        if len(found) == 1:
            self.hwnd = found[0]
        elif len(found) > 1:
            self.hwnd = None
        return self.hwnd

    def foreground(self):
        if not self.hwnd or not self.u.IsWindow(self.hwnd):
            self.discover()
        if not self.hwnd:
            return False
        owner = wt.DWORD()
        if not self.u.GetWindowThreadProcessId(self.hwnd, ctypes.byref(owner)) or owner.value != self.pid:
            self.hwnd = None
            return False
        return bool(self.u.GetForegroundWindow() == self.hwnd)

    def sample(self):
        if not self.foreground():
            return set(), False
        down = {v for v in WATCH | MODIFIERS | {1, 2, 4}
                if self.u.GetAsyncKeyState(v) & 0x8000}
        # Fail closed if focus changed during polling.
        return (down, True) if self.foreground() else (set(), False)

    def inside(self, viewport):
        if not self.foreground() or viewport is None:
            return False
        p, rect = wt.POINT(), wt.RECT()
        if not (self.u.GetCursorPos(ctypes.byref(p)) and
                self.u.ScreenToClient(self.hwnd, ctypes.byref(p)) and
                self.u.GetClientRect(self.hwnd, ctypes.byref(rect))):
            return False
        # Both Win32 client coordinates and the MuJoCo viewport are pixels in
        # this GLFW DPI-aware process. Reject inconsistent bounds, never guess.
        if viewport.left+viewport.width > rect.right+2 or viewport.bottom+viewport.height > rect.bottom+2:
            return False
        y = rect.bottom-p.y
        return (viewport.left <= p.x < viewport.left+viewport.width and
                viewport.bottom <= y < viewport.bottom+viewport.height)


class InputMonitor:
    """Keep click/release sampling separate from slow physics/render frames."""
    def __init__(self, reader):
        self.reader = reader
        self.viewport = None
        self.samples = queue.SimpleQueue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        previous, old_mouse = None, False
        while not self.stop.is_set():
            down, foreground = self.reader.sample()
            mouse = bool(down & {1, 2, 4})
            click = None
            if foreground and mouse and not old_mouse:
                click = self.reader.inside(self.viewport) if not down & MODIFIERS and 1 in down else False
            current = (frozenset(down), foreground)
            if current != previous or click is not None:
                self.samples.put((down, foreground, click, mouse))
                previous = current
            old_mouse = mouse
            self.stop.wait(.005)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)


class TeachFlight(core.KeyboardFlight):
    """Only reference ownership changes; the original outer/shaft loops run unchanged."""
    def __init__(self, m, d):
        super().__init__(m, d)
        self.teach = None

    def update_reference(self, d, keys):
        if self.teach is not None and self.teach.mode == 'PLAYING':
            if not keys.focused or keys.down or keys.edges:
                self.teach.stop(d, self, 'Manual input/focus: playback stopped; braking.')
                keys.clear()
            elif self.state != 'FLIGHT' or self.path:
                self.teach.stop(d, self, 'Playback lost flight reference ownership.')
            else:
                self.last_input = keys
                self.support(d)
                self.teach.reference(d, self)
                return
        if self.teach is not None and self.teach.mode in ('RECORDING', 'PLAYING'):
            if keys.edges & {core.glfw.KEY_U, core.glfw.KEY_F} or any(
                    core.glfw.KEY_1 <= x <= core.glfw.KEY_6 for x in keys.down):
                keys.edges.difference_update({core.glfw.KEY_U, core.glfw.KEY_F})
                keys.down = {x for x in keys.down if not core.glfw.KEY_1 <= x <= core.glfw.KEY_6}
                self.teach.message = 'Teaching v1 requires folded arm; joint/helper input refused.'
        super().update_reference(d, keys)


class TeachSession:
    """One RAM-only demonstration, conservative geometry gate, timed quintic replay.

    The path is a chain of straight segments with zero velocity/acceleration at
    retained knots. This deliberately slows corners rather than cutting them.
    No real simulation qpos/qvel is ever assigned here.
    """
    SAMPLE = .02
    RADIUS, HALF_HEIGHT = .45, .30  # folded envelope, including <=10deg tilt
    CLEARANCE, TRACK_ALLOWANCE = .20, .10
    MAX_RECORD_SECONDS = 1200.

    def __init__(self, m):
        self.m = m
        self.mode = 'EMPTY'
        self.message = 'RAM only: Ctrl+T record, Ctrl+Y finish, Ctrl+B replay, Ctrl+X stop.'
        self.rows, self.segments = [], []
        self.fingerprint = None
        self.pending = False
        self.stats = None
        self.last_stats = None
        self.last_summary = None
        self.stable_since = None
        self.next_sample = 0.
        self.clock = 0.
        self.ground = m.geom('ground').id
        self.next_check = 0.
        self.blockers = []
        self.valid = False

    def scene_hash(self):
        h = hashlib.sha256()
        # Include model physics, keyframes and geometry; not filenames or live qpos.
        for name in ('body_pos', 'body_quat', 'body_mass', 'body_inertia', 'body_ipos',
                     'body_iquat', 'jnt_type', 'jnt_range', 'jnt_axis', 'geom_type',
                     'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_size', 'geom_dataid',
                     'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref',
                     'geom_solimp', 'geom_margin', 'mesh_vert', 'mesh_face', 'qpos0',
                     'actuator_gainprm', 'actuator_biasprm', 'actuator_gear',
                     'actuator_ctrlrange', 'actuator_forcerange', 'key_qpos', 'key_ctrl'):
            h.update(np.asarray(getattr(self.m, name)).tobytes())
        h.update(self.m.opt.__repr__().encode())
        return h.hexdigest()

    def refresh_obstacles(self):
        self.blockers = []
        for g in range(self.m.ngeom):
            if g == self.ground or not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                continue
            if self.m.body_weldid[self.m.geom_bodyid[g]] == 0:
                if self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                    raise ValueError('Unsupported static collider: '+str(self.m.geom(g).name))
                self.blockers.append(g)
            elif self.m.body_rootid[self.m.geom_bodyid[g]] != self.m.body('base_link').id:
                raise ValueError('Unmodelled moving obstacle: '+str(self.m.geom(g).name))

    @staticmethod
    def yaw(d, c):
        r = d.xmat[c.base].reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def gap(self, position, d):
        # Upright boxes: distance to a box minus enclosing robot cylinder radius.
        p = np.asarray(position)
        ground_z = float(d.geom_xpos[self.ground, 2])
        gap = float(p[2]-self.HALF_HEIGHT-ground_z)
        for g in self.blockers:
            r = d.geom_xmat[g].reshape(3, 3)
            if not np.allclose(r[2], [0, 0, 1], atol=1e-8):
                raise ValueError('Tilted obstacle is not supported by folded-cylinder gate')
            local = r.T@(p-d.geom_xpos[g]); size = self.m.geom_size[g]
            if abs(local[2]) <= size[2]+self.HALF_HEIGHT:
                gap = min(gap, float(np.linalg.norm(np.maximum(np.abs(local[:2])-size[:2], 0))
                                     - self.RADIUS))
        # Treat finite displayed ground as the operational boundary.
        ground_r = d.geom_xmat[self.ground].reshape(3, 3)
        if not np.allclose(ground_r, np.eye(3)):
            raise ValueError('Tilted ground is not supported')
        bounds = self.m.geom_size[self.ground, :2]
        if np.any(bounds <= 0):
            raise ValueError('Finite operational ground bounds required')
        gap = min(gap, float(np.min(bounds-np.abs(p[:2]-d.geom_xpos[self.ground, :2]))-self.RADIUS))
        return gap

    def stable(self, d, c):
        r = d.xmat[c.base].reshape(3, 3)
        return (c.state == 'FLIGHT' and not c.path and
                np.linalg.norm(d.qvel[c.vadr:c.vadr+3]) < .03 and
                np.linalg.norm(d.qvel[c.vadr+3:c.vadr+6]) < np.deg2rad(3) and
                r[2, 2] > np.cos(np.deg2rad(3)) and
                np.max(np.abs(d.qpos[c.arm_qadr]-core.FOLDED)) < np.deg2rad(1) and
                np.max(np.abs(c.arm_target-core.FOLDED)) < 1e-6)

    def conditions(self, d, c):
        if self.stable_since is None or d.time-self.stable_since < .5 or not self.stable(d,c):
            raise ValueError('Need folded FLIGHT, speed <0.03m/s, tilt <3deg, stable for 0.5s')
        self.refresh_obstacles()
        if self.gap(d.xpos[c.base], d) < self.CLEARANCE+self.TRACK_ALLOWANCE+.02:
            raise ValueError('Insufficient whole-vehicle clearance at current position')

    def row(self, d, c):
        return {'t': float(d.time-self.epoch), 'p': d.xpos[c.base].copy(),
                'v': d.qvel[c.vadr:c.vadr+3].copy(), 'yaw': self.yaw(d,c),
                'arm_target': c.arm_target.copy(), 'arm_actual': d.qpos[c.arm_qadr].copy(),
                'state': c.state}

    def start_record(self, d, c):
        self.conditions(d, c)
        self.epoch = float(d.time)
        self.fingerprint = self.scene_hash()
        self.rows, self.segments = [self.row(d,c)], []
        self.next_sample = float(d.time)+self.SAMPLE
        self.valid, self.pending = False, False
        self.mode = 'RECORDING'
        self.message = 'Recording RAM trajectory. Folded arm only; Ctrl+Y ends at a stable hover.'

    def action(self, action, d, c):
        try:
            if action == 'record':
                if self.mode in ('RECORDING', 'PLAYING', 'BRAKING'):
                    raise ValueError('Finish/stop current operation first')
                self.conditions(d,c)
                if self.rows:
                    self.pending = True
                    self.message = 'Replace RAM route? Ctrl+Y confirms, Ctrl+X cancels (old route retained).'
                else:
                    self.start_record(d,c)
            elif action == 'finish':
                if self.pending:
                    self.start_record(d,c)
                elif self.mode == 'RECORDING':
                    self.conditions(d,c)
                    if d.time-self.epoch < 1.:
                        raise ValueError('Record at least 1 second')
                    if d.time-self.epoch > self.rows[-1]['t']+1e-8:
                        self.rows.append(self.row(d,c))
                    try:
                        self.prepare(d,c)
                    except ValueError:
                        self.mode,self.valid='INVALID',False
                        raise
                    self.mode, self.valid = 'READY', True
                    self.message = f'RAM route checked: {len(self.rows)} samples, {self.duration:.1f}s playback. Return manually to start.'
                else:
                    raise ValueError('Not recording and no replacement confirmation pending')
            elif action == 'replay':
                if self.pending or self.mode in ('RECORDING','PLAYING','BRAKING'):
                    raise ValueError('Finish/stop current operation first')
                if not self.valid or not self.rows:
                    raise ValueError('No valid recorded route')
                self.conditions(d,c)
                self.prepare(d,c)  # Always recompute complete path gate.
                first=self.rows[0]
                start_error=float(np.linalg.norm(d.xpos[c.base]-first['p']))
                if start_error > .05:
                    raise ValueError(f'Return manually to start: error {start_error:.3f}m, allowed 0.05m')
                yaw_error=np.arctan2(np.sin(self.yaw(d,c)-first['yaw']),np.cos(self.yaw(d,c)-first['yaw']))
                if abs(yaw_error)>np.deg2rad(3):
                    raise ValueError('Return to starting yaw within 3deg')
                if np.max(np.abs(d.qpos[c.arm_qadr]-first['arm_actual']))>np.deg2rad(1):
                    raise ValueError('Starting arm angle mismatch')
                self.mode, self.clock = 'PLAYING', 0.
                self.stats = {'samples':0, 'sum_sq':0., 'max_error':0., 'max_z_error':0.,
                              'max_yaw_deg':0., 'max_tilt_deg':0., 'max_arm_deg':0.,
                              'min_gap':float('inf'), 'actual_length':0., 'sum_dt':0.,
                              'saturation_seconds':[0.,0.,0.], 'longest_saturation_seconds':[0.,0.,0.],
                              'current_saturation_seconds':[0.,0.,0.], 'completed':False}
                self.last_p=d.xpos[c.base].copy();self.next_stat=float(d.time)
                self.next_hash_check=float(d.time);self.hash_bad=False
                self.message='Playing RAM route; Ctrl+X or flight input brakes to hold.'
            elif action == 'cancel':
                if self.pending:
                    self.pending=False;self.message='Replacement cancelled; previous RAM route retained.'
                elif self.mode=='RECORDING':
                    self.valid=False;self.mode='INVALID';self.message='Recording aborted; partial route cannot play.'
                elif self.mode=='PLAYING':
                    self.stop(d,c,'Playback cancelled; braking to hold.')
                else:
                    self.message='No active operation to cancel.'
        except (ValueError, FloatingPointError) as e:
            self.message='REFUSED: '+str(e)

    def prepare(self, d, c):
        if self.fingerprint != self.scene_hash():
            self.valid=False
            self.mode='INVALID'
            raise ValueError('Scene/model changed: record again; previous route is invalid')
        self.refresh_obstacles()
        if len(self.rows)<2:
            raise ValueError('Too few samples')
        t=np.array([r['t'] for r in self.rows]);p=np.array([r['p'] for r in self.rows])
        y=np.unwrap([r['yaw'] for r in self.rows])
        if not np.all(np.isfinite(np.c_[t,p,y])) or np.any(np.diff(t)<=0):
            raise ValueError('Non-finite or non-monotonic trajectory')
        for row in self.rows:
            if not all(np.all(np.isfinite(row[field])) for field in ('v','arm_target','arm_actual')):
                raise ValueError('Non-finite recorded velocity/arm state')
            if row['state']!='FLIGHT' or np.max(np.abs(row['arm_target']-core.FOLDED))>1e-6 or np.max(np.abs(row['arm_actual']-core.FOLDED))>np.deg2rad(1):
                raise ValueError('Route is not a folded-arm flight demonstration')
        if np.max(np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(t))>.5:
            raise ValueError('Position discontinuity/overspeed in recording')
        # Time-spaced anchors, with RDP subdivision to <=5mm chord error, and
        # retained stationary intervals. Yaw is scaled to preserve turning knots.
        values=np.c_[p,.4*y]
        keep={0,len(t)-1}
        for i in range(1,len(t)-1):
            if t[i]-t[max(keep)]>=2.:
                keep.add(i)
        def split(a,b):
            if b<=a+1:return
            u=(t[a+1:b]-t[a])/(t[b]-t[a])
            errors=np.linalg.norm(values[a+1:b]-(values[a]+u[:,None]*(values[b]-values[a])),axis=1)
            j=int(np.argmax(errors))+a+1
            if float(errors.max())>.005:
                keep.add(j);split(a,j);split(j,b)
        initial=sorted(keep)
        for a,b in zip(initial[:-1],initial[1:]):split(a,b)
        anchors=sorted(keep);segments=[];total=0.
        for a,b in zip(anchors[:-1],anchors[1:]):
            delta=p[b]-p[a];dy=float(y[b]-y[a]);xy=float(np.linalg.norm(delta[:2]));z=abs(float(delta[2]))
            duration=max(t[b]-t[a], .2, 1.875*xy/.15, 1.875*z/.10,
                         np.sqrt(5.774*float(np.linalg.norm(delta))/.10),
                         1.875*abs(dy)/np.deg2rad(10),np.sqrt(5.774*abs(dy)/np.deg2rad(20)))
            # At most 1cm between probes. Distance is 1-Lipschitz, so an extra
            # 1cm covers the unsampled straight portion between adjacent probes.
            n=max(2,int(np.ceil(np.linalg.norm(delta)/.01))+1)
            for s in np.linspace(0,1,n):
                pos=p[a]+s*delta
                if not .8<=pos[2]-c.ground_z<=core.MAX_HEIGHT:
                    raise ValueError('Route height outside 0.8..2m')
                if self.gap(pos,d)<self.CLEARANCE+self.TRACK_ALLOWANCE+.01:
                    raise ValueError('Whole-vehicle path clearance insufficient; re-teach farther away')
            segments.append((total,duration,p[a].copy(),delta.copy(),float(y[a]),dy))
            total+=duration
        self.segments,self.duration=segments,total
        self.recorded_length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        self.reference_length=float(sum(np.linalg.norm(s[3]) for s in segments))

    def reference(self, d, c):
        try:
            gap=self.gap(d.xpos[c.base],d)
        except ValueError:
            gap=-float('inf')
        if self.hash_bad or np.linalg.norm(d.xpos[c.base]-c.p_ref)>.10 or gap<self.CLEARANCE:
            self.stop(d,c,'Tracking/clearance/model guard stopped playback; braking.')
            c.update_reference(d,core.Keys())
            return
        elapsed=min(self.clock,self.duration)
        seg=next((s for s in self.segments if elapsed<s[0]+s[1]),self.segments[-1])
        start,T,p,delta,y,dy=seg;s=float(np.clip((elapsed-start)/T,0,1))
        h=core.smoothstep(s);v=30*s*s*(1-s)**2/T;a=60*s*(1-s)*(1-2*s)/T**2
        c.p_ref=p+delta*h;c.v_ref=delta*v;c.a_ref=delta*a
        c.yaw=y+dy*h;c.yaw_rate=dy*v;c.yaw_acceleration=dy*a
        c.arm_target=core.FOLDED.copy();c.arm_velocity[:]=0.;c.arm_acceleration[:]=0.
        self.clock+=c.period
        if self.clock>self.duration+2.:
            end_error=float(np.linalg.norm(d.xpos[c.base]-self.rows[-1]['p']))
            self.stats['completed']=end_error<=.10 and np.linalg.norm(d.qvel[c.vadr:c.vadr+3])<.03
            self.stop(d,c,'Playback finished; endpoint hold.', completed=self.stats['completed'])

    def stop(self, d, c, reason, completed=False):
        if self.mode=='PLAYING':
            self.last_stats=self.summary()
            self.last_stats['completed']=bool(completed)
            print('RAM replay result:',json.dumps(self.last_stats,ensure_ascii=True),flush=True)
            self.mode='READY' if completed else 'BRAKING'
            self.brake_start=float(d.time)
            # Keep derivative continuity at handover; normal manual zero-input
            # reference generator ramps these down under its existing limits.
        self.message=reason

    def summary(self):
        if not self.stats:return {}
        out=dict(self.stats)
        out['rms_error']=float(np.sqrt(out.pop('sum_sq')/max(1,out['samples'])))
        out.update(recorded_length_m=self.recorded_length,reference_length_m=self.reference_length,
                   reference_duration_s=self.duration,playback_elapsed_s=self.clock,
                   warnings=[int(w.number) for w in self.last_data.warning])
        out['scope']='Conservative folded-envelope clearance; saturation sampled at 50Hz'
        return out

    def interrupt(self, d, c, reason):
        self.pending=False
        if self.mode=='RECORDING':
            self.mode,self.valid='INVALID',False;self.message=reason+'; partial recording invalid.'
        elif self.mode=='PLAYING':
            self.stop(d,c,reason+'; braking to hold.')

    def post_step(self, d, c):
        self.last_data=d
        if self.stable(d,c):
            if self.stable_since is None:self.stable_since=float(d.time)
        else:self.stable_since=None
        if self.mode=='BRAKING' and np.linalg.norm(c.v_ref)<1e-6 and abs(c.yaw_rate)<1e-6:
            self.mode='READY' if self.valid else 'INVALID'
        if self.mode=='RECORDING' and d.time+1e-9>=self.next_sample:
            if d.time-self.epoch>self.MAX_RECORD_SECONDS:
                self.interrupt(d,c,'20-minute RAM recording limit');return
            if c.state!='FLIGHT' or not c.folded(d):
                self.interrupt(d,c,'Flight/folded condition lost');return
            self.rows.append(self.row(d,c));self.next_sample+=self.SAMPLE
        if self.mode=='PLAYING' and d.time+1e-9>=self.next_stat:
            dt=self.SAMPLE;self.next_stat=float(d.time)+dt
            if d.time>=self.next_hash_check:
                self.hash_bad=self.scene_hash()!=self.fingerprint
                self.next_hash_check=float(d.time)+.5
            r=d.xmat[c.base].reshape(3,3);err=d.xpos[c.base]-c.p_ref
            st=self.stats;st['samples']+=1;st['sum_sq']+=float(err@err)
            st['max_error']=max(st['max_error'],float(np.linalg.norm(err)))
            st['max_z_error']=max(st['max_z_error'],abs(float(err[2])))
            dy=self.yaw(d,c)-c.yaw
            st['max_yaw_deg']=max(st['max_yaw_deg'],float(abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy))))))
            st['max_tilt_deg']=max(st['max_tilt_deg'],float(np.rad2deg(np.arccos(np.clip(r[2,2],-1,1)))))
            st['max_arm_deg']=max(st['max_arm_deg'],float(np.rad2deg(np.max(np.abs(d.qpos[c.arm_qadr]-c.arm_target)))))
            st['min_gap']=min(st['min_gap'],self.gap(d.xpos[c.base],d))
            st['actual_length']+=float(np.linalg.norm(d.xpos[c.base]-self.last_p));self.last_p=d.xpos[c.base].copy()
            st['sum_dt']+=dt
            flags=[bool(c.saturated),bool(np.any(np.abs(d.actuator_force[c.shaft_ids])>=c.torque_cap-1e-6)),
                   bool(np.any(np.abs(d.actuator_force[c.arm_ids])>=self.m.actuator_forcerange[c.arm_ids,1]-1e-6))]
            for i,flag in enumerate(flags):
                st['saturation_seconds'][i]+=dt*flag
                st['current_saturation_seconds'][i]=(st['current_saturation_seconds'][i]+dt) if flag else 0.
                st['longest_saturation_seconds'][i]=max(st['longest_saturation_seconds'][i],st['current_saturation_seconds'][i])
            if max(st['current_saturation_seconds'])>.5:
                self.stop(d,c,'Saturation persisted >0.5s; braking.')


def reset(m, d):
    mujoco.mj_resetDataKeyframe(m, d, m.key('keyboard_ready').id)
    mujoco.mj_forward(m, d)
    return TeachFlight(m, d)


def state_copy(d):
    return float(d.time), d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy()


def same_state(before, d):
    return (before[0] == float(d.time) and
            all(np.array_equal(a, b) for a, b in zip(before[1:], (d.qpos, d.qvel, d.ctrl, d.act))))


def overlay(viewer, m, d, c, router, paused, hard_fault):
    state = core.metrics(m, d, c)
    mode = 'CONTROL ACTIVE' if router.armed and not router.wait_release else 'CONTROL OFF'
    phase = c.state+(' / PAUSED' if paused else '')
    message = c.message.replace('Hold W', 'Hold Ctrl+W').replace('W released', 'Ctrl+W released')
    message = message.replace('Use F first', 'Use Ctrl+F first').replace('F/L', 'Ctrl+F/Enter')
    if c.path:
        message = f'{c.path_kind}: segment {c.path_index+1}/{len(c.path)}; keep this window focused.'
    text = (f'{mode} | {phase}\n'
            f'Height {state["base_z_m"]-c.ground_z:.3f} / target {c.p_ref[2]-c.ground_z:.3f} m\n'
            f'Input: {router.last_chord}\n'
            f'Arm target: {np.round(np.rad2deg(c.arm_target),1).tolist()}\n'
            f'Arm actual: {np.round(np.rad2deg(d.qpos[c.arm_qadr]),1).tolist()}\n'
            f'Rotor rad/s: {np.round(d.qvel[c.rotor_vadr],1).tolist()}\n'
            f'Status: {hard_fault or message}\n'
            f'{router.message}')
    if c.teach is not None:
        teach=c.teach
        text += f'\nTEACH {teach.mode}: {len(teach.rows)} samples | RAM only, lost on exit\n{teach.message}'
        if teach.rows:
            text += '\nRoute start (m): '+str(np.round(teach.rows[0]['p'],3).tolist())
            current=d.xpos[c.base]
            text += '\nCurrent XYZ (m): '+str(np.round(current,3).tolist())
            error=float(np.linalg.norm(current-teach.rows[0]['p']))
            dy=teach.yaw(d,c)-teach.rows[0]['yaw']
            dy=float(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy))))
            text += f'\nStart mismatch: {error:.3f}m / {dy:.1f}deg (limits 0.05m / 3deg)'
        if teach.last_stats:
            s=teach.last_stats
            text += f'\nLast replay: complete={s["completed"]}, RMS={s["rms_error"]:.3f}m, max={s["max_error"]:.3f}m'
    help_text = ('Ctrl+W/S: rise/descend | Ctrl+I/K: forward/back | Ctrl+E/R: yaw L/R\n'
                 'Ctrl+1..6: joint + | Alt+1..6: joint - | NO strafe bindings\n'
                 'Ctrl+U/F: unfold/fold | Ctrl+Enter: land | Ctrl+F10: pause/resume\n'
                 'Ctrl+F11: ground reset | Click 3D to enable; menu/focus loss disables\n'
                 'Ctrl+T: record | Ctrl+Y: finish/confirm replace | Ctrl+B: replay | Ctrl+X: abort\n'
                 'Teach at ~1.2m, folded arm. RAM data LOST ON EXIT.\n'
                 'Native Pause/Run do not drive passive physics. Use Ctrl+F10.')
    viewer.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''),
                      (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, help_text, '')])


def run(window_test=False, duration=180.):
    m, d = core.load()
    # Visual-only quality settings: keep the heavy source meshes and collisions.
    # No shadow map allocation; this does not change physics or XML on disk.
    m.vis.quality.shadowsize = 0
    m.vis.quality.offsamples = 0
    c = TeachFlight(m, d)
    teach = TeachSession(m)
    c.teach = teach
    router, reader = Chords(), ForegroundKeys()
    callback_events = queue.SimpleQueue()
    gui_counts = Counter()
    transitions, observed = [], deque(maxlen=40)
    last_actions = {}
    paused, hard_fault = False, None
    baseline_options = m.opt.__repr__()

    def on_key(key):
        if not reader.foreground() or not router.armed or router.wait_release:
            return
        # This callback executes on the viewer's UI/message thread. GetKeyState
        # reflects modifiers for that queued message, unlike asynchronous state
        # which may already be released when a brief tap is dispatched.
        down = {v for v in MODIFIERS if reader.u.GetKeyState(v) & 0x8000}
        ctrl = VK_CTRL in down and not down & {VK_ALT, VK_SHIFT, 0x5B, 0x5C}
        alt = VK_ALT in down and not down & {VK_CTRL, VK_SHIFT, 0x5B, 0x5C}
        # Only record our chords, not arbitrary user keystrokes.
        if (ctrl or alt) and key in RECOGNIZED_GLFW:
            callback_events.put(('observed', int(key), bool(ctrl), bool(alt)))
        if ctrl and key in GLFW_SINGLE:
            callback_events.put(('action', GLFW_SINGLE[key]))

    started, frames = time.perf_counter(), 0
    max_arm_error, max_tilt, max_height = 0., 0., 0.
    state_before_sync = state_copy(d)
    print('Model:', core.XML, flush=True)
    print('Native viewer: click 3D, release keys, then Ctrl+W to start.', flush=True)
    print('Ctrl+J and Ctrl+H are intentionally UNBOUND. Native menu shortcuts are retained.', flush=True)
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key,
                                     show_left_ui=True, show_right_ui=True) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0., 0., .6]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 140., -18.
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[4] = 0
        monitor = InputMonitor(reader)
        monitor.start()
        live_down, live_foreground = set(), False
        while viewer.is_running():
            wall = time.perf_counter()
            monitor.viewport = viewer.viewport
            commands = set()
            while not monitor.samples.empty():
                down, foreground, click, mouse_down = monitor.samples.get()
                if click is not None:
                    router.click(click)
                    gui_counts['3d_click' if click else 'menu_click'] += 1
                if not foreground and router.armed:
                    gui_counts['focus_loss'] += 1
                guarded_down = down | {VK_SHIFT} if mouse_down else down
                commands.update(router.update(guarded_down, foreground, paused))
                live_down, live_foreground = guarded_down, foreground
            # Rearm after a click-and-release even when no key state changed.
            commands.update(router.update(live_down, live_foreground, paused))
            events = set()
            while not callback_events.empty():
                event = callback_events.get()
                if event[0] == 'action':
                    events.add(event[1])
                else:
                    observed.append({'key': event[1], 'ctrl': event[2], 'alt': event[3],
                                     'simulation_s': float(d.time)})
                    gui_counts['native_key_callback'] += 1
            commands.update(router.update(live_down, live_foreground, paused, events))
            if not router.armed or not live_foreground:
                commands.clear()
            # Poll and callback can observe the same press in adjacent frames.
            # Debounce only discrete actions; continuous commands never latch.
            commands = {a for a in commands if wall-last_actions.get(a, -float('inf')) > .15}
            for action in commands:
                last_actions[action] = wall
                router.last_chord = {'pause':'Ctrl+F10','reset':'Ctrl+F11',
                                     'unfold':'Ctrl+U','fold':'Ctrl+F','land':'Ctrl+Enter',
                                     'record':'Ctrl+T','finish':'Ctrl+Y','replay':'Ctrl+B','cancel':'Ctrl+X'}[action]
            if not router.armed or not live_foreground:
                teach.interrupt(d,c,'Focus/menu input disabled')
            if router.keys.down and teach.pending:
                teach.pending=False
                teach.message='Replacement confirmation cancelled by movement.'
            teach_actions=commands & {'cancel','record','finish','replay'}
            if len(teach_actions)>1 or teach_actions and commands & {'reset','pause','land','unfold','fold'}:
                teach.message='Teaching command conflict: press only one action chord at a time.'
            else:
                for action in teach_actions:
                    if paused or hard_fault:
                        teach.message='Teaching action refused while paused/faulted.'
                    else:
                        teach.action(action,d,c)
            if 'unfold' in commands and c.state != 'FLIGHT':
                c.message = 'Unfold refused: take off and hold above 1 m first.'
            if 'fold' in commands and c.state != 'FLIGHT':
                c.message = 'Fold helper requires FLIGHT; ground pose is already folded.'
            if 'reset' in commands:
                teach.interrupt(d,c,'Ground reset; completed RAM route retained, recheck required')
                with viewer.lock():
                    c = reset(m, d)
                    c.teach=teach
                    teach.stable_since=None
                    if teach.mode=='BRAKING':teach.mode='READY' if teach.valid else 'INVALID'
                hard_fault = None
                paused = False
                if m.opt.__repr__() != baseline_options:
                    paused = True
                    hard_fault = 'Physics settings changed. Restore original settings or restart.'
                router.click(True)
                gui_counts['reset'] += 1
                state_before_sync = state_copy(d)
            elif 'pause' in commands:
                if hard_fault:
                    c.message = 'Fault is latched; use Ctrl+F11 to reset or restart.'
                else:
                    paused = not paused
                    c.path_abort = bool(c.path)
                    router.keys.clear()
                    router.wait_release = True
                    router.keys.focused = False
                    gui_counts['pause' if paused else 'resume'] += 1
            with viewer.lock():
                if not paused:
                    if teach.mode=='PLAYING' and router.armed and live_foreground:
                        router.keys.focused=True  # release gate after pause is not focus loss
                    for _ in range(round(1./(60.*m.opt.timestep))):
                        c.step(d, router.keys)
                        mujoco.mj_step(m, d)
                        mujoco.mj_forward(m, d)
                        teach.post_step(d,c)
                        fault = c.safety(d)
                        if fault:
                            teach.interrupt(d,c,'Physics safety fault: '+fault)
                            paused, hard_fault = True, fault
                            router.disarm('FAULT: Ctrl+F11 resets to ground.')
                            break
                state_before_sync = state_copy(d)
                max_arm_error = max(max_arm_error, core.metrics(m,d,c)['arm_max_error_deg'])
                max_tilt = max(max_tilt, core.metrics(m,d,c)['tilt_deg'])
                max_height = max(max_height, float(d.xpos[c.base,2]-c.ground_z))
            overlay(viewer, m, d, c, router, paused, hard_fault)
            # sync propagates native menu edits. Detect them BEFORE another step.
            viewer.sync()
            with viewer.lock():
                if not same_state(state_before_sync, d):
                    teach.interrupt(d,c,'Native state/control edited')
                    paused = True
                    hard_fault = 'Native state/control edited. Physics paused; Ctrl+F11 restores known state.'
                    router.disarm('Click 3D, release keys, then Ctrl+F11.')
                    gui_counts['native_edit_detected'] += 1
                if m.opt.__repr__() != baseline_options:
                    teach.interrupt(d,c,'Native physics edited')
                    paused = True
                    hard_fault = 'Native physics settings edited; restore original settings or restart.'
                    router.disarm(hard_fault)
                if viewer.m is not m or viewer.d is not d:
                    raise RuntimeError('Viewer model changed; controller stopped before further steps.')
            if c.events and len(c.events) > len(transitions):
                transitions = list(c.events)
            frames += 1
            if window_test and time.perf_counter()-started >= duration:
                break
            delay = 1./60.-(time.perf_counter()-wall)
            if delay > 0:
                time.sleep(delay)
        monitor.close()
        result = {'native_viewer_started': True, 'frames': frames,
                  'simulation_s': float(d.time), 'wall_s': time.perf_counter()-started,
                  'keyboard_scope': 'Actual foreground polling and native callback observations; automated OS input is not a human hardware test.',
                  'input_counts': dict(router.accepted), 'gui_counts': dict(gui_counts),
                  'observed_chords': list(observed), 'transitions': transitions,
                  'final_state': c.state, 'paused': paused, 'fault': hard_fault,
                  'maximum_height_m': max_height, 'maximum_tilt_deg': max_tilt,
                  'maximum_arm_error_deg': max_arm_error,
                  'warnings': [int(w.number) for w in d.warning],
                  'source_sha256': hashlib.sha256((HERE/'keyboard_control_x500_so101.py').read_bytes()).hexdigest(),
                  'strafe_bindings': [], 'original_options_repr': baseline_options}
    if window_test:
        result['native_viewer_clean_shutdown'] = True
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-test', action='store_true', help='Timed native window/input smoke test; console only')
    parser.add_argument('--duration', type=float, default=180., help='Wall seconds for --window-test only')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    run(args.window_test, args.duration)


if __name__ == '__main__':
    main()
