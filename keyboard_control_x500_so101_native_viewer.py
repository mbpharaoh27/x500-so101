"""Native MuJoCo viewer; Windows foreground-only chord polling, no keyboard hook.

Normal run: click 3D, release keys, then Ctrl+W/S, Ctrl+I/K, Ctrl+E/R.
No strafe bindings. Ctrl/Alt+1..6: positive/negative arm joint velocity.
Ctrl+U/F: unfold/fold; Ctrl+Enter: land; Ctrl+F10: pause; Ctrl+F11: reset.
The original V07 controller, XML and actuator limits are imported unchanged.
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
REPORT = HERE / 'native_viewer_window_validation.json'
VK_CTRL, VK_ALT, VK_SHIFT = 0x11, 0x12, 0x10
VK_LBUTTON, VK_RBUTTON, VK_MBUTTON = 1, 2, 4
VK_ENTER, VK_F10, VK_F11 = 0x0D, 0x79, 0x7A
FLIGHT = {ord('W'): core.glfw.KEY_W, ord('S'): core.glfw.KEY_S,
          ord('I'): core.glfw.KEY_UP, ord('K'): core.glfw.KEY_DOWN,
          ord('E'): core.glfw.KEY_Q, ord('R'): core.glfw.KEY_E}
SINGLE = {ord('U'): 'unfold', ord('F'): 'fold', VK_ENTER: 'land',
          VK_F10: 'pause', VK_F11: 'reset'}
WATCH = set(FLIGHT) | set(SINGLE) | set(range(ord('1'), ord('6')+1))
MODIFIERS = {VK_CTRL, VK_ALT, VK_SHIFT, 0x5B, 0x5C}
GLFW_SINGLE = {core.glfw.KEY_U: 'unfold', core.glfw.KEY_F: 'fold',
               core.glfw.KEY_ENTER: 'land', core.glfw.KEY_F10: 'pause',
               core.glfw.KEY_F11: 'reset'}
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


def reset(m, d):
    mujoco.mj_resetDataKeyframe(m, d, m.key('keyboard_ready').id)
    mujoco.mj_forward(m, d)
    return core.KeyboardFlight(m, d)


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
    help_text = ('Ctrl+W/S: rise/descend | Ctrl+I/K: forward/back | Ctrl+E/R: yaw L/R\n'
                 'Ctrl+1..6: joint + | Alt+1..6: joint - | NO strafe bindings\n'
                 'Ctrl+U/F: unfold/fold | Ctrl+Enter: land | Ctrl+F10: pause/resume\n'
                 'Ctrl+F11: ground reset | Click 3D to enable; menu/focus loss disables\n'
                 'Native Pause/Run do not drive passive physics. Use Ctrl+F10.')
    viewer.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''),
                      (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, help_text, '')])


def run(window_test=False, duration=180.):
    m, d = core.load()
    # Visual-only quality settings: keep the heavy source meshes and collisions.
    # No shadow map allocation; this does not change physics or XML on disk.
    m.vis.quality.shadowsize = 0
    m.vis.quality.offsamples = 0
    c = core.KeyboardFlight(m, d)
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
                                     'unfold':'Ctrl+U','fold':'Ctrl+F','land':'Ctrl+Enter'}[action]
            if 'unfold' in commands and c.state != 'FLIGHT':
                c.message = 'Unfold refused: take off and hold above 1 m first.'
            if 'fold' in commands and c.state != 'FLIGHT':
                c.message = 'Fold helper requires FLIGHT; ground pose is already folded.'
            if 'reset' in commands:
                with viewer.lock():
                    c = reset(m, d)
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
                    for _ in range(round(1./(60.*m.opt.timestep))):
                        c.step(d, router.keys)
                        mujoco.mj_step(m, d)
                        mujoco.mj_forward(m, d)
                        fault = c.safety(d)
                        if fault:
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
                    paused = True
                    hard_fault = 'Native state/control edited. Physics paused; Ctrl+F11 restores known state.'
                    router.disarm('Click 3D, release keys, then Ctrl+F11.')
                    gui_counts['native_edit_detected'] += 1
                if m.opt.__repr__() != baseline_options:
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
        REPORT.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-test', action='store_true', help='Timed native window/input test; writes its own JSON')
    parser.add_argument('--duration', type=float, default=180., help='Wall seconds for --window-test only')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    run(args.window_test, args.duration)


if __name__ == '__main__':
    main()
