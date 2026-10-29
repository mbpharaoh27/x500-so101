"""Input and protection tests; prediction poses are NOT flight demonstrations."""
from pathlib import Path
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import mujoco
import numpy as np
import keyboard_control_x500_so101 as k

HERE = Path(__file__).resolve().parent


def main():
    results = {}
    keys = k.Keys()
    keys.key(None, k.glfw.KEY_1, 0, k.glfw.PRESS, 0)
    keys.key(None, k.glfw.KEY_LEFT_SHIFT, 0, k.glfw.PRESS, 0)
    assert k.glfw.KEY_1 in keys.down and keys.shift
    before = (keys.down.copy(), keys.edges.copy())
    keys.key(None, k.glfw.KEY_1, 0, k.glfw.REPEAT, 0)
    assert before == (keys.down, keys.edges)
    keys.key(None, k.glfw.KEY_1, 0, k.glfw.RELEASE, 0)
    assert k.glfw.KEY_1 not in keys.down
    keys.key(None, k.glfw.KEY_W, 0, k.glfw.PRESS, 0)
    keys.key(None, k.glfw.KEY_S, 0, k.glfw.PRESS, 0)
    assert keys.axis(k.glfw.KEY_W, k.glfw.KEY_S) == 0.
    keys.key(None, k.glfw.KEY_SPACE, 0, k.glfw.PRESS, 0)
    assert keys.paused and not keys.down
    keys.key(None, k.glfw.KEY_UP, 0, k.glfw.PRESS, 0)
    assert not keys.down
    keys.key(None, k.glfw.KEY_SPACE, 0, k.glfw.PRESS, 0)
    assert not keys.paused and not keys.down
    keys.key(None, k.glfw.KEY_R, 0, k.glfw.PRESS, 0)
    assert keys.reset and not keys.down
    keys.focus(None, False)
    keys.key(None, k.glfw.KEY_W, 0, k.glfw.PRESS, 0)
    assert not keys.down
    keys.focus(None, True)
    results['press_release_shift_repeat_opposites_pause_reset_focus'] = True

    m, d = k.load()
    c = k.KeyboardFlight(m, d)
    q, v = d.qpos.copy(), d.qvel.copy()
    c.target_speed = c.spin*500.
    c.apply_inner(d)
    assert np.array_equal(d.qpos, q) and np.array_equal(d.qvel, v)
    assert np.array_equal(c.force, np.zeros(4))
    assert np.max(np.abs(d.ctrl[c.shaft_ids])) <= c.torque_cap
    results['nonzero_target_speed_but_actual_rest_has_exact_zero_thrust'] = {
        'actual_speed_rad_s': d.qvel[c.rotor_vadr].tolist(), 'force_N': c.force.tolist()}

    m, d = k.load(); c = k.KeyboardFlight(m, d); keys = k.Keys()
    keys.key(None, k.glfw.KEY_2, 0, k.glfw.PRESS, 0)
    for _ in range(round(.5/m.opt.timestep)):
        c.step(d, keys); mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
    assert np.allclose(c.arm_target, k.FOLDED, rtol=0., atol=1e-12) and c.state == 'GROUND'
    results['ground_joint_motion_rejected_without_locking_joint'] = True
    keys.clear()
    assert c.request_fold(d) and not c.path
    results['already_folded_F_does_not_move_arm'] = True

    # Offline geometry poses: one planning MjData at 1.5m, no physics advance.
    p = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, p, m.key('keyboard_ready').id)
    p.qpos[2] = 1.5
    mujoco.mj_forward(m, p)
    qbefore, vbefore = p.qpos.copy(), p.qvel.copy()
    assert c.guard.check(p, k.FOLDED)[0]
    assert c.guard.check(p, k.DOWN)[0]
    collision = None
    for s in np.linspace(.05, .95, 19):
        candidate = k.FOLDED+s*(k.DOWN-k.FOLDED)
        ok, reason = c.guard.check(p, candidate)
        if not ok and 'clearance' in reason:
            collision = {'fraction': float(s), 'reason': reason}
            break
    assert collision is not None, 'Direct unsafe unfold must be rejected'
    results['unsafe_direct_unfold_rejected'] = collision
    illegal = k.DOWN.copy(); illegal[0] = np.deg2rad(120.)
    assert not c.guard.check(p, illegal)[0]
    results['joint_limit_and_braking_margin_rejection'] = True
    ok, reason = c.guard.check(p, k.DOWN, torque_fraction=.00001)
    assert not ok and 'torque' in reason
    results['gravity_torque_reserve_predicate'] = {
        'passed': True, 'scope': 'Injected conservative reserve threshold; no hardware/physical overload claimed.'}
    assert np.array_equal(p.qpos, qbefore) and np.array_equal(p.qvel, vbefore)
    results['guard_does_not_overwrite_supplied_state'] = True

    saved = c.guard.check
    c.guard.check = lambda *args, **kwargs: (False, 'injected unsafe path')
    c.arm_target = k.DOWN.copy()
    assert not c.request_fold(p, landing=True) and not c.land_after_fold and not c.path
    c.guard.check = saved
    results['landing_refuses_unverifiable_fold_path'] = True

    # Real gravity/free-base dynamics for helper focus-loss braking.
    m, d = k.load(); c = k.KeyboardFlight(m, d); keys = k.Keys()
    phase, begin, stop_time = 'idle', 0., None
    for step in range(round(22./m.opt.timestep)):
        if phase == 'idle' and d.time > .25:
            keys.key(None, k.glfw.KEY_W, 0, k.glfw.PRESS, 0); phase = 'rise'
        elif phase == 'rise' and d.xpos[c.base, 2] > 1.2:
            keys.clear(); phase, begin = 'hold', float(d.time)
        elif phase == 'hold' and d.time-begin > 2.:
            keys.key(None, k.glfw.KEY_U, 0, k.glfw.PRESS, 0); phase, begin = 'helper', float(d.time)
        elif phase == 'helper' and d.time-begin > 1.:
            keys.focus(None, False); phase, begin = 'stop', float(d.time)
        elif phase == 'stop' and d.time-begin > 1.5 and not c.path:
            stop_time = float(d.time); break
        c.step(d, keys); mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
        assert c.safety(d) is None, c.safety(d)
    assert stop_time is not None and not c.path and np.max(np.abs(c.arm_velocity)) < 1e-8
    results['focus_loss_brakes_helper_on_verified_path'] = {'passed': True, 'time_s': stop_time}

    report = json.loads((HERE/'keyboard_control_validation.json').read_text())
    assert report['passed']
    rows = list(csv.DictReader((HERE/'keyboard_control_timeseries.csv').open()))
    directions = {}
    for name, axis, sign in [('forward', 0, 1), ('backward', 0, -1), ('left', 1, 1), ('right', 1, -1)]:
        phase = report['phases'][name]
        displacement = np.array(phase['last_position_m'])-phase['first_position_m']
        assert sign*displacement[axis] > .03, (name, displacement)
        directions[name] = displacement.tolist()
    section = [row for row in rows if row['phase'] == 'forward_after_yaw']
    delta = np.array([float(section[-1][axis])-float(section[0][axis]) for axis in ('base_x_m','base_y_m')])
    yaw = np.deg2rad(float(section[0]['yaw_deg']))
    forward = np.array([np.cos(yaw),np.sin(yaw)])
    assert np.dot(delta,forward) > .1 and delta[1] > .03
    directions['forward_after_yaw'] = {'world_delta_m':delta.tolist(), 'starting_yaw_deg':float(section[0]['yaw_deg'])}
    results['body_frame_direction_checks'] = directions
    t = np.array([float(row['time_s']) for row in rows])
    q = np.deg2rad([[float(row[name+'_target_deg']) for name in k.ARM_NAMES] for row in rows])
    # CSV is 100Hz, controller 250Hz: a 10ms difference alternates between
    # 8ms/12ms of zero-order-held target updates. 20ms spans exactly 5 updates.
    speed = (q[2:]-q[:-2])/(t[2:]-t[:-2])[:,None]
    results['sampled_peak_reference_joint_speed_deg_s'] = np.rad2deg(np.max(np.abs(speed),axis=0)).tolist()
    results['joint_speed_estimation_window_s'] = .02
    assert np.max(np.abs(speed)) < np.deg2rad(30.1)
    assert all(item['plus_target_deg']-item['minus_target_deg'] > 1. for item in report['joints'])
    results['all_six_joint_positive_and_negative_motions_verified'] = True
    result = {'passed': True, 'tests': results,
              'launcher_sha256': hashlib.sha256((HERE/'keyboard_control_x500_so101.py').read_bytes()).hexdigest(),
              'scope': 'Synthetic input and physics/protection tests. Not human keyboard hardware testing.'}
    (HERE/'keyboard_control_details_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
