"""Synthetic chord/physics tests; no viewer and no Windows input injection."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import keyboard_control_x500_so101_native_viewer as native

core = native.core
HERE = Path(__file__).resolve().parent


def ctrl(key):
    return {native.VK_CTRL, ord(key)}


def input_tests():
    r = native.Chords()
    r.update(ctrl('W'))
    assert not r.keys.down
    r.click(True)
    r.update(ctrl('W'))
    assert not r.keys.down and r.wait_release
    r.update(set())
    r.update(ctrl('W'))
    assert r.keys.down == {core.glfw.KEY_W}
    r.update({ord('W')})
    assert not r.keys.down
    for key in ('J', 'H'):
        r.update(ctrl(key))
        assert not r.keys.down
    r.update(ctrl('I'))
    assert r.keys.down == {core.glfw.KEY_UP}
    r.update(ctrl('K'))
    assert r.keys.down == {core.glfw.KEY_DOWN}
    r.update({native.VK_CTRL, ord('W'), ord('S'), ord('I'), ord('K'), ord('E'), ord('R')})
    assert r.keys.axis(core.glfw.KEY_W,core.glfw.KEY_S) == 0
    assert r.keys.axis(core.glfw.KEY_UP,core.glfw.KEY_DOWN) == 0
    assert r.keys.axis(core.glfw.KEY_Q,core.glfw.KEY_E) == 0
    for i in range(6):
        r.update({native.VK_CTRL,ord('1')+i})
        assert r.keys.down == {core.glfw.KEY_1+i}
        r.update({native.VK_ALT,ord('1')+i})
        assert r.keys.down == {core.glfw.KEY_1+i,core.glfw.KEY_LEFT_SHIFT}
    r.update({native.VK_CTRL,native.VK_ALT,ord('1')})
    assert not r.keys.down
    r.update(set())
    r.update(ctrl('U'))
    assert core.glfw.KEY_U in r.keys.edges
    r.keys.edges.clear()
    r.update(ctrl('U'))
    assert not r.keys.edges
    r.update(ctrl('W'),foreground=False)
    assert not r.armed and not r.keys.down and not r.keys.focused
    r.update(ctrl('W'))
    assert not r.keys.down
    r.click(True); r.update(set())
    r.update(ctrl('W'),paused=True)
    assert not r.keys.down
    r.click(False)
    assert not r.armed
    return {'foreground_click_release_gate':True,'modifier_release_stops':True,
            'ctrl_J_H_unbound':True,'opposite_commands_cancel':True,
            'all_six_two_key_joint_directions':True,'mixed_modifiers_rejected':True,
            'helper_edge_single_trigger':True,'focus_loss_and_no_automatic_rearm':True,
            'paused_and_menu_mode_no_motion':True}


def main():
    checks = input_tests()
    m,d = core.load()
    core.model_audit(m,d)
    c = core.KeyboardFlight(m,d)
    r = native.Chords(); r.click(True); r.update(set())
    rows=[]
    max_arm = max_tilt = max_limit = max_formula = 0.
    peak_xy = peak_z = peak_yaw = 0.
    peak_torque=np.zeros(6)
    shaft_saturation=arm_saturation=allocation_saturation=0
    next_sample=0.
    phase='idle'
    def tick(down):
        nonlocal next_sample,max_arm,max_tilt,max_limit,max_formula,peak_xy,peak_z,peak_yaw
        nonlocal peak_torque,shaft_saturation,arm_saturation,allocation_saturation
        r.update(down)
        c.step(d,r.keys)
        mujoco.mj_forward(m,d)
        state=core.metrics(m,d,c)
        max_arm=max(max_arm,state['arm_max_error_deg'])
        max_tilt=max(max_tilt,state['tilt_deg'])
        peak_xy=max(peak_xy,state['horizontal_speed_m_s'])
        peak_z=max(peak_z,abs(state['vertical_speed_m_s']))
        peak_yaw=max(peak_yaw,abs(state['actual_yaw_rate_deg_s']))
        limits=m.jnt_range[c.arm_joints]
        max_limit=max(max_limit,float(np.max(limits[:,0]-d.qpos[c.arm_qadr])),
                      float(np.max(d.qpos[c.arm_qadr]-limits[:,1])))
        w=d.qvel[c.rotor_vadr]
        max_formula=max(max_formula,float(np.max(np.abs(d.actuator_force[c.aero_ids]-c.spin*c.kf*w*np.abs(w)))))
        torque=np.abs(d.actuator_force[c.arm_ids])
        peak_torque=np.maximum(peak_torque,torque)
        arm_saturation+=int(np.any(torque>=m.actuator_forcerange[c.arm_ids,1]-1e-8))
        shaft_saturation+=int(np.any(np.abs(d.actuator_force[c.shaft_ids])>=c.torque_cap-1e-8))
        allocation_saturation+=int(c.saturated)
        if d.time>=next_sample:
            next_sample=float(d.time)+.02
            row=dict(state,phase=phase)
            for i in range(6):
                row[f'joint_{i+1}_target_deg']=float(np.rad2deg(c.arm_target[i]))
                row[f'joint_{i+1}_actual_deg']=float(np.rad2deg(d.qpos[c.arm_qadr[i]]))
            rows.append(row)
        mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
        assert c.safety(d) is None,(phase,c.safety(d),float(d.time))
    def hold(seconds,down=set(),label=None):
        nonlocal phase
        if label:
            phase=label
            print(f'native-test {phase} t={d.time:.2f} z={d.xpos[c.base,2]:.3f}',flush=True)
        for _ in range(round(seconds/m.opt.timestep)):
            tick(down)
    def until(predicate,timeout,down=set(),label=None):
        nonlocal phase
        if label:
            phase=label
            print(f'native-test {phase} t={d.time:.2f}',flush=True)
        start=float(d.time)
        while not predicate():
            assert d.time-start<timeout,(phase,'timeout',c.message)
            tick(down)
    hold(1.,label='ground_idle')
    assert c.state=='GROUND' and np.max(np.abs(c.force))<1e-8
    hold(.6,ctrl('W'),'cancel_start')
    hold(2.,label='cancel_wait')
    assert c.state=='GROUND'
    until(lambda:d.xpos[c.base,2]-c.ground_z>1.3,20.,ctrl('W'),'rise')
    hold(12.,label='hover')
    hover=[x for x in rows if x['phase']=='hover'][-500:]
    assert max(abs(x['height_error_m']) for x in hover)<.03
    assert max(x['horizontal_error_m'] for x in hover)<.03
    for key,label in [('I','forward'),('K','backward'),('E','yaw_left'),('R','yaw_right')]:
        hold(1.5,ctrl(key),label)
        hold(2.,label=label+'_hold')
    before=c.p_ref.copy()
    hold(1.,ctrl('J'),'unbound_J'); hold(1.,ctrl('H'),'unbound_H')
    assert np.linalg.norm(c.p_ref[:2]-before[:2])<1e-5
    hold(.04,ctrl('U'),'unfold_request')
    assert c.path,c.message
    until(lambda:not c.path,25.,label='unfold')
    assert np.max(np.abs(c.arm_target-core.DOWN))<1e-5
    joints=[]
    for i in range(6):
        before=c.arm_target.copy()
        hold(.7 if i<5 else 2.,{native.VK_CTRL,ord('1')+i},f'joint_{i+1}_plus')
        hold(.5,label=f'joint_{i+1}_hold')
        plus=c.arm_target.copy()
        hold(.7 if i<5 else 2.,{native.VK_ALT,ord('1')+i},f'joint_{i+1}_minus')
        hold(.5,label=f'joint_{i+1}_minus_hold')
        assert plus[i]-before[i]>.01
        assert plus[i]-c.arm_target[i]>.01
        joints.append({'joint':core.ARM_NAMES[i],'plus_deg':float(np.rad2deg(plus[i])),
                       'minus_deg':float(np.rad2deg(c.arm_target[i]))})
    # Freeze boundary test: no physics/control steps while paused, as in GUI loop.
    saved=native.state_copy(d)
    for _ in range(50):r.update(ctrl('W'),paused=True)
    assert native.same_state(saved,d)
    r.disarm('synthetic focus loss')
    hold(2.,label='focus_loss_hold')
    r.click(True);r.update(set())
    hold(.04,{native.VK_CTRL,native.VK_ENTER},'land_request')
    until(lambda:c.state=='GROUND',45.,label='land_and_stop')
    hold(5.,label='landed_support')
    assert c.ground_force>.95*c.mass*c.g
    assert np.max(np.abs(c.force))<1e-8
    warnings=[int(w.number) for w in d.warning]
    assert not any(warnings)
    assert not arm_saturation and not shaft_saturation and not allocation_saturation
    assert max_tilt<10. and max_arm<1. and max_limit<1e-6 and max_formula<1e-10
    result={'passed':True,'scope':'Synthetic state samples and real MuJoCo dynamics, not native window/human keyboard testing.',
            'input_checks':checks,'simulation_s':float(d.time),'hover_last_10s_max_height_error_m':max(abs(x['height_error_m']) for x in hover),
            'hover_last_10s_max_xy_error_m':max(x['horizontal_error_m'] for x in hover),
            'max_tilt_deg':max_tilt,'max_joint_error_deg':max_arm,'max_joint_limit_excess_rad':max_limit,
            'peak_xy_speed_m_s':peak_xy,'peak_z_speed_m_s':peak_z,'peak_yaw_rate_deg_s':peak_yaw,
            'max_actual_speed_thrust_formula_error_N':max_formula,'peak_arm_torque_Nm':peak_torque.tolist(),
            'arm_saturation_steps':arm_saturation,'shaft_saturation_steps':shaft_saturation,
            'allocation_saturation_steps':allocation_saturation,'joint_results':joints,
            'final_state':c.state,'final_support_N':c.ground_force,'weight_N':c.mass*c.g,
            'final_actual_rotor_speed_rad_s':d.qvel[c.rotor_vadr].tolist(),'final_thrust_N':c.force.tolist(),
            'warnings':warnings,'source_sha256':hashlib.sha256((HERE/'keyboard_control_x500_so101.py').read_bytes()).hexdigest()}
    with (HERE/'native_viewer_control_timeseries.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (HERE/'native_viewer_control_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
