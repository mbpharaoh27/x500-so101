# X500 + SO-101 — 原生 MuJoCo 单窗口控制

## 应运行的文件

```text
D:\fangzhenmoxing\fangzhenmoxing\PX4\x500_so101_mujoco\keyboard_control_x500_so101_native_viewer.py
```

这是独立启动器，导入现有 `keyboard_control_x500_so101.py` 的控制器；没有覆盖 V07，没有新建或改写 XML。
它读取 `PX4_X500_SO101_keyboard.xml`，并保留全部原有资产、关节、质量、碰撞、重力、执行器上限。
旋翼仍采用“有限轴力矩 → 实际关节转速 → 气动力”，不是按目标转速直接产生推力。

## 中文 PyCharm 设置

1. 顶部运行配置下拉菜单 → 编辑配置 → ＋ → Python。
2. 名称：`X500_SO101_Native_Viewer_Control`。
3. 解释器：`C:\Users\hunter\.conda\envs\mujoco-env\python.exe`。
4. 脚本路径：填写本文最上方的新 `.py` 完整路径。
5. 工作目录：`D:\fangzhenmoxing\fangzhenmoxing\PX4\x500_so101_mujoco`。
6. 脚本形参：留空。应用 → 确定 → 运行。

没有自动修改你的 PyCharm 配置。旧键盘版仍可使用。

## 实际操作顺序

1. 建议最大化原生窗口；可折叠菜单或用 Tab / Shift+Tab 切换左右面板，给三维区域留出空间。
2. 点击中央三维区域，然后松开鼠标及所有控制键。
3. 左上角出现 `CONTROL ACTIVE` 后才能控制。普通鼠标操作仍由原生窗口处理。
4. 按住 Ctrl，再按住 W。电机先用约 2 秒仿真时间加速，再继续起飞。
5. 松开 W 后松开 Ctrl：平滑减速、保持高度。高度是 base_link 原点距地面高度，上限 2 米。
6. 超过 1 米并悬停后，可按一下 Ctrl+U 展开，整个展开约 15.5 秒仿真时间。
7. 展开过程中数字手动控制暂不接管；状态行显示当前路径段。
8. 展开完成后按住 Ctrl+数字或 Alt+数字控制关节。
9. 松开关节按键、等待制动后，按一下 Ctrl+Enter 请求收臂并降落。
10. 真正接地且稳定后卸载并停桨；关闭窗口本身不是自动降落。

仿真时间不一定等于实际时间，机器较慢时要等待更久。Ctrl+W 短按只会开始/取消电机启动，不是自动飞到两米。

## 最终按键表

| 动作 | 按键 | 操作方式 |
|---|---|---|
| 上升 / 下降 | Ctrl+W / Ctrl+S | 按住；松开后平滑保持 |
| 前进 / 后退 | Ctrl+I / Ctrl+K | 按住；相对当前机头方向 |
| 机头左转 / 右转 | Ctrl+E / Ctrl+R | 按住；松开保持朝向 |
| 关节 1 肩部水平转动 | Ctrl+1 / Alt+1 | 正向 / 反向，按住 |
| 关节 2 肩部抬升 | Ctrl+2 / Alt+2 | 正向 / 反向，按住 |
| 关节 3 肘部弯曲 | Ctrl+3 / Alt+3 | 正向 / 反向，按住 |
| 关节 4 腕部俯仰 | Ctrl+4 / Alt+4 | 正向 / 反向，按住 |
| 关节 5 腕部旋转 | Ctrl+5 / Alt+5 | 正向 / 反向，按住 |
| 关节 6 夹爪 | Ctrl+6 / Alt+6 | 张开 / 闭合，按住 |
| 展开 / 收回 | Ctrl+U / Ctrl+F | 按一下 |
| 完整降落 | Ctrl+Enter | 按一下 |
| 暂停 / 继续 | Ctrl+F10 | 按一下；实际暂停物理和控制 |
| 重置 | Ctrl+F11 | 按一下；仿真调试用，恢复已知地面初始状态 |

**没有 Ctrl+J 或 Ctrl+H 平移功能，也没有其他直接侧移按键。**位置闭环仍会稳定两个水平方向。
飞行组合示例：Ctrl+W+I 同时上升和前进。相反方向指令抵消。
使用主键盘数字 1—6；Ctrl 和 Alt 同时按下、带 Shift 或 Windows 修饰键时拒绝运动输入。

## 输入与原生界面如何共存

- 仅一个 `mujoco.viewer.launch_passive` 原生窗口，左右菜单可用；没有第二个控制面板。
- Windows 公开 API（经标准库 ctypes）：EnumWindows、GetWindowThreadProcessId、GetForegroundWindow、GetAsyncKeyState、GetKeyState 等。
- 根据同一进程和唯一 MuJoCo 标题识别公开 Win32 HWND，持续核对所属 PID；不读取私有 GLFW 窗口句柄。
- 独立线程约每 5 ms 采样一次当前按键，避免慢物理/渲染帧漏掉鼠标点击。使用高位状态，不使用“不可靠的最近按过”低位。
- 一次性动作还在公开 key_callback 内使用 UI 消息线程的 GetKeyState 核实修饰键，避免快速松开 Ctrl 导致短按丢失。
- 轮询和回调对同一按键的重复观察使用 0.15 秒去重，仅针对一次性动作，不用超时猜测松键。
- 不使用系统级键盘钩子，不向其他程序注入控制，不安装新依赖，不修改 MuJoCo 安装源码。
- 点击菜单、窗口失焦或无法确认窗口身份：清空运动输入，参考平滑转为保持，自动机械臂路径按原控制器逻辑制动停止。
- 切回来不会自动继续旧指令；需要重新点击三维区域、释放按键后再操作。
- 鼠标拖动/扰动期间不接受运动按键；需要再次点击三维区域启用。

## 原生快捷键及菜单的限制

普通 W/S、U/F、数字、F1、Tab、Esc 等仍保留原生用途；运动控制必须用本文组合键。
MuJoCo 3.15.0 的 UI 快捷键精确匹配修饰键，因此 Ctrl/Alt+数字不等于普通数字/Shift+数字的显示组操作。
没有占用 Ctrl+L（重载）、Ctrl+Q（退出）、Ctrl+A（对齐）等已存在的组合键。

原生 **Pause/Run 在 passive 模式不可用于暂停外部物理循环**，请用 Ctrl+F10；原生单步和速度控制也不驱动本脚本。
原生 Reset、Load key、Joint、Control 对真实状态/控制的修改，会在 sync 后、下一步物理前检测并暂停。
这时先点击三维区域、释放按键，再 Ctrl+F11 恢复已知初始状态，不能直接沿用旧控制器继续。
Control 面板不是并行遥控器；飞行控制器拥有执行器输入。
原生 Physics 参数一旦变化会暂停；需恢复原参数或关闭后重启。Ctrl+F11 只重置状态，不偷偷恢复用户修改的物理参数。
不支持飞行中更换模型；发现 viewer 的模型/数据发生替换会停止控制。
视觉显示可以调整。新启动器只在内存中关闭重型阴影贴图与离屏多重采样以减轻渲染，不改变磁盘模型或碰撞。

## 验证结果与范围

`validate_native_viewer_control.py` 使用相同组合键转换器和原有控制器做无窗口自动测试。
测试包括地面等待、提前取消启动、真实旋翼动力学起飞、至少十秒悬停、前后移动、偏航、无效侧移键、展开、六关节正反向、收臂降落和停桨后五秒承重。

当前测试结果：

- 仿真约 110 秒，最后十秒悬停最大高度误差约 0.3445 mm。
- 最大机身倾斜约 2.1411°，六关节最大跟踪误差约 0.1686°。
- 六个关节分别正反向移动；无关节越限，无数值警告。
- 电机轴力矩、机械臂力矩及分配器饱和次数均为零。
- 最终起落架支撑力约 26.45084 N，与模型重量一致；旋翼实际转速与气动推力接近数值零。
- Ctrl+J/H 无移动映射，释放修饰键、失焦、菜单模式和暂停输入保护通过模拟状态测试。

原生窗口已实际显示并检查左右菜单、CONTROL ACTIVE 点击启用、Ctrl+F10 暂停/恢复、地面 Ctrl+U 明确拒绝、原生 Reset 触发暂停及 Ctrl+F11 恢复。
窗口验证使用 Computer Use 经操作系统发送的组合键，区别于直接调用回调；**不是人工实体键盘长按试飞**。
实体键盘连续长按、所有修饰键释放顺序、完整空中操作和各种显示缩放仍需用户试用确认，不声称已全部人工验证。
Ctrl/Alt+数字的具体实际关节运动通过自动动力学测试验证，原生窗口内的全套关节长按尚未逐项验证。
在测试机器上原生窗口带重型网格的运行可能慢于实时，没有通过改变物理步长或降低碰撞精度掩盖性能问题。

## 新增测试文件

- `validate_native_viewer_control.py`：自动输入和真实动力学测试入口。
- `native_viewer_control_validation.json`：自动测试结果。
- `native_viewer_control_timeseries.csv`：50 Hz 左右的状态、关节参考和实际角记录。
- `native_viewer_window_validation.json`：原生窗口与已观察组合键的记录，不等同于完整飞行验证。

复测自动测试会覆盖以上新版本专属 JSON/CSV，不会写入旧 V07 报告。
启动器的 `--window-test --duration 120` 会打开限时原生窗口并写入窗口报告，日常启动参数留空。

## 官方依据

- https://mujoco.readthedocs.io/en/3.15.0/programming/samples.html#shortcuts
- https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/simulate.cc
- https://github.com/google-deepmind/mujoco/blob/3.15.0/src/ui/ui_main.c#L1112
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getasynckeystate
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getforegroundwindow
