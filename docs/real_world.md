# Real-world deployment

The physical and simulation paths share the same policy classes and method
router. The paper profile uses `DiffusionUnetTimmPolicyEmbodiSteer`.
The supported entry point is:

```console
python eval_real.py \
  --input /private/checkpoints/experiment \
  --output /private/episodes \
  --robot_config /private/configs/robot.yaml \
  --obstacle_config /private/configs/obstacles.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

The policy YAML selects Cartesian or joint-space inference, guidance, CBF/SDF
settings. The paper profile selects joint-space CBF
guidance. At runtime, compatible checkpoint Hydra targets are selected by
`runtime_config.policy_target`: joint CBF uses the paper class, while joint
GD/no-guidance uses `ee2joint.DiffusionUnetTimmPolicyJointSpace`.

## Configuration boundary

Start from [`configs/real/robot.template.yaml`](../configs/real/robot.template.yaml)
and keep the populated copy outside version control. It must contain one
robot/gripper pair per entry, a 4x4 `tx_left_right` transform, and explicit
addresses. Obstacle geometry is selected separately with
[`configs/real/obstacles/door_frame.yaml`](../configs/real/obstacles/door_frame.yaml)
or a private YAML in the robot base frame.

### Teleoperation input device

Set the top-level `input_device` in the robot YAML passed to `--robot_config`:

```yaml
input_device: keyboard  # Default; change to spacemouse to use a physical SpaceMouse.
```

Omitting this field keeps keyboard control for existing configurations. The
keyboard backend uses WASD/RF for translation, IJKL/UO for rotation and Z/X
for gripper close/open. The SpaceMouse backend uses its six motion axes and
two buttons for the same operations. Episode/policy shortcuts still use the
keyboard in both modes, so `pynput` and a graphical session remain required.

The terminal prints input-specific instructions each time human control begins,
and the stop/return shortcut when policy execution begins. The instructions also
show applicable dataset-matching and multi-robot shortcuts. In both input modes,
P selects the previous matched episode when `--match_episode` is set (E selects
the next), and 0 selects all robots; 1/2 select robot 1/2. These replace the old
W/A shortcuts, leaving WASD/RF, IJKL/UO and Z/X dedicated to keyboard motion and
gripper control. Use lowercase letters, without Shift. S retains its phase-specific
roles: stopping an episode during policy execution and commanding -X motion in
human control; release it promptly when returning. Software shortcuts do not
replace a hardware emergency stop.

Keyboard control needs no SpaceMouse packages. For a physical SpaceMouse,
install the optional pinned backend and native packages on Ubuntu:

```console
sudo apt-get install libspnav-dev spacenavd
python -m pip install -r environment/spacemouse-requirements.txt
sudo systemctl start spacenavd
```

Connect the input device and verify access to the daemon in the lab's bring-up
procedure. These setup commands start only the input-device daemon, not a robot
controller. Missing `spnav`/`libspnav` produces an explicit error when selecting
the physical backend; daemon startup failure is bounded by a timeout.
Dry-run prints the selected input device but never starts either input process.
Offline preflight requires the `spnav` package only when `spacemouse` is selected;
package discovery does not verify native-library loading, daemon or device readiness.

### Gripper selection and connection parameters

`gripper_type` supports `robotiq` (the default when omitted) and `wsg50`.
Choose one entry per robot, in matching order. Both types use
`gripper_obs_latency` and `gripper_action_latency`, in seconds; calibrate these
for the workstation rather than treating the example values as measurements.

For the serial Robotiq 2F-85 backend:

```yaml
grippers:
  - gripper_type: robotiq
    gripper_serial_port: /dev/ttyUSB0
    gripper_slave_id: 9
    gripper_obs_latency: 0.01
    gripper_action_latency: 0.1
```

`gripper_serial_port` is required in the YAML. Use the actual device path,
preferably a stable `/dev/serial/by-id/...` path, and ensure the process has
serial-device permissions. `gripper_slave_id` defaults to `9`. No
`gripper_ip` or TCP `gripper_port` is needed. For multiple serial grippers,
set each entry's device path independently. The pinned driver fixes the line
settings to 115200 baud, 8 data bits, no parity, and 1 stop bit; these are not
configurable here. The controller retains its 0.085 m width calibration.

For WSG50, replace the Robotiq entry with:

```yaml
grippers:
  - gripper_type: wsg50
    gripper_ip: YOUR_GRIPPER_IP
    gripper_port: 1000
    gripper_obs_latency: 0.01
    gripper_action_latency: 0.1
```

`gripper_ip` is the required hostname or IP address. `gripper_port` is a TCP
port and defaults to `1000`. No serial parameters are needed. The controller
uses `use_meters=True`, keeping the environment's width observations and
commands in meters for both backends.

`BimanualUmiEnv` reads these entries directly. The single-arm `UmiEnv` API
accepts the same connection fields as keyword arguments; its Robotiq defaults
are `gripper_serial_port='/dev/ttyUSB0'` and `gripper_slave_id=9`, and
`gripper_ip` is optional for Robotiq.

This selection configures the hardware controller, not the robot geometry.
The current joint-space real-world path still selects Robotiq-specific cuRobo
models and finger-joint conversion. WSG50 joint-space/whole-body collision
guidance requires matching kinematics and collision assets; selecting
`gripper_type: wsg50` alone does not provide that support.

### Gripper dependencies

The real profile (`environment/environment-real.yaml`) includes the
[`robotiq_modbus_gripper` driver](https://github.com/frankWang67/robotiq_modbus_gripper),
pinned to `582a6c26a2462adb58c60ba229ad9578556cf464`, plus `pymodbus==3.8.6`
and `pyserial==3.5`. The distribution/import name is `robotiq_gripper`.
The simulation profile does not install these packages. The real profile
installs only cuRobo from the fork manifest; it does not require ManiSkill or
SAPIEN. See [installation](installation.md) for separate environment commands.
For an existing environment, install just these additions with:

```console
python -m pip install \
  "robotiq_gripper @ https://github.com/frankWang67/robotiq_modbus_gripper/archive/582a6c26a2462adb58c60ba229ad9578556cf464.tar.gz" \
  "pymodbus==3.8.6" "pyserial==3.5"
```

WSG50 uses the in-tree TCP driver and needs no Robotiq/Modbus package. The
selected backend is imported only when constructing its controller. In
particular, constructing `RobotiqController` activates the gripper; do not
instantiate it just to check installation.

### Configuration-only check

Validate before connecting to hardware:

```console
python eval_real.py --dry_run \
  --robot_config /private/configs/robot.yaml \
  --obstacle_config /private/configs/obstacles.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

The dry run assumes the real runtime profile is installed and imports the
launcher's runtime modules normally. It does not require ManiSkill/SAPIEN,
load a checkpoint, construct controllers, connect to devices, move a robot
or start a camera. Template addresses are allowed only in this mode.
It checks finite numeric fields, latency/port ranges, transforms, obstacle
geometry and supported policy/robot/gripper combinations. Successful imports
are not a complete dependency, CUDA or device-readiness check; use the separate
preflight for selected-package checks. A physical run still requires an emergency-stop
operator, controller/camera checks, collision-scene review and a low-speed
bring-up. The release code cannot establish those safety guarantees for a
particular lab setup.

### Real-only preflight

Use a populated site configuration for the separate offline check:

```console
python scripts/preflight_real.py \
  --robot_config /private/configs/robot.yaml \
  --obstacle_config /private/configs/obstacles.yaml \
  --policy-config configs/policy/embodisteer.yaml
```

It reports `PASS`, `FAIL`, `WARN` and `SKIP` and returns a nonzero status for
failures. It rejects placeholders and discovers package metadata/module specs
for the selected robot, gripper and inference space. It never imports a
controller or instantiates `RobotiqController`. `--record-realsense` requires
the optional SDK but does not enumerate or start RealSense devices.

cuRobo and `pytorch-kinematics` are checked in both EE and joint modes: the
public policy package currently imports its joint-policy modules even when
selecting an EE policy. Both dependencies are supplied by the real installation.

Add `--check-devices` to check serial/UVC character-device paths and read/write
permissions, plus local `lsusb`/display prerequisites. It uses filesystem
metadata only: no serial opening, camera capture, USB reset or permission
changes. `--connect` (alias `--check-network`) explicitly permits TCP checks:
UR RTDE port 30004, Franka RPC port 4242, and the configured WSG50 port. These
connections send no application commands; Robotiq serial I/O/activation is
never performed. `--timeout 2` controls each socket timeout (DNS resolution is
OS-managed). Network checks are skipped if earlier offline checks fail.

Package presence is not an import/ABI/CUDA test. A TCP connection is not a
protocol/readiness test. This command does not certify model assets,
calibration, camera streaming, GPU execution, activation or safe operation;
those remain part of the lab's controlled bring-up. Configuration dry-runs
and simulated hardware tests cannot replace that process.
