# Third-party notices

This file is a release template for the EmbodiSteer research repository. It
must be kept with any distribution of the corresponding third-party code or
assets.

## Diffusion Policy and UMI components

The `diffusion_policy/` and `umi/` components are distributed under the MIT
License. The complete license text is in [`LICENSE`](LICENSE). Preserve the
upstream attribution and acknowledgements when redistributing these files.

## ManiSkill fork

The ManiSkill source code is Apache-2.0. ManiSkill's README states that its
assets are licensed under CC BY-NC 4.0. Code and assets must therefore be
listed separately in any release, with attribution and the non-commercial
restriction preserved. See the dependency fork's `LICENSE` and
`LICENSE-3RD-PARTY` files.

## cuRobo fork

cuRobo is distributed under the NVIDIA License. The license requires retaining
the license and attribution notices and limits the Work and derivative works
to non-commercial research or evaluation. External URDF and mesh terms are
listed in cuRobo's `LICENSE_ASSETS` and must not be replaced by the project
license.

## Robotiq Modbus driver

The serial gripper backend depends on
[`frankWang67/robotiq_modbus_gripper`](https://github.com/frankWang67/robotiq_modbus_gripper)
at commit `582a6c26a2462adb58c60ba229ad9578556cf464`. Its package metadata
declares MIT, but that revision does not include a standalone license file.
The source is installed as an external pip dependency, not bundled here;
confirm the complete upstream license notice before redistributing its source.

## Project-specific asset ledger

Three benchmark asset groups are included in the repository. Their complete
paths, source revisions, license evidence and review status are recorded in
[`third_party/assets.yaml`](third_party/assets.yaml):

- Block Pushing URDF/OBJ assets are included in the repository. Adjacent
  Reach ML source files carry Apache-2.0 headers, but an asset-level upstream
  license confirmation is still required before public redistribution.
- D'Suite Kitchen scenes, meshes and textures include their Apache-2.0 license
  and ROBEL attribution in the distributed files.
- Franka MuJoCo XML, meshes and texture include an Apache-2.0 license and
  upstream source notices in the distributed files.

ManiSkill and cuRobo assets are materialized from their fixed forks at install
time and are not copied into this tree. Local calibration recordings are not
included. Any future mesh, URDF, texture, generated asset or calibration file
must be added to the ledger before publication.

“Open source” is not by itself a redistribution grant. If an asset cannot be
redistributed under the release terms, the public repository must reference a
download/setup step instead of bundling it.
