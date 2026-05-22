"""
RUKA MJCF to URDF converter (v2 - clean frame handling).

Strategy:
  - All MJCF body local poses are translated to URDF joint <origin>
    directly. No per-body frame rotation is applied. Mesh, inertia,
    joint axis, joint origin all stay in MJCF body-local frame.
  - For Z-up (Isaac convention), add a synthetic 'base_link' as the
    URDF root, connected to Palm_Link via a fixed joint whose origin
    rpy contains the Y-up -> Z-up rotation. Whole hand rotates as a
    rigid block at the root; no body-local frame is touched.
  - For X-mirror (left/right hand), if MIRROR_X is True we flip the X
    coordinate of every position AND reflect every quaternion. Inertia
    tensor is reflected accordingly. Joint axes are X-flipped. Joint
    limits are negated and swapped.
  - The dummy MJCF Palm_Joint (slide, near-zero range) is dropped.

Output URDF tree:
    base_link (root, world-fixed)
      -- fixed joint (rpy = Rx(180 deg)) --> Palm_Link
                                              |-- Index_MCP_Joint -> ...
                                              |-- Middle_MCP_Joint -> ...
                                              |-- Ring_MCP_Joint -> ...
                                              |-- Pinky_MCP_Joint -> ...
                                              |-- Thumb_CMC_Joint -> ...
"""

import xml.etree.ElementTree as ET
import numpy as np

# ---------- Configuration ----------
MJCF_PATH = "/home/claude/ruka/hand_assembly.xml"
URDF_PATH = "/home/claude/ruka/ruka_left.urdf"
MESH_PREFIX = ""           # appended to mesh file path
MESH_REWRITE = ("meshes/", "meshes_left/")  # (from, to) substring substitution
MIRROR_X = True            # MJCF is actually right-hand; mirror to get left hand
ROBOT_NAME = "ruka_left_hand"

# Y-up -> Z-up rotation at base_link fixed joint.
# Rx(180 deg): MJCF (X, Y, Z) -> (X, -Y, -Z)
# MJCF -Z (finger direction) -> URDF +Z (pointing up)
# MJCF +Y (palm-up direction) -> URDF -Y
BASE_RPY = (np.pi, 0.0, 0.0)

# ---------- Math ----------
def quat_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])

def R_to_rpy(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy >= 1e-6:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw])

def parse_vec(s, n):
    if s is None:
        return None
    parts = [float(t) for t in s.split()]
    assert len(parts) == n, f"Expected {n}, got {len(parts)}: {s}"
    return np.array(parts)

def parse_pos(s):
    return np.zeros(3) if s is None else parse_vec(s, 3)

def parse_quat(s):
    return np.array([1.0, 0.0, 0.0, 0.0]) if s is None else parse_vec(s, 4)

M_MIRROR = np.diag([-1.0, 1.0, 1.0])

def mirror_pos(p):  return M_MIRROR @ p
def mirror_R(R):    return M_MIRROR @ R @ M_MIRROR
def mirror_axis(a): return M_MIRROR @ a

def build_inertia(diag, q):
    R = quat_to_R(q)
    return R @ np.diag(diag) @ R.T

# ---------- MJCF parsing ----------
def find_palm(worldbody):
    for b in worldbody.iter("body"):
        if b.get("name") == "Palm_Link":
            return b
    raise RuntimeError("Palm_Link not found")

def gather_meshes(asset):
    return {m.get("name"): m.get("file") for m in asset.findall("mesh")}

# ---------- URDF emission ----------
def add_origin(parent, xyz, rpy):
    o = ET.SubElement(parent, "origin")
    o.set("xyz", f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}")
    o.set("rpy", f"{rpy[0]:.9g} {rpy[1]:.9g} {rpy[2]:.9g}")

def add_inertial(link, inertial_mjcf, mirror):
    pos = parse_pos(inertial_mjcf.get("pos"))
    quat = parse_quat(inertial_mjcf.get("quat"))
    diag = parse_vec(inertial_mjcf.get("diaginertia"), 3)
    mass = float(inertial_mjcf.get("mass"))

    I = build_inertia(diag, quat)
    if mirror:
        pos = mirror_pos(pos)
        I = M_MIRROR @ I @ M_MIRROR

    el = ET.SubElement(link, "inertial")
    add_origin(el, pos, np.zeros(3))
    ET.SubElement(el, "mass").set("value", f"{mass:.9g}")
    inert = ET.SubElement(el, "inertia")
    inert.set("ixx", f"{I[0,0]:.9g}")
    inert.set("ixy", f"{I[0,1]:.9g}")
    inert.set("ixz", f"{I[0,2]:.9g}")
    inert.set("iyy", f"{I[1,1]:.9g}")
    inert.set("iyz", f"{I[1,2]:.9g}")
    inert.set("izz", f"{I[2,2]:.9g}")

def add_default_inertial(link, mass=0.05, moi=1e-4):
    el = ET.SubElement(link, "inertial")
    add_origin(el, np.zeros(3), np.zeros(3))
    ET.SubElement(el, "mass").set("value", str(mass))
    inert = ET.SubElement(el, "inertia")
    inert.set("ixx", str(moi)); inert.set("ixy", "0"); inert.set("ixz", "0")
    inert.set("iyy", str(moi)); inert.set("iyz", "0"); inert.set("izz", str(moi))

def geom_local_pose(geom, mirror):
    pos = parse_pos(geom.get("pos"))
    quat = parse_quat(geom.get("quat"))
    R = quat_to_R(quat)
    if mirror:
        pos = mirror_pos(pos)
        R = mirror_R(R)
    return pos, R_to_rpy(R)

def add_visual_mesh(link, mesh_file, pos, rpy, name):
    v = ET.SubElement(link, "visual")
    if name: v.set("name", name)
    add_origin(v, pos, rpy)
    geom = ET.SubElement(v, "geometry")
    ET.SubElement(geom, "mesh").set("filename", mesh_file)
    mat = ET.SubElement(v, "material")
    mat.set("name", "ruka_grey")
    ET.SubElement(mat, "color").set("rgba", "0.439 0.475 0.502 1")

def add_visual_sphere(link, radius, pos, rpy, name):
    v = ET.SubElement(link, "visual")
    if name: v.set("name", name)
    add_origin(v, pos, rpy)
    geom = ET.SubElement(v, "geometry")
    ET.SubElement(geom, "sphere").set("radius", f"{radius:.9g}")

def add_collision_box(link, half_size, pos, rpy, name):
    c = ET.SubElement(link, "collision")
    if name: c.set("name", name)
    add_origin(c, pos, rpy)
    geom = ET.SubElement(c, "geometry")
    full = 2.0 * np.asarray(half_size)
    ET.SubElement(geom, "box").set("size",
        f"{full[0]:.9g} {full[1]:.9g} {full[2]:.9g}")

# ---------- Tree walk ----------
def emit(urdf, parent_name, body, meshes, mirror):
    name = body.get("name")
    body_pos = parse_pos(body.get("pos"))
    body_quat = parse_quat(body.get("quat"))

    # Joint from parent to this body
    if parent_name is not None:
        joints_in_body = body.findall("joint")
        bp = body_pos.copy()
        bR = quat_to_R(body_quat)
        if mirror:
            bp = mirror_pos(bp)
            bR = mirror_R(bR)
        rpy = R_to_rpy(bR)

        if not joints_in_body:
            jname, jtype, jaxis, jrange, axis_flipped = f"{name}_fixed", "fixed", None, None, False
        else:
            j = joints_in_body[0]
            jname = j.get("name")
            jtype_mjcf = j.get("type", "hinge")
            jtype = "revolute" if jtype_mjcf in ("hinge", None) else "prismatic"
            jaxis_orig = parse_vec(j.get("axis", "0 0 1"), 3)
            if mirror:
                # X-mirror flips frame handedness. To keep joint angles
                # producing the same physical motion (finger flexes toward
                # palm), we EITHER negate the axis OR swap-negate the limits.
                # We choose to keep the axis as-is and swap-negate the limit;
                # this leaves the URDF readable (axis (0, 1, 0) preserved)
                # while reproducing the same flexion direction.
                jaxis = jaxis_orig
                axis_flipped = True   # forces limit swap-negate below
            else:
                jaxis = jaxis_orig
                axis_flipped = False
            r_str = j.get("range")
            jrange = parse_vec(r_str, 2) if r_str else None

        je = ET.SubElement(urdf, "joint")
        je.set("name", jname)
        je.set("type", jtype)
        ET.SubElement(je, "parent").set("link", parent_name)
        ET.SubElement(je, "child").set("link", name)
        add_origin(je, bp, rpy)
        if jaxis is not None:
            ET.SubElement(je, "axis").set("xyz",
                f"{jaxis[0]:.9g} {jaxis[1]:.9g} {jaxis[2]:.9g}")
        if jrange is not None:
            lim = ET.SubElement(je, "limit")
            if axis_flipped:
                lo, hi = -jrange[1], -jrange[0]
            else:
                lo, hi = jrange[0], jrange[1]
            lim.set("lower", f"{lo:.9g}")
            lim.set("upper", f"{hi:.9g}")
            lim.set("effort", "10")
            lim.set("velocity", "10")
        dyn = ET.SubElement(je, "dynamics")
        dyn.set("damping", "0.05")
        dyn.set("friction", "0.01")

    # Link
    link = ET.SubElement(urdf, "link")
    link.set("name", name)

    inertial_mjcf = body.find("inertial")
    if inertial_mjcf is not None:
        add_inertial(link, inertial_mjcf, mirror=mirror)
    else:
        add_default_inertial(link)

    for geom in body.findall("geom"):
        gtype = geom.get("type", "sphere")
        gclass = geom.get("class", "")
        gname = geom.get("name")
        pos, rpy = geom_local_pose(geom, mirror=mirror)

        if gclass == "visual":
            if gtype == "mesh":
                mesh_name = geom.get("mesh")
                mesh_file = MESH_PREFIX + meshes[mesh_name]
                if MESH_REWRITE is not None:
                    mesh_file = mesh_file.replace(*MESH_REWRITE)
                add_visual_mesh(link, mesh_file, pos, rpy, name=gname)
            elif gtype == "sphere":
                radius = parse_vec(geom.get("size"), 1)[0]
                add_visual_sphere(link, radius, pos, rpy, name=gname)
        elif gclass == "collision":
            if gtype == "box":
                size = parse_vec(geom.get("size"), 3)
                add_collision_box(link, size, pos, rpy, name=gname)

    for child_body in body.findall("body"):
        emit(urdf, name, child_body, meshes, mirror)

def convert():
    tree = ET.parse(MJCF_PATH)
    mjcf = tree.getroot()
    meshes = gather_meshes(mjcf.find("asset"))
    palm = find_palm(mjcf.find("worldbody"))

    urdf = ET.Element("robot")
    urdf.set("name", ROBOT_NAME)

    mat = ET.SubElement(urdf, "material")
    mat.set("name", "ruka_grey")
    ET.SubElement(mat, "color").set("rgba", "0.439 0.475 0.502 1")

    # Synthetic base_link as URDF root, fixed-jointed to Palm_Link with
    # the Y-up -> Z-up rotation in the joint origin rpy.
    base = ET.SubElement(urdf, "link")
    base.set("name", "base_link")
    add_default_inertial(base, mass=0.001, moi=1e-6)

    bj = ET.SubElement(urdf, "joint")
    bj.set("name", "base_to_palm")
    bj.set("type", "fixed")
    ET.SubElement(bj, "parent").set("link", "base_link")
    ET.SubElement(bj, "child").set("link", "Palm_Link")
    add_origin(bj, np.zeros(3), np.array(BASE_RPY))

    # All bodies emitted with no frame rotation -- pristine MJCF transforms.
    emit(urdf, None, palm, meshes, mirror=MIRROR_X)

    revjoints = [j for j in urdf.findall("joint") if j.get("type") == "revolute"]
    print(f"Total revolute joints: {len(revjoints)}")
    for j in revjoints:
        print(f"  - {j.get('name')}")

    ET.indent(urdf, space="  ")
    ET.ElementTree(urdf).write(URDF_PATH, encoding="utf-8",
                                xml_declaration=True)
    print(f"\nWrote URDF to: {URDF_PATH}")

if __name__ == "__main__":
    convert()
