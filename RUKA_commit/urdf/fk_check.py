"""Forward kinematics sanity check on the generated URDF.

For each finger, walk from Palm_Link to the tip link with all joints at 0,
and print the world-frame position of the fingertip's TIP visual marker.

What we expect for a "left hand", looking at the palm from the back of the
hand (Z-up convention):
  - Thumb on the LEFT side (negative X) if it's a left hand
  - Pinky on the RIGHT side (positive X)
  - Fingers extending upward (positive Z) since we rotated to Z-up

(This is one convention; the actual side depends on how Isaac will display.)
"""
import xml.etree.ElementTree as ET
import numpy as np

URDF_PATH = "/home/claude/ruka/ruka_left.urdf"

def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

tree = ET.parse(URDF_PATH)
root = tree.getroot()

links = {l.get("name"): l for l in root.findall("link")}
joints = root.findall("joint")
# child_link -> joint
child_to_joint = {j.find("child").get("link"): j for j in joints}

def pose_chain(target_link):
    """Return list of (parent, child, T_parent_child_at_zero) from root to target."""
    chain = []
    cur = target_link
    while cur in child_to_joint:
        j = child_to_joint[cur]
        parent = j.find("parent").get("link")
        origin = j.find("origin")
        xyz = np.array([float(x) for x in origin.get("xyz").split()])
        rpy = np.array([float(x) for x in origin.get("rpy").split()])
        T = np.eye(4)
        T[:3,:3] = rpy_to_R(rpy)
        T[:3, 3] = xyz
        chain.append((parent, cur, T))
        cur = parent
    return list(reversed(chain))

def fk(target_link):
    T = np.eye(4)
    for parent, child, T_local in pose_chain(target_link):
        T = T @ T_local
    return T

def tip_local_pos(link_name):
    """Read the TIP visual marker's local position from the link."""
    if link_name not in links:
        return None
    l = links[link_name]
    for v in l.findall("visual"):
        if v.get("name", "").endswith("_TIP"):
            o = v.find("origin")
            return np.array([float(x) for x in o.get("xyz").split()])
    return None

# RUKA tip links (the most distal ones)
tips = {
    "Index":  "Index_PIP_Link",
    "Middle": "Middle_PIP_Link",
    "Ring":   "Ring_PIP_Link",
    "Pinky":  "Pinky_PIP_Link",
    "Thumb":  "Thumb_PIP_Link",
}

print(f"{'Finger':8s} {'tip world pos (x,y,z)':32s}")
print("-" * 50)
for name, link in tips.items():
    T = fk(link)
    tip_local = tip_local_pos(link)
    if tip_local is None:
        # use link origin
        p = T[:3, 3]
    else:
        p = (T @ np.array([*tip_local, 1.0]))[:3]
    print(f"{name:8s} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

# Also print Palm position of each finger's MCP joint
print()
print("MCP joint origins in palm frame (where each finger attaches):")
print(f"{'Finger':8s} {'pos':32s}")
for name, link in {"Index":"Index_MCP_Link", "Middle":"Middle_MCP_Link",
                   "Ring":"Ring_MCP_Link", "Pinky":"Pinky_MCP_Link",
                   "Thumb":"Thumb_MCP_Link"}.items():
    j = child_to_joint[link]
    o = j.find("origin")
    xyz = [float(x) for x in o.get("xyz").split()]
    print(f"{name:8s} ({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})")
