"""Validate generated URDF for common issues that break Isaac Sim import."""
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
import sys

URDF_PATH = "/home/claude/ruka/ruka_left.urdf"

tree = ET.parse(URDF_PATH)
root = tree.getroot()

errors = []
warnings = []

# 1) Collect links and joints
links = {l.get("name"): l for l in root.findall("link")}
joints = root.findall("joint")
joint_pairs = [(j.get("name"), j.get("type"),
                j.find("parent").get("link"),
                j.find("child").get("link")) for j in joints]

print(f"Links: {len(links)}")
print(f"Joints: {len(joints)} ({sum(1 for j in joints if j.get('type') != 'fixed')} non-fixed)")

# 2) Tree structure: every link except the root must be a child of exactly one joint
child_count = defaultdict(int)
parent_count = defaultdict(int)
for jn, jt, p, c in joint_pairs:
    child_count[c] += 1
    parent_count[p] += 1
    if p not in links:
        errors.append(f"Joint {jn}: parent link '{p}' not defined")
    if c not in links:
        errors.append(f"Joint {jn}: child link '{c}' not defined")

roots = [l for l in links if child_count[l] == 0]
if len(roots) != 1:
    errors.append(f"Expected exactly 1 root link, got {len(roots)}: {roots}")
else:
    print(f"Root link: {roots[0]}")

# Multiple parent check
for l, n in child_count.items():
    if n > 1:
        errors.append(f"Link {l} has {n} parents (URDF must be a tree)")

# 3) Inertia validity for every link
for ln, l in links.items():
    inertial = l.find("inertial")
    if inertial is None:
        warnings.append(f"Link {ln} has no inertial (Isaac will assign defaults)")
        continue
    mass = float(inertial.find("mass").get("value"))
    if mass <= 0:
        errors.append(f"Link {ln}: non-positive mass {mass}")
    I = inertial.find("inertia")
    ixx = float(I.get("ixx")); iyy = float(I.get("iyy")); izz = float(I.get("izz"))
    ixy = float(I.get("ixy")); ixz = float(I.get("ixz")); iyz = float(I.get("iyz"))
    M = np.array([[ixx, ixy, ixz],
                  [ixy, iyy, iyz],
                  [ixz, iyz, izz]])
    eigs = np.linalg.eigvalsh(M)
    if (eigs <= 0).any():
        errors.append(f"Link {ln}: inertia not PD, eigenvalues={eigs}")
    elif (eigs < 1e-12).any():
        warnings.append(f"Link {ln}: inertia near-singular, eigenvalues={eigs}")
    # Triangle inequality on principal moments (often required by physics engines)
    p = sorted(eigs.tolist())
    if p[0] + p[1] < p[2] - 1e-9:
        warnings.append(f"Link {ln}: principal inertia violates triangle inequality "
                        f"(I3 > I1 + I2): {p}")

# 4) Joint validity
for j in joints:
    jn = j.get("name")
    jt = j.get("type")
    if jt in ("revolute", "prismatic"):
        ax = j.find("axis")
        if ax is None:
            errors.append(f"Joint {jn}: type {jt} requires axis")
        else:
            v = np.array([float(x) for x in ax.get("xyz").split()])
            n = np.linalg.norm(v)
            if abs(n - 1.0) > 1e-3:
                warnings.append(f"Joint {jn}: axis not unit length, norm={n}")
        lim = j.find("limit")
        if lim is None and jt == "revolute":
            warnings.append(f"Joint {jn}: revolute without limit (treated as continuous)")
        elif lim is not None:
            lo = float(lim.get("lower"))
            hi = float(lim.get("upper"))
            if hi < lo:
                errors.append(f"Joint {jn}: limit upper {hi} < lower {lo}")
            if "effort" not in lim.attrib or "velocity" not in lim.attrib:
                warnings.append(f"Joint {jn}: limit missing effort/velocity")

# 5) Print summary
print()
print(f"Errors: {len(errors)}")
for e in errors:
    print(f"  ERROR: {e}")
print(f"Warnings: {len(warnings)}")
for w in warnings:
    print(f"  WARN:  {w}")

if errors:
    sys.exit(1)
print("\nURDF passes basic validation.")
