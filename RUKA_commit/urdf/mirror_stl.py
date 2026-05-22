"""
Mirror all STL meshes along the X axis to create a left-hand variant.

For each mesh:
  - Negate the X component of every vertex
  - Reverse triangle winding order (so normals still point outward)
  - Negate the X component of each face normal

Output goes to meshes_left/ next to the original meshes/ folder.
This way the URDF can reference 'meshes_left/Palm.STL' for a true
left-handed model where both the link frames AND the mesh geometry
are mirrored, giving a physically correct left hand.
"""
import struct
import os
from pathlib import Path

INPUT_DIR = "/home/claude/ruka/meshes"  # placeholder; real input on Windows
OUTPUT_DIR = "/home/claude/ruka/meshes_left"

def mirror_binary_stl(input_path, output_path):
    """Read a binary STL, mirror X, reverse winding, write out."""
    with open(input_path, "rb") as f:
        header = f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        triangles = []
        for _ in range(n_tri):
            data = f.read(50)
            if len(data) < 50:
                raise IOError("Truncated STL")
            # 12 floats (normal + 3 verts) + 2 byte attribute
            nx, ny, nz = struct.unpack("<fff", data[0:12])
            v1 = struct.unpack("<fff", data[12:24])
            v2 = struct.unpack("<fff", data[24:36])
            v3 = struct.unpack("<fff", data[36:48])
            attr = data[48:50]
            # Mirror X
            nx = -nx
            v1 = (-v1[0], v1[1], v1[2])
            v2 = (-v2[0], v2[1], v2[2])
            v3 = (-v3[0], v3[1], v3[2])
            # Reverse winding (swap v2 and v3) so the front face stays
            # consistent after the X reflection.
            triangles.append((nx, ny, nz, v1, v3, v2, attr))

    with open(output_path, "wb") as f:
        f.write(b"left-hand mirror of " + os.path.basename(input_path).encode()[:50])
        # pad to 80 bytes
        f.write(b"\0" * max(0, 80 - f.tell()))
        f.write(struct.pack("<I", len(triangles)))
        for nx, ny, nz, v1, v2, v3, attr in triangles:
            f.write(struct.pack("<fff", nx, ny, nz))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<fff", *v3))
            f.write(attr)

def mirror_ascii_stl(input_path, output_path):
    """Handle the rare ASCII STL format."""
    with open(input_path, "r") as f:
        text = f.read()
    # Quick & dirty parser
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("facet normal"):
            parts = ln.split()
            nx, ny, nz = -float(parts[2]), float(parts[3]), float(parts[4])
            out.append(f"  facet normal {nx} {ny} {nz}")
            i += 1
            out.append(lines[i])  # outer loop
            i += 1
            verts = []
            for k in range(3):
                p = lines[i].strip().split()
                vx, vy, vz = -float(p[1]), float(p[2]), float(p[3])
                verts.append((vx, vy, vz))
                i += 1
            # Reverse winding
            verts = [verts[0], verts[2], verts[1]]
            for v in verts:
                out.append(f"      vertex {v[0]} {v[1]} {v[2]}")
            out.append(lines[i])  # endloop
            i += 1
            out.append(lines[i])  # endfacet
            i += 1
        else:
            out.append(lines[i])
            i += 1
    with open(output_path, "w") as f:
        f.write("\n".join(out))

def is_binary_stl(path):
    """STL is binary unless it starts with 'solid' AND is mostly ASCII."""
    with open(path, "rb") as f:
        head = f.read(5)
    if head != b"solid":
        return True
    # Some binary files also start with 'solid' so check size:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(80)
        n_tri_bytes = f.read(4)
    if len(n_tri_bytes) == 4:
        n_tri = struct.unpack("<I", n_tri_bytes)[0]
        expected = 80 + 4 + n_tri * 50
        if expected == file_size:
            return True
    return False

def mirror_dir(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stl in in_dir.glob("*.STL"):
        out = out_dir / stl.name
        if is_binary_stl(stl):
            mirror_binary_stl(stl, out)
        else:
            mirror_ascii_stl(stl, out)
        print(f"  mirrored: {stl.name}")
    # Also handle .stl (lowercase) and non-STL meshes (.obj copied as-is)
    for stl in in_dir.glob("*.stl"):
        out = out_dir / stl.name
        if is_binary_stl(stl):
            mirror_binary_stl(stl, out)
        else:
            mirror_ascii_stl(stl, out)
        print(f"  mirrored: {stl.name}")

if __name__ == "__main__":
    import sys
    in_dir = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    print(f"Mirroring {in_dir} -> {out_dir}")
    mirror_dir(in_dir, out_dir)
    print("Done.")
