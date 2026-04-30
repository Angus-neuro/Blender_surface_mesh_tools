# common.py — shared helpers for all tabs (no panels/operators here)

from __future__ import annotations
import os
import numpy as np

try:
    import bpy
except Exception:
    bpy = None

# ---------------- paths ----------------

def abspath(path_in_ui: str) -> str:
    """Expand Blender-style path; return '' if empty/invalid instead of '.'."""
    if not path_in_ui or str(path_in_ui).strip() == "":
        return ""

    try:
        p = bpy.path.abspath(path_in_ui) if bpy else os.path.expanduser(path_in_ui)
    except Exception:
        p = os.path.expanduser(path_in_ui)

    p = os.path.expanduser(p)
    p = os.path.normpath(p)

    # If the result is '.' treat it as empty
    if p in (".", "./"):
        return ""

    return p

# Back-compat alias some code used earlier
_abspath = abspath

# ---------------- nibabel (lazy) ----------------

_nib = None

def get_nib():
    """Lazy import nibabel so the add-on loads even if nibabel is absent."""
    global _nib
    if _nib is None:
        import nibabel as nib  # raises if missing
        _nib = nib
    return _nib

class _NibProxy:
    def __getattr__(self, name):
        return getattr(get_nib(), name)

nib = _NibProxy()

# ---------------- math helpers ----------------

def apply_affine_np(M4x4: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 affine to Nx3 points (NumPy)."""
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    P = np.hstack([pts.astype(np.float64), ones])
    return (M4x4 @ P.T).T[:, :3]

def sample_trilinear_with_mask(volume: np.ndarray, ijk: np.ndarray):
    """Trilinear sample volume at floating ijk (Nx3). Returns (values, valid_mask)."""
    nx, ny, nz = volume.shape

    i = ijk[:, 0]
    j = ijk[:, 1]
    k = ijk[:, 2]

    valid = (
        (i >= 0) & (i <= nx - 1) &
        (j >= 0) & (j <= ny - 1) &
        (k >= 0) & (k <= nz - 1)
    )

    out = np.zeros(ijk.shape[0], dtype=np.float32)

    if not np.any(valid):
        return out, valid

    iv = i[valid]
    jv = j[valid]
    kv = k[valid]

    i0 = np.floor(iv).astype(int)
    j0 = np.floor(jv).astype(int)
    k0 = np.floor(kv).astype(int)

    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    k1 = np.clip(k0 + 1, 0, nz - 1)

    di = iv - i0
    dj = jv - j0
    dk = kv - k0

    c000 = volume[i0, j0, k0]
    c100 = volume[i1, j0, k0]
    c010 = volume[i0, j1, k0]
    c110 = volume[i1, j1, k0]
    c001 = volume[i0, j0, k1]
    c101 = volume[i1, j0, k1]
    c011 = volume[i0, j1, k1]
    c111 = volume[i1, j1, k1]

    c00 = c000 * (1 - di) + c100 * di
    c01 = c001 * (1 - di) + c101 * di
    c10 = c010 * (1 - di) + c110 * di
    c11 = c011 * (1 - di) + c111 * di

    c0 = c00 * (1 - dj) + c10 * dj
    c1 = c01 * (1 - dj) + c11 * dj

    vals = c0 * (1 - dk) + c1 * dk

    out[valid] = vals.astype(np.float32)

    return out, valid

def auto_vmin_vmax(vals: np.ndarray, *, symmetric=False, p_lo=2.0, p_hi=98.0):
    """Percentile-based display range. If symmetric=True, use ±vmax from |vals|."""
    finite = vals[np.isfinite(vals)]

    if finite.size == 0:
        return 0.0, 1.0

    if symmetric:
        vmax = float(np.percentile(np.abs(finite), p_hi)) or 1.0
        return -vmax, vmax

    vmin = float(np.percentile(finite, p_lo))
    vmax = float(np.percentile(finite, p_hi))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return vmin, vmax

# ---------------- FS tkRAS -> target voxel ----------------

def build_tkras_to_target_vox(fs_ref_path: str, target_img) -> np.ndarray:
    """
    Return 4x4 mapping tkRAS -> voxel(target) using nu.mgz header:

        T = inv(vox2ras(target)) @ vox2ras(ref) @ inv(vox2ras_tkr(ref))
    """
    nb = get_nib()
    ref = nb.load(abspath(fs_ref_path))

    try:
        M_tkr = ref.header.get_vox2ras_tkr()   # tkRAS <- vox(ref)
        A_ref = ref.header.get_vox2ras()       # scannerRAS <- vox(ref)
    except Exception as e:
        raise RuntimeError(
            "FS reference must be MGZ/MGH with vox2ras_tkr, e.g. nu.mgz."
        ) from e

    A_tgt = np.asarray(target_img.affine, dtype=np.float64)

    return np.linalg.inv(A_tgt) @ A_ref @ np.linalg.inv(M_tkr)

# ---------------- colour utilities ----------------

def write_colours(mesh, rgba: np.ndarray, attr_name: str = "Col"):
    """Create/overwrite a per-vertex colour layer with RGBA values."""
    import bpy

    if hasattr(mesh, "color_attributes"):  # Blender 3.x+
        ca = mesh.color_attributes.get(attr_name)

        if ca and (ca.domain != 'POINT' or ca.data_type not in {'FLOAT_COLOR', 'BYTE_COLOR'}):
            mesh.color_attributes.remove(ca)
            ca = None

        if not ca:
            ca = mesh.color_attributes.new(attr_name, 'FLOAT_COLOR', 'POINT')

        n = min(len(ca.data), rgba.shape[0])

        for i in range(n):
            ca.data[i].color = rgba[i].tolist()

        mesh.color_attributes.active_color = ca
        mesh.update()

    else:  # Blender 2.93
        vcols = mesh.vertex_colors
        layer = vcols.get(attr_name) or vcols.new(name=attr_name)

        for poly in mesh.polygons:
            for li in poly.loop_indices:
                vi = mesh.loops[li].vertex_index
                c = rgba[vi]
                layer.data[li].color = (
                    float(c[0]),
                    float(c[1]),
                    float(c[2]),
                    float(c[3]),
                )

        mesh.update()

def _normalize(vals, vmin, vmax):
    return np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)

def cmap_bluered(vals, vmin, vmax):
    t = (2.0 * (vals - vmin) / (vmax - vmin)) - 1.0
    t = np.clip(t, -1.0, 1.0)

    rgb = np.zeros((vals.shape[0], 3), dtype=np.float32)

    neg = t < 0
    pos = ~neg

    tn = -t[neg]
    tp = t[pos]

    rgb[neg, :] = np.stack(
        [1.0 - tn, 1.0 - tn, np.ones_like(tn)],
        axis=1,
    )

    rgb[pos, :] = np.stack(
        [np.ones_like(tp), 1.0 - tp, 1.0 - tp],
        axis=1,
    )

    return np.clip(rgb, 0, 1)

def cmap_greys(vals, vmin, vmax):
    t = _normalize(vals, vmin, vmax)
    return np.stack([t, t, t], axis=1)

CMAP_FUNCS = {
    "BLUERED": cmap_bluered,
    "GREYS": cmap_greys,
}

def map_vals_to_rgba(vals, vmin, vmax, palette="BLUERED", invert=False):
    """Map scalar values to RGBA using a small set of palettes."""
    if invert:
        vmin, vmax = vmax, vmin

    cmap = CMAP_FUNCS.get(palette, cmap_bluered)
    rgb = cmap(vals, vmin, vmax)

    return np.concatenate(
        [rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

def ensure_material_with_underlay(
    obj,
    stats_attr="Col",
    underlay_attr="Curv",
    *,
    epsilon=1e-3,
    use_underlay=True,
    show_stats=True,
):
    """Wire material so stats overlay curvature where stats != 0."""
    import bpy

    mat = obj.data.materials[0] if obj.data.materials else bpy.data.materials.new("LayerTools")

    if not obj.data.materials:
        obj.data.materials.append(mat)

    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    out = next(
        (n for n in nodes if n.type == 'OUTPUT_MATERIAL'),
        nodes.new("ShaderNodeOutputMaterial"),
    )

    bsdf = next(
        (n for n in nodes if n.type == 'BSDF_PRINCIPLED'),
        nodes.new("ShaderNodeBsdfPrincipled"),
    )

    out.location = (400, 0)
    bsdf.location = (100, 0)

    if not any(l.to_node == out and l.from_node == bsdf for l in links):
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    def attr(name, loc):
        n = nodes.new("ShaderNodeAttribute")
        n.attribute_name = name
        n.location = loc
        return n

    statsN = attr(stats_attr, (-500, 60))
    curvN = attr(underlay_attr, (-500, -80))

    base_in = bsdf.inputs['Base Color']

    for l in list(base_in.links):
        links.remove(l)

    if show_stats and not use_underlay:
        links.new(statsN.outputs['Color'], base_in)
        return

    if not show_stats and use_underlay:
        links.new(curvN.outputs['Color'], base_in)
        return

    if not show_stats and not use_underlay:
        return

    lengthN = nodes.new("ShaderNodeVectorMath")
    lengthN.operation = 'LENGTH'
    lengthN.location = (-300, 60)

    gtN = nodes.new("ShaderNodeMath")
    gtN.operation = 'GREATER_THAN'
    gtN.inputs[1].default_value = float(epsilon)
    gtN.location = (-120, 60)

    mixN = nodes.new("ShaderNodeMixRGB")
    mixN.location = (-120, -40)
    mixN.blend_type = 'MIX'

    links.new(statsN.outputs['Color'], lengthN.inputs[0])
    links.new(lengthN.outputs['Value'], gtN.inputs[0])
    links.new(gtN.outputs['Value'], mixN.inputs['Fac'])
    links.new(curvN.outputs['Color'], mixN.inputs['Color1'])
    links.new(statsN.outputs['Color'], mixN.inputs['Color2'])
    links.new(mixN.outputs['Color'], base_in)

__all__ = [
    "abspath",
    "_abspath",
    "nib",
    "get_nib",
    "apply_affine_np",
    "sample_trilinear_with_mask",
    "auto_vmin_vmax",
    "build_tkras_to_target_vox",
    "write_colours",
    "map_vals_to_rgba",
    "ensure_material_with_underlay",
]