# tab_nifti_paint.py — NIfTI Paint (3D/4D) + Curvature + Link Inflated + Plot to Image
# Adds a fast path to paint directly on NIFTI slice meshes (built by the slice add-on)
# by using per-face (i,j,k) attributes + affine-aware sampling.
#
# V1.1 NEW:
#   Adds a Surface coordinate space toggle:
#       1) FreeSurfer tkRAS
#       2) Scanner/NIfTI RAS
#
#   Adds unique paint layers:
#       Each paint can create a unique colour attribute and a unique material.
#       This allows previously painted maps to remain switchable from the
#       mesh material slots instead of being overwritten.
#
#   FreeSurfer tkRAS:
#       Use for recon-all surfaces such as lh.white/rh.white/lh.pial/rh.pial.
#       Requires an FS reference volume such as nu.mgz.
#
#   Scanner/NIfTI RAS:
#       Use for meshes whose vertices are already in scanner/world RAS mm space,
#       for example many Connectome Workbench / GIFTI / OBJ exports.
#       Does not require nu.mgz.

from __future__ import annotations

import os
import numpy as np

import bpy
import bmesh
from mathutils import Vector
from mathutils.kdtree import KDTree

from .common import (
    abspath,
    get_nib,
    apply_affine_np,
    sample_trilinear_with_mask,
    auto_vmin_vmax,
    build_tkras_to_target_vox,
)

# -------------------------------------------------------------------
# Palette helpers
# -------------------------------------------------------------------

def _normalize(vals, vmin, vmax):
    return np.clip((vals - vmin) / max(1e-12, (vmax - vmin)), 0.0, 1.0)


def cmap_greys(vals, vmin, vmax):
    t = _normalize(vals, vmin, vmax)
    return np.stack([t, t, t], axis=1)


def cmap_bluered(vals, vmin, vmax):
    t = (2.0 * (vals - vmin) / max(1e-12, (vmax - vmin))) - 1.0
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


def cmap_jet(vals, vmin, vmax):
    t = _normalize(vals, vmin, vmax)

    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)

    return np.stack([r, g, b], axis=1)


def cmap_cool(vals, vmin, vmax):
    t = _normalize(vals, vmin, vmax)

    r = t
    g = 1.0 - t
    b = np.ones_like(t)

    return np.stack([r, g, b], axis=1)


def cmap_hot(vals, vmin, vmax):
    t = _normalize(vals, vmin, vmax)

    r = np.clip(3 * t, 0, 1)
    g = np.clip(3 * t - 1, 0, 1)
    b = np.clip(3 * t - 2, 0, 1)

    return np.stack([r, g, b], axis=1)


_POLAR_COLORS = [
    (255, 0, 0),   (255, 28, 0), (255, 57, 0), (255, 85, 0), (255, 113, 0),
    (255, 142, 0), (255, 170, 0), (255, 198, 0), (255, 227, 0), (255, 255, 0),
    (0, 0, 255),   (0, 28, 227), (0, 57, 198), (0, 85, 170), (0, 113, 142),
    (0, 142, 113), (0, 170, 85), (0, 198, 57), (0, 227, 28), (0, 255, 0),
]

_ECC_COLORS = [
    (255, 0, 0), (255, 36, 0), (255, 73, 0), (255, 109, 0), (255, 146, 0),
    (255, 182, 0), (255, 219, 0), (255, 255, 0), (191, 255, 0), (128, 255, 0),
    (64, 255, 0), (0, 255, 0), (0, 242, 64), (0, 230, 128), (0, 217, 191),
    (0, 204, 255), (0, 153, 255), (0, 102, 255), (0, 51, 255), (0, 0, 255),
]


def _cmap_discrete_from_list(vals, vmin, vmax, colors_255):
    t = _normalize(vals, vmin, vmax)

    n = len(colors_255)
    idx = np.floor(t * (n - 1) + 1e-9).astype(int)
    idx = np.clip(idx, 0, n - 1)

    return (np.asarray(colors_255, dtype=np.float32) / 255.0)[idx]


def cmap_polar(vals, vmin, vmax):
    return _cmap_discrete_from_list(vals, vmin, vmax, _POLAR_COLORS)


def cmap_eccentricity(vals, vmin, vmax):
    return _cmap_discrete_from_list(vals, vmin, vmax, _ECC_COLORS)


CMAP_FUNCS = {
    'BLUERED':      cmap_bluered,
    'GREYS':        cmap_greys,
    'JET':          cmap_jet,
    'COOL':         cmap_cool,
    'HOT':          cmap_hot,
    'POLAR':        cmap_polar,
    'ECCENTRICITY': cmap_eccentricity,
}


def map_vals_to_rgba(vals, vmin, vmax, palette='BLUERED', invert=False):
    cmap = CMAP_FUNCS.get(palette, cmap_bluered)

    if invert:
        vmin, vmax = vmax, vmin

    rgb = cmap(vals, vmin, vmax)

    return np.concatenate(
        [rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)],
        axis=1,
    )


# -------------------------------------------------------------------
# Mesh colour utilities (Blender 2.93–4.5)
# -------------------------------------------------------------------

def write_colours_blender_3x(mesh, rgba, attr_name):
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


def write_colours_blender_293(mesh, rgba, layer_name):
    vcols = getattr(mesh, "vertex_colors", None) or mesh.vertex_colors
    layer = vcols.get(layer_name) or vcols.new(name=layer_name)
    colours = layer.data

    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            c = rgba[vi]
            colours[li].color = (
                float(c[0]),
                float(c[1]),
                float(c[2]),
                float(c[3]),
            )

    mesh.update()


def write_colours(mesh, rgba, attr_name):
    if hasattr(mesh, "color_attributes"):
        write_colours_blender_3x(mesh, rgba, attr_name)
    else:
        write_colours_blender_293(mesh, rgba, attr_name)


def get_vertex_rgba_or_none(obj, attr_name="Col"):
    me = obj.data

    if hasattr(me, "color_attributes"):
        ca = me.color_attributes.get(attr_name)

        if ca and ca.domain == 'POINT':
            out = np.zeros((len(me.vertices), 4), dtype=np.float32)
            n = min(len(ca.data), len(me.vertices))

            for i in range(n):
                out[i, :] = np.array(ca.data[i].color, dtype=np.float32)

            return out

        return None

    vcols = getattr(me, "vertex_colors", None)

    if not vcols:
        return None

    layer = vcols.get(attr_name)

    if not layer:
        return None

    acc = np.zeros((len(me.vertices), 4), dtype=np.float32)
    cnt = np.zeros(len(me.vertices), dtype=np.int32)

    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            acc[vi] += np.array(layer.data[li].color, dtype=np.float32)
            cnt[vi] += 1

    cnt = np.maximum(cnt, 1)

    acc[:, 0] /= cnt
    acc[:, 1] /= cnt
    acc[:, 2] /= cnt
    acc[:, 3] /= cnt

    return acc


# -------------------------------------------------------------------
# Material wiring: unique paint materials + mix stats over curvature
# -------------------------------------------------------------------

def _safe_name(s: str, max_len: int = 48) -> str:
    s = str(s or "").strip()

    bad = '<>:"/\\|?* \t\n\r'

    for ch in bad:
        s = s.replace(ch, "_")

    while "__" in s:
        s = s.replace("__", "_")

    s = s.strip("_")

    if not s:
        s = "Paint"

    return s[:max_len]


def _nifti_stem(path: str) -> str:
    base = os.path.basename(str(path or ""))

    for suf in (".nii.gz", ".nii", ".mgz", ".mgh", ".gz"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
            break

    return _safe_name(base or "map")


def _unique_colour_attr_name(mesh, base_name: str) -> str:
    base = _safe_name(base_name, 40)

    if hasattr(mesh, "color_attributes"):
        getter = mesh.color_attributes.get
    else:
        getter = mesh.vertex_colors.get

    if getter(base) is None:
        return base

    i = 1

    while True:
        name = f"{base}_{i:03d}"

        if getter(name) is None:
            return name

        i += 1


def _unique_material_name(base_name: str) -> str:
    base = _safe_name(base_name, 60)

    if bpy.data.materials.get(base) is None:
        return base

    i = 1

    while True:
        name = f"{base}_{i:03d}"

        if bpy.data.materials.get(name) is None:
            return name

        i += 1


def make_unique_paint_names(obj, nifti_path: str, attr_base: str = "Col"):
    """
    Return a unique vertex colour attribute name and material name for this paint.

    A unique material alone is not enough. The material must read a unique
    colour attribute, otherwise all old materials would still show the newest
    paint because they would all read the overwritten 'Col' layer.
    """
    stem = _nifti_stem(nifti_path)
    obj_stem = _safe_name(obj.name, 32)

    attr_prefix = _safe_name(attr_base or "Col", 16)
    attr_name = _unique_colour_attr_name(obj.data, f"{attr_prefix}_{stem}")

    mat_name = _unique_material_name(f"NiftiPaint_{obj_stem}_{stem}")

    return attr_name, mat_name


def _node_by_type(nodes, t):
    for n in nodes:
        if n.type == t:
            return n

    return None


def _assign_material_to_all_faces(obj, slot_index: int):
    """Make the whole mesh display the chosen material slot."""
    if not obj or obj.type != 'MESH':
        return

    me = obj.data
    slot_index = int(slot_index)

    for poly in me.polygons:
        poly.material_index = slot_index

    obj.active_material_index = slot_index
    me.update()


def _ensure_material_slot(obj, mat):
    """Append material to obj if needed and return its slot index."""
    for i, m in enumerate(obj.data.materials):
        if m == mat:
            return i

    obj.data.materials.append(mat)

    return len(obj.data.materials) - 1


def _configure_nifti_material(mat, stats_attr="Col", underlay_attr="Curv",
                              epsilon=1e-3, use_underlay=True, show_stats=True):
    """
    Build/rebuild a material that displays one specific stats colour attribute,
    optionally mixed over one curvature underlay attribute.
    """
    mat.use_nodes = True

    mat["nifti_paint_material"] = True
    mat["nifti_paint_stats_attr"] = str(stats_attr)
    mat["nifti_paint_underlay_attr"] = str(underlay_attr)

    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (500, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)

    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    base_in = bsdf.inputs['Base Color']
    base_in.default_value = (0.7, 0.7, 0.7, 1.0)

    def attr_node(name, locx=-450, locy=0):
        n = nodes.new("ShaderNodeAttribute")
        n.attribute_name = name
        n.location = (locx, locy)
        return n

    statsN = attr_node(stats_attr, -500, 80)
    curvN = attr_node(underlay_attr, -500, -80)

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
    lengthN.location = (-280, 80)

    gtN = nodes.new("ShaderNodeMath")
    gtN.operation = 'GREATER_THAN'
    gtN.inputs[1].default_value = float(epsilon)
    gtN.location = (-80, 80)

    mixN = nodes.new("ShaderNodeMixRGB")
    mixN.location = (70, -40)
    mixN.blend_type = 'MIX'

    links.new(statsN.outputs['Color'], lengthN.inputs[0])
    links.new(lengthN.outputs['Value'], gtN.inputs[0])
    links.new(gtN.outputs['Value'], mixN.inputs['Fac'])
    links.new(curvN.outputs['Color'], mixN.inputs['Color1'])
    links.new(statsN.outputs['Color'], mixN.inputs['Color2'])
    links.new(mixN.outputs['Color'], base_in)


def ensure_material_with_underlay(obj, stats_attr="Col", underlay_attr="Curv",
                                  epsilon=1e-3, use_underlay=True, show_stats=True,
                                  material_name=None, assign_to_mesh=True):
    """
    If material_name is provided, a named unique material is created/appended.
    The material reads only stats_attr, so older paint maps remain switchable.

    If material_name is None, a reusable legacy material named 'NiftiPaint' is
    used. This avoids overwriting old unique materials if the user temporarily
    disables unique-material mode.
    """
    if not obj or obj.type != 'MESH':
        return None

    if material_name:
        mat = bpy.data.materials.get(material_name)

        if mat is None:
            mat = bpy.data.materials.new(name=material_name)

        slot = _ensure_material_slot(obj, mat)

        if assign_to_mesh:
            _assign_material_to_all_faces(obj, slot)

    else:
        mat = None
        slot = None

        for i, m in enumerate(obj.data.materials):
            if m and m.name == "NiftiPaint":
                mat = m
                slot = i
                break

        if mat is None:
            mat = bpy.data.materials.new(name="NiftiPaint")
            slot = _ensure_material_slot(obj, mat)

        if assign_to_mesh:
            _assign_material_to_all_faces(obj, slot)

    _configure_nifti_material(
        mat,
        stats_attr=stats_attr,
        underlay_attr=underlay_attr,
        epsilon=epsilon,
        use_underlay=use_underlay,
        show_stats=show_stats,
    )

    return mat


def _iter_nifti_paint_materials(obj):
    if not obj or obj.type != 'MESH':
        return

    seen = set()

    for mat in obj.data.materials:
        if not mat:
            continue

        if mat.as_pointer() in seen:
            continue

        seen.add(mat.as_pointer())

        if bool(mat.get("nifti_paint_material", False)):
            yield mat


def _rebuild_nifti_materials_for_obj(obj, props):
    """
    Rebuild all existing NIfTI paint materials on this object using the current
    underlay/show toggles, while preserving the unique stats attribute each one
    reads.
    """
    mats = list(_iter_nifti_paint_materials(obj))

    if mats:
        for mat in mats:
            stats_attr = str(mat.get("nifti_paint_stats_attr", props.attribute_name))
            underlay_attr = str(mat.get("nifti_paint_underlay_attr", "Curv"))

            _configure_nifti_material(
                mat,
                stats_attr=stats_attr,
                underlay_attr=underlay_attr,
                epsilon=1e-3,
                use_underlay=props.use_curvature_underlay,
                show_stats=props.show_stats_map,
            )

    else:
        ensure_material_with_underlay(
            obj,
            stats_attr=props.attribute_name,
            underlay_attr="Curv",
            epsilon=1e-3,
            use_underlay=props.use_curvature_underlay,
            show_stats=props.show_stats_map,
            material_name=None,
            assign_to_mesh=False,
        )


def _copy_nifti_material_slots(source_obj, target_obj):
    """
    Make sure the linked inflated mesh has the same NIfTI paint materials.
    """
    if not source_obj or not target_obj:
        return

    active_mat = source_obj.active_material

    for mat in source_obj.data.materials:
        if mat and bool(mat.get("nifti_paint_material", False)):
            _ensure_material_slot(target_obj, mat)

    if active_mat:
        for i, m in enumerate(target_obj.data.materials):
            if m == active_mat:
                _assign_material_to_all_faces(target_obj, i)
                break


def _get_linked_inflated(base_obj):
    name = base_obj.get("inflated_link_name", "")

    if name and name in bpy.data.objects:
        return bpy.data.objects[name]

    guess = base_obj.name + "_inflated"

    return bpy.data.objects.get(guess)


def _rebuild_material_for_active_and_linked(context):
    obj = context.active_object

    if not obj or obj.type != 'MESH':
        return

    p = context.scene.nifti_paint_props

    for o in (obj, _get_linked_inflated(obj)):
        if o and o.type == 'MESH':
            _rebuild_nifti_materials_for_obj(o, p)


def _update_show_stats_map(self, context):
    try:
        _rebuild_material_for_active_and_linked(context)
    except Exception:
        pass


# -------------------------------------------------------------------
# Slice-mesh helpers (fast path)
# -------------------------------------------------------------------

def _is_slice_mesh(obj: bpy.types.Object | None) -> bool:
    if not obj or obj.type != 'MESH':
        return False

    me = obj.data
    attrs = getattr(me, "attributes", None)

    if not attrs:
        return False

    ai = attrs.get("nifti_ijk_i")
    aj = attrs.get("nifti_ijk_j")
    ak = attrs.get("nifti_ijk_k")

    return bool(
        ai and aj and ak and
        ai.domain == 'FACE' and aj.domain == 'FACE' and ak.domain == 'FACE' and
        ai.data_type == 'INT' and aj.data_type == 'INT' and ak.data_type == 'INT'
    )


def _get_slice_face_ijk_from_mesh(obj):
    """Return Nx3 int array of per-face (i,j,k) if this is a NIfTI slice mesh; else None."""
    if not _is_slice_mesh(obj):
        return None

    me = obj.data
    attrs = me.attributes

    ai = attrs.get("nifti_ijk_i")
    aj = attrs.get("nifti_ijk_j")
    ak = attrs.get("nifti_ijk_k")

    I = np.array([d.value for d in ai.data], dtype=np.int32)
    J = np.array([d.value for d in aj.data], dtype=np.int32)
    K = np.array([d.value for d in ak.data], dtype=np.int32)

    return np.stack([I, J, K], axis=1)


def _write_face_rgba_slicecol(me, rgba_per_face):
    """Write face colours into CORNER 'SliceCol' so the slice material renders crisply."""
    if hasattr(me, "color_attributes"):
        ca = me.color_attributes.get("SliceCol")

        if ca and (ca.domain != 'CORNER' or ca.data_type not in {'FLOAT_COLOR', 'BYTE_COLOR'}):
            me.color_attributes.remove(ca)
            ca = None

        if not ca:
            ca = me.color_attributes.new("SliceCol", 'FLOAT_COLOR', 'CORNER')

        layer = ca.data

    else:
        vcols = me.vertex_colors
        ca = vcols.get("SliceCol") or vcols.new(name="SliceCol")
        layer = ca.data

    for fi, f in enumerate(me.polygons):
        c = tuple(float(x) for x in rgba_per_face[fi])

        for li in f.loop_indices:
            layer[li].color = c

    me.update()

    if hasattr(me, "color_attributes"):
        me.color_attributes.active_color = me.color_attributes["SliceCol"]


def _vox_to_vox_mat(src_affine, dst_affine):
    """Map dst voxel indices -> src voxel indices."""
    return np.linalg.inv(src_affine) @ dst_affine


def build_surface_to_target_vox(surface_space: str, fs_ref_path: str, target_img):
    """
    Return a 4x4 matrix mapping mesh vertex coordinates into target_img voxel ijk.

    surface_space:
      FS_TKRAS     : mesh vertices are FreeSurfer surface/tkRAS coordinates.
      SCANNER_RAS  : mesh vertices are scanner/world RAS mm coordinates matching
                     the target NIfTI affine space.
    """
    surface_space = str(surface_space or "FS_TKRAS").upper()

    if surface_space == "FS_TKRAS":
        if not fs_ref_path:
            raise RuntimeError("FS ref (nu.mgz) is required when surface space is FreeSurfer tkRAS.")

        return build_tkras_to_target_vox(fs_ref_path, target_img)

    if surface_space == "SCANNER_RAS":
        return np.linalg.inv(target_img.affine)

    raise RuntimeError(f"Unknown surface coordinate space: {surface_space}")


def _sample_nearest_3d(arr, x, y, z):
    I, J, K = arr.shape

    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    zi = np.rint(z).astype(np.int64)

    valid = (
        (xi >= 0) &
        (yi >= 0) &
        (zi >= 0) &
        (xi < I) &
        (yi < J) &
        (zi < K)
    )

    out = np.zeros_like(x, dtype=np.float32)
    out[valid] = arr[xi[valid], yi[valid], zi[valid]]

    return out


def _sample_trilinear_3d(arr, x, y, z):
    I, J, K = arr.shape

    x = np.clip(x, 0, I - 1)
    y = np.clip(y, 0, J - 1)
    z = np.clip(z, 0, K - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)

    x1 = np.clip(x0 + 1, 0, I - 1)
    y1 = np.clip(y0 + 1, 0, J - 1)
    z1 = np.clip(z0 + 1, 0, K - 1)

    xd = (x - x0).astype(np.float32)
    yd = (y - y0).astype(np.float32)
    zd = (z - z0).astype(np.float32)

    c000 = arr[x0, y0, z0]
    c100 = arr[x1, y0, z0]
    c010 = arr[x0, y1, z0]
    c110 = arr[x1, y1, z0]

    c001 = arr[x0, y0, z1]
    c101 = arr[x1, y0, z1]
    c011 = arr[x0, y1, z1]
    c111 = arr[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return (c0 * (1 - zd) + c1 * zd).astype(np.float32)


def _copy_face_color_attr(me, src_name, dst_name):
    """Copy CORNER colour attribute src_name -> dst_name. Returns True if ok."""
    if hasattr(me, "color_attributes"):
        src = me.color_attributes.get(src_name)

        if not src or src.domain != 'CORNER':
            return False

        dst = me.color_attributes.get(dst_name)

        if not dst or dst.domain != 'CORNER':
            if dst:
                me.color_attributes.remove(dst)

            dst = me.color_attributes.new(dst_name, 'FLOAT_COLOR', 'CORNER')

        sd, dd = src.data, dst.data

        if len(dd) != len(sd):
            tmp = [None] * len(sd)

            for i in range(len(sd)):
                tmp[i] = sd[i].color[:]

            me.color_attributes.remove(dst)
            dst = me.color_attributes.new(dst_name, 'FLOAT_COLOR', 'CORNER')
            dd = dst.data

            for i in range(len(sd)):
                dd[i].color = tmp[i]

        else:
            n = min(len(sd), len(dd))

            for i in range(n):
                dd[i].color = sd[i].color

        me.update()
        me.color_attributes.active_color = dst

        return True

    vcols = me.vertex_colors
    src = vcols.get(src_name)

    if not src:
        return False

    dst = vcols.get(dst_name) or vcols.new(name=dst_name)

    for i in range(len(src.data)):
        dst.data[i].color = src.data[i].color

    me.update()

    return True


def _scene_slice_matches_meta(meta: dict) -> bool:
    """Compare current Scene slice selector with the mesh's stored meta."""
    sp = getattr(bpy.context.scene, "nifti_slice_props", None)

    if not sp or not meta:
        return True

    try:
        same_path = os.path.abspath(sp.nifti_path) == os.path.abspath(meta.get("nifti_path", ""))
    except Exception:
        same_path = False

    same_orient = getattr(sp, "orientation", None) == meta.get("orientation")

    try:
        same_slice = int(getattr(sp, "slice_index", -1)) == int(meta.get("slice_index", -999999))
    except Exception:
        same_slice = False

    return bool(same_path and same_orient and same_slice)


# -------------------------------------------------------------------
# Material underlay toggling
# -------------------------------------------------------------------

def _apply_slice_material_rules_if_needed(obj):
    """On real slice meshes we avoid interfering with their material; they use 'SliceCol'."""
    if _is_slice_mesh(obj):
        return


# -------------------------------------------------------------------
# Mesh normals/verts
# -------------------------------------------------------------------

def _mesh_local_vertices_and_normals(obj):
    me = obj.data

    if hasattr(me, "calc_normals"):
        me.calc_normals()
    elif hasattr(me, "calc_normals_split"):
        me.calc_normals_split()

    verts = np.array([tuple(v.co) for v in me.vertices], dtype=np.float64)
    norms = np.array([tuple(v.normal) for v in me.vertices], dtype=np.float64)

    norms = norms / np.maximum(
        1e-12,
        np.linalg.norm(norms, axis=1, keepdims=True),
    )

    return verts, norms


# -------------------------------------------------------------------
# Painting (3D) for cortical meshes
# -------------------------------------------------------------------

def _get_linked_white(obj):
    """Robustly return the WHITE mesh linked to `obj`, which may be white or inflated."""
    if obj.get("inflated_link_name"):
        return obj

    name = obj.get("linked_white_name") or obj.get("white_link_name")

    if name and name in bpy.data.objects:
        return bpy.data.objects[name]

    for o in bpy.data.objects:
        if o.type == 'MESH' and o.get("inflated_link_name") == obj.name:
            return o

    guess = obj.name.replace("_inflated", "")

    return bpy.data.objects.get(guess)


def _stamp_paint_metadata(base_obj, props):
    """
    Store enough info on the WHITE mesh for the Vertex Select auto-peak tool
    to re-sample the stats NIfTI.
    """
    white = _get_linked_white(base_obj) or base_obj

    try:
        white["nifti_paint_path"] = abspath(props.nifti_path)
        white["fs_ref_path"] = abspath(props.fs_ref_path)
        white["nifti_sample_neg_mm"] = float(props.sample_neg_mm)
        white["nifti_sample_pos_mm"] = float(props.sample_pos_mm)
        white["nifti_sample_step_mm"] = float(props.sample_step_mm)
        white["nifti_surface_space"] = str(props.surface_space)
        white["nifti_last_paint_attribute_name"] = str(props.last_paint_attribute_name)
        white["nifti_last_paint_material_name"] = str(props.last_paint_material_name)
    except Exception:
        pass


def _sample_points_from_mesh_or_white(obj, use_white: bool):
    """Return (P_local, N_local) from obj, optionally mapped to linked WHITE."""
    if not use_white:
        return _mesh_local_vertices_and_normals(obj)

    white = _get_linked_white(obj)

    if white:
        return _mesh_local_vertices_and_normals(white)

    return _mesh_local_vertices_and_normals(obj)


def paint_vertex_colours_fs(obj, nifti_path, fs_ref_path, attr_name="Col",
                            *, surface_space="FS_TKRAS",
                            symmetric=True, vmin=None, vmax=None,
                            p_lo=2.0, p_hi=98.0, palette='BLUERED', invert=False,
                            sample_neg_mm=0.0, sample_pos_mm=0.0, sample_step_mm=0.5,
                            use_curvature_underlay=True, show_stats_map=True,
                            sample_using_linked_white_normals=False,
                            unique_material_per_paint=True):
    nib = get_nib()

    nifti_path = abspath(nifti_path)
    fs_ref_path = abspath(fs_ref_path)

    if not os.path.isfile(nifti_path):
        raise RuntimeError(f"NIfTI not found: {nifti_path}")

    img_tgt = nib.load(nifti_path)

    if img_tgt.ndim != 3:
        raise RuntimeError("Selected NIfTI is not 3D. Use the 4D timecourse plot for 4D images.")

    vol = img_tgt.get_fdata().astype(np.float64, copy=False)

    T_surface_to_ijk = build_surface_to_target_vox(
        surface_space,
        fs_ref_path,
        img_tgt,
    )

    verts_local, norms_local = _sample_points_from_mesh_or_white(
        obj,
        sample_using_linked_white_normals,
    )

    if (sample_neg_mm > 0.0) or (sample_pos_mm > 0.0):
        neg_mm = float(max(0.0, sample_neg_mm))
        pos_mm = float(max(0.0, sample_pos_mm))
        step_mm = float(max(1e-6, sample_step_mm))

        offs_neg = np.arange(-neg_mm, 0.0, step_mm) if neg_mm > 0 else np.array([], dtype=np.float64)
        offs_pos = np.arange(step_mm, pos_mm + 1e-9, step_mm) if pos_mm > 0 else np.array([], dtype=np.float64)
        offsets = np.concatenate([offs_neg, np.array([0.0]), offs_pos], axis=0)

        acc = np.zeros(verts_local.shape[0], dtype=np.float64)
        cnt = np.zeros(verts_local.shape[0], dtype=np.int32)

        for d in offsets:
            P_surface = verts_local + norms_local * d
            ijk = apply_affine_np(T_surface_to_ijk, P_surface)
            vals, valid = sample_trilinear_with_mask(vol, ijk)

            good = valid & np.isfinite(vals)

            acc[good] += vals[good]
            cnt[good] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.where(
                cnt > 0,
                acc / np.maximum(cnt, 1),
                0.0,
            ).astype(np.float32)

    else:
        ijk = apply_affine_np(T_surface_to_ijk, verts_local)
        vals, _ = sample_trilinear_with_mask(vol, ijk)

    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = auto_vmin_vmax(
            vals,
            symmetric=symmetric,
            p_lo=p_lo,
            p_hi=p_hi,
        )

        if vmin is None:
            vmin = vmin_auto

        if vmax is None:
            vmax = vmax_auto

    if symmetric:
        a = max(abs(vmin), abs(vmax))
        vmin, vmax = -a, a

    if vmax <= vmin:
        vmax = vmin + 1e-6

    vals = np.nan_to_num(vals, nan=0.0, posinf=vmax, neginf=vmin)
    vals = np.clip(vals, vmin, vmax)

    rgba = map_vals_to_rgba(
        vals,
        vmin,
        vmax,
        palette=palette,
        invert=invert,
    )

    me = obj.data

    if unique_material_per_paint:
        paint_attr_name, paint_material_name = make_unique_paint_names(
            obj,
            nifti_path,
            attr_base=attr_name,
        )
    else:
        paint_attr_name = attr_name
        paint_material_name = None

    write_colours(me, rgba, paint_attr_name)

    mat = ensure_material_with_underlay(
        obj,
        stats_attr=paint_attr_name,
        underlay_attr="Curv",
        epsilon=1e-3,
        use_underlay=use_curvature_underlay,
        show_stats=show_stats_map,
        material_name=paint_material_name,
        assign_to_mesh=True,
    )

    infl = _get_linked_inflated(obj)

    if infl:
        _copy_attr_by_index_or_nearest(obj, infl, attr_name=paint_attr_name)

        ensure_material_with_underlay(
            infl,
            stats_attr=paint_attr_name,
            underlay_attr="Curv",
            epsilon=1e-3,
            use_underlay=use_curvature_underlay,
            show_stats=show_stats_map,
            material_name=mat.name if mat else paint_material_name,
            assign_to_mesh=True,
        )

        _copy_nifti_material_slots(obj, infl)

    return paint_attr_name, mat.name if mat else ""


# -------------------------------------------------------------------
# Curvature
# -------------------------------------------------------------------

def _compute_signed_mean_curvature(obj):
    me = obj.data

    bm = bmesh.new()
    bm.from_mesh(me)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    V = bm.verts
    E = bm.edges

    L = [Vector((0.0, 0.0, 0.0)) for _ in V]
    W = [0.0 for _ in V]

    for e in E:
        v0, v1 = e.verts
        cot_sum = 0.0

        for f in e.link_faces:
            tri = [l.vert for l in f.loops]

            if len(tri) < 3:
                continue

            k = next((vv for vv in tri if vv not in (v0, v1)), None)

            if k is None:
                continue

            a = v0.co - k.co
            b = v1.co - k.co

            cr_len = a.cross(b).length

            if cr_len > 1e-12:
                cot = a.dot(b) / cr_len
                cot = max(-1e3, min(1e3, cot))
                cot_sum += cot

        w = max(0.0, cot_sum) or 1.0
        d01 = v1.co - v0.co

        L[v0.index] += d01 * float(w)
        L[v1.index] -= d01 * float(w)

        W[v0.index] += w
        W[v1.index] += w

    for i in range(len(V)):
        if W[i] > 1e-12:
            L[i] /= float(W[i])

    k = np.array(
        [float(L[i].dot(v.normal)) for i, v in enumerate(V)],
        dtype=np.float64,
    )

    bm.free()

    return k


def paint_curvature(obj, attr_name="Curv",
                    p_hi=95.0, trough_lum=0.97, peak_lum=0.07, contrast_x=2.0):
    k = _compute_signed_mean_curvature(obj)
    finite = k[np.isfinite(k)]

    scale = float(np.percentile(np.abs(finite), p_hi)) if finite.size else 1.0

    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    t = np.clip(k / scale, -1.0, 1.0)
    t = np.clip(t * float(contrast_x), -1.0, 1.0)

    mid = 0.5 * (trough_lum + peak_lum)
    half = 0.5 * (trough_lum - peak_lum)

    L = np.clip(mid - half * t, 0.0, 1.0).astype(np.float32)

    rgba = np.stack([L, L, L, np.ones_like(L)], axis=1)

    write_colours(obj.data, rgba, attr_name)


def smooth_vertex_colour_step(obj, attr_name="Curv", lam=0.5):
    rgba = get_vertex_rgba_or_none(obj, attr_name)

    if rgba is None:
        raise RuntimeError(f"Attribute '{attr_name}' not found.")

    me = obj.data
    n = len(me.vertices)

    neigh = [[] for _ in range(n)]

    for e in me.edges:
        v0, v1 = e.vertices

        neigh[v0].append(v1)
        neigh[v1].append(v0)

    rgb = rgba[:, :3].copy()
    out = rgb.copy()

    for i in range(n):
        nb = neigh[i]

        if not nb:
            continue

        avg = np.mean(rgb[nb, :], axis=0)
        out[i, :] = (1.0 - lam) * rgb[i, :] + lam * avg

    rgba[:, :3] = out

    write_colours(me, rgba, attr_name)


# -------------------------------------------------------------------
# Link Inflated
# -------------------------------------------------------------------

def _set_link(base_obj, infl_obj):
    base_obj["inflated_link_name"] = infl_obj.name
    infl_obj["linked_white_name"] = base_obj.name


def _import_obj_compat(path: str):
    before = {o.name for o in bpy.data.objects}

    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        try:
            bpy.ops.import_scene.obj(
                filepath=path,
                use_split_objects=False,
                use_split_groups=False,
            )
        except TypeError:
            bpy.ops.import_scene.obj(filepath=path)

    return [
        o for o in bpy.data.objects
        if o.name not in before and o.type == 'MESH'
    ]


def _copy_attr_by_index_or_nearest(source_obj, target_obj, attr_name="Col"):
    me_s, me_d = source_obj.data, target_obj.data
    src_rgba = get_vertex_rgba_or_none(source_obj, attr_name)

    if src_rgba is None:
        return

    if len(me_s.vertices) == len(me_d.vertices):
        write_colours(me_d, src_rgba, attr_name)
        return

    S, D = source_obj.matrix_world, target_obj.matrix_world
    pts = [S @ v.co for v in me_s.vertices]

    tree = KDTree(len(pts))

    for i, p in enumerate(pts):
        tree.insert(p, i)

    tree.balance()

    out = np.zeros((len(me_d.vertices), 4), np.float32)

    for i, v in enumerate(me_d.vertices):
        _, j, _ = tree.find(D @ v.co)
        out[i, :] = src_rgba[j, :]

    write_colours(me_d, out, attr_name)


def _iter_colour_attribute_names(obj):
    me = obj.data

    if hasattr(me, "color_attributes"):
        for ca in me.color_attributes:
            if ca.domain == 'POINT':
                yield ca.name
    else:
        vcols = getattr(me, "vertex_colors", None)

        if vcols:
            for layer in vcols:
                yield layer.name


def link_inflated_once(base_obj, inflated_path):
    existing = _get_linked_inflated(base_obj)

    if existing is not None:
        existing.parent = base_obj
        existing.matrix_parent_inverse = base_obj.matrix_world.inverted()

        for attr in _iter_colour_attribute_names(base_obj):
            if get_vertex_rgba_or_none(base_obj, attr) is not None:
                _copy_attr_by_index_or_nearest(base_obj, existing, attr)

        _copy_nifti_material_slots(base_obj, existing)

        _set_link(base_obj, existing)

        return existing, False

    new_objs = _import_obj_compat(inflated_path)

    if not new_objs:
        raise RuntimeError("No mesh object was imported from the Inflated OBJ.")

    infl = new_objs[0]
    infl.name = base_obj.name + "_inflated"

    B = base_obj.matrix_world.copy()
    M = infl.matrix_world.copy()

    infl.data.transform(B.inverted() @ M)
    infl.matrix_world = B

    infl.parent = base_obj
    infl.matrix_parent_inverse = base_obj.matrix_world.inverted()

    _set_link(base_obj, infl)

    for attr in _iter_colour_attribute_names(base_obj):
        if get_vertex_rgba_or_none(base_obj, attr) is not None:
            _copy_attr_by_index_or_nearest(base_obj, infl, attr)

    _copy_nifti_material_slots(base_obj, infl)

    return infl, True


# -------------------------------------------------------------------
# Selection helpers for 4D plotting
# -------------------------------------------------------------------

def _selection_indices_in_edit(obj):
    if bpy.context.object != obj:
        bpy.context.view_layer.objects.active = obj

    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    vids = [v.index for v in bm.verts if v.select]

    if vids:
        return vids

    vset = set()

    for e in bm.edges:
        if e.select:
            vset.add(e.verts[0].index)
            vset.add(e.verts[1].index)

    if not vset:
        for f in bm.faces:
            if f.select:
                for v in f.verts:
                    vset.add(v.index)

    return sorted(vset)


def _selection_in_white_tkras(active_obj, white_obj):
    """Return WHITE-space verts/normals for current selection on active."""
    idxs = _selection_indices_in_edit(active_obj)

    if not idxs:
        return np.zeros((0, 3)), np.zeros((0, 3)), [], "none"

    Vw, Nw = _mesh_local_vertices_and_normals(white_obj)
    Va, Na = _mesh_local_vertices_and_normals(active_obj)

    Wa = active_obj.matrix_world
    Ww = white_obj.matrix_world

    if len(Va) == len(Vw):
        map_ids = idxs
        mode = "index"

    else:
        tree = KDTree(len(Vw))

        for i, v in enumerate(Vw):
            tree.insert(Ww @ Vector(v), i)

        tree.balance()

        map_ids = []

        for i in idxs:
            _, j, _ = tree.find(Wa @ Vector(Va[i]))
            map_ids.append(j)

        mode = "nearest"

    P = np.asarray([Vw[j] for j in map_ids], dtype=np.float64)
    N = np.asarray([Nw[j] for j in map_ids], dtype=np.float64)

    return P, N, map_ids, mode


# -------------------------------------------------------------------
# 4D timecourse computation + plotting to Image
# -------------------------------------------------------------------

def _compute_avg_timecourse_4d(obj, fs_ref_path, nifti_path,
                               surface_space="FS_TKRAS",
                               use_linked_white_normals=False,
                               neg_mm=0.0, pos_mm=0.0, step_mm=0.5):
    """Average selection timecourse from obj in Edit Mode."""
    nib = get_nib()

    img = nib.load(abspath(nifti_path))
    vol = img.get_fdata(dtype=np.float64)

    if img.ndim != 4:
        raise RuntimeError("Selected NIfTI is not 4D.")

    T = vol.shape[3]
    zooms = img.header.get_zooms()
    dt = float(zooms[3]) if len(zooms) > 3 and zooms[3] else 1.0

    T_surface_to_ijk = build_surface_to_target_vox(
        surface_space,
        fs_ref_path,
        img,
    )

    idxs = _selection_indices_in_edit(obj)

    if not idxs:
        return np.zeros(0), np.zeros(0), 0, 0.0

    if use_linked_white_normals:
        white = _get_linked_white(obj)

        if white:
            P_all, N_all, _, _ = _selection_in_white_tkras(obj, white)
        else:
            V, N = _mesh_local_vertices_and_normals(obj)
            P_all, N_all = V[idxs], N[idxs]

    else:
        V, N = _mesh_local_vertices_and_normals(obj)
        P_all, N_all = V[idxs], N[idxs]

    n_sel = int(P_all.shape[0])

    if n_sel == 0:
        return np.zeros(0), np.zeros(0), 0, 0.0

    neg = float(max(0.0, neg_mm))
    pos = float(max(0.0, pos_mm))
    step = float(max(1e-6, step_mm))

    if (neg > 0) or (pos > 0):
        offs_neg = np.arange(-neg, 0.0, step) if neg > 0 else np.array([], np.float64)
        offs_pos = np.arange(step, pos + 1e-9, step) if pos > 0 else np.array([], np.float64)
        offsets = np.concatenate([offs_neg, np.array([0.0]), offs_pos], 0)
    else:
        offsets = np.array([0.0], dtype=np.float64)

    avg_tc = np.zeros(T, np.float64)
    denom = np.zeros(T, np.float64)

    for d in offsets:
        P = P_all + N_all * d
        ijk = apply_affine_np(T_surface_to_ijk, P)

        for t in range(T):
            vals, valid = sample_trilinear_with_mask(vol[..., t], ijk)
            good = valid & np.isfinite(vals)

            if np.any(good):
                avg_tc[t] += float(np.mean(vals[good]))
                denom[t] += 1.0

    nonzero = denom > 0

    if np.any(nonzero):
        avg_tc[nonzero] /= denom[nonzero]

    t_axis = np.arange(T, dtype=np.float64) * dt
    valid_frac = float(np.mean(nonzero)) if T > 0 else 0.0

    return t_axis, avg_tc, n_sel, valid_frac


def _compute_avg_timecourse_4d_from_points(verts_local, norms_local, fs_ref_path, nifti_path,
                                           surface_space="FS_TKRAS",
                                           neg_mm=0.0, pos_mm=0.0, step_mm=0.5):
    nib = get_nib()

    img = nib.load(abspath(nifti_path))
    vol = img.get_fdata(dtype=np.float64)

    if img.ndim != 4:
        raise RuntimeError("Selected NIfTI is not 4D.")

    T = vol.shape[3]
    zooms = img.header.get_zooms()
    dt = float(zooms[3]) if len(zooms) > 3 and zooms[3] else 1.0

    T_surface_to_ijk = build_surface_to_target_vox(
        surface_space,
        fs_ref_path,
        img,
    )

    neg = float(max(0.0, neg_mm))
    pos = float(max(0.0, pos_mm))
    step = float(max(1e-6, step_mm))

    if (neg > 0) or (pos > 0):
        offs_neg = np.arange(-neg, 0.0, step) if neg > 0 else np.array([], np.float64)
        offs_pos = np.arange(step, pos + 1e-9, step) if pos > 0 else np.array([], np.float64)
        offsets = np.concatenate([offs_neg, np.array([0.0]), offs_pos], 0)
    else:
        offsets = np.array([0.0], dtype=np.float64)

    n_sel = int(verts_local.shape[0])

    avg_tc = np.zeros(T, np.float64)
    denom = np.zeros(T, np.float64)

    for d in offsets:
        P = verts_local + norms_local * d
        ijk = apply_affine_np(T_surface_to_ijk, P)

        for t in range(T):
            vals, valid = sample_trilinear_with_mask(vol[..., t], ijk)
            good = valid & np.isfinite(vals)

            if np.any(good):
                avg_tc[t] += float(np.mean(vals[good]))
                denom[t] += 1.0

    nonzero = denom > 0

    if np.any(nonzero):
        avg_tc[nonzero] /= denom[nonzero]

    t_axis = np.arange(T, dtype=np.float64) * dt
    valid_frac = float(np.mean(nonzero)) if T > 0 else 0.0

    return t_axis, avg_tc, n_sel, valid_frac


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

_LINE_THICKNESS = 8


def _plot_pixel_block(img, x, y, color, radius):
    H, W, _ = img.shape

    xi0 = max(0, x - radius)
    xi1 = min(W - 1, x + radius)
    yi0 = max(0, y - radius)
    yi1 = min(H - 1, y + radius)

    img[yi0:yi1 + 1, xi0:xi1 + 1, :] = color


def _draw_line(img, x0, y0, x1, y1, color, thickness=_LINE_THICKNESS):
    """Integer line with square brush thickness."""
    H, W, _ = img.shape

    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx + dy
    radius = max(1, int(thickness // 2))

    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            _plot_pixel_block(img, x0, y0, color, radius)

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 >= dy:
            err += dy
            x0 += sx

        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_polyline(img, xs, ys, color, thickness=_LINE_THICKNESS):
    for i in range(len(xs) - 1):
        _draw_line(
            img,
            xs[i],
            ys[i],
            xs[i + 1],
            ys[i + 1],
            color,
            thickness=thickness,
        )


def _get_or_make_plot_image(name, W, H):
    """Return an Image with the exact W×H size, scaling or recreating if needed."""
    img = bpy.data.images.get(name)

    if img is None:
        return bpy.data.images.new(name=name, width=W, height=H, alpha=True)

    try:
        iw, ih = int(img.size[0]), int(img.size[1])
    except Exception:
        iw, ih = int(getattr(img, "width", 0)), int(getattr(img, "height", 0))

    if iw != W or ih != H:
        try:
            img.scale(W, H)
        except Exception:
            name_old = img.name
            bpy.data.images.remove(img)
            img = bpy.data.images.new(name=name_old, width=W, height=H, alpha=True)

    return img


def _plot_to_image(name, t, y, *, ylab=None, title=None):
    W, H = 2048, 600
    pad = 20

    x0, y0 = pad, pad
    x1, y1 = W - pad, H - pad

    bg = np.zeros((H, W, 4), dtype=np.uint8)
    bg[..., 3] = 255

    _draw_line(bg, x0, y0, x1, y0, (255, 255, 255, 255))
    _draw_line(bg, x1, y0, x1, y1, (255, 255, 255, 255))
    _draw_line(bg, x1, y1, x0, y1, (255, 255, 255, 255))
    _draw_line(bg, x0, y1, x0, y0, (255, 255, 255, 255))

    if isinstance(t, np.ndarray) and isinstance(y, np.ndarray) and t.size > 1 and y.size == t.size:
        mask = np.isfinite(t) & np.isfinite(y)

        if np.any(mask):
            t = t[mask]
            y = y[mask]

            if t.size > 1:
                t0, t1 = float(t[0]), float(t[-1])
                ymin, ymax = float(np.min(y)), float(np.max(y))

                if ymax <= ymin:
                    ymax = ymin + 1e-6

                X = x0 + (t - t0) * (x1 - x0) / max(1e-12, (t1 - t0))
                Y = y1 - (y - ymin) * (y1 - y0) / (ymax - ymin)

                _draw_polyline(bg, X, Y, (255, 255, 255, 255))

    img = _get_or_make_plot_image(name, W, H)

    pixels = (np.flipud(bg).astype(np.float32) / 255.0).ravel()

    img.pixels.foreach_set(pixels)
    img.update()

    return img


def _show_image_in_any_image_editor(img):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for space in area.spaces:
                    if space.type == 'IMAGE_EDITOR':
                        space.image = img
                        return True

    return False


# -------------------------------------------------------------------
# Gizmo lock
# -------------------------------------------------------------------

_gizmo_prev = {}


def _set_tool_select(win, area):
    """Force Selection tool so transform widgets will not show."""
    try:
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)

        if region:
            bpy.ops.wm.tool_set_by_id(
                {'window': win, 'area': area, 'region': region},
                name="builtin.select_box",
            )

    except Exception:
        pass


def _apply_gizmo_lock(enabled: bool):
    wm = bpy.context.window_manager

    for win in wm.windows:
        for area in win.screen.areas:
            if area.type != 'VIEW_3D':
                continue

            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue

                key = (win.as_pointer(), area.as_pointer())

                if enabled:
                    if key not in _gizmo_prev:
                        _gizmo_prev[key] = {
                            "tool": bool(getattr(space, "show_gizmo_tool", True)),
                            "tr": getattr(space, "show_gizmo_object_translate", None),
                            "rot": getattr(space, "show_gizmo_object_rotate", None),
                        }

                    if hasattr(space, "show_gizmo_tool"):
                        space.show_gizmo_tool = False

                    if hasattr(space, "show_gizmo_object_translate"):
                        space.show_gizmo_object_translate = False

                    if hasattr(space, "show_gizmo_object_rotate"):
                        space.show_gizmo_object_rotate = False

                    _set_tool_select(win, area)

                else:
                    prev = _gizmo_prev.pop(key, None)

                    if prev:
                        if hasattr(space, "show_gizmo_tool"):
                            space.show_gizmo_tool = prev["tool"]

                        if prev["tr"] is not None and hasattr(space, "show_gizmo_object_translate"):
                            space.show_gizmo_object_translate = bool(prev["tr"])

                        if prev["rot"] is not None and hasattr(space, "show_gizmo_object_rotate"):
                            space.show_gizmo_object_rotate = bool(prev["rot"])

                    else:
                        if hasattr(space, "show_gizmo_tool"):
                            space.show_gizmo_tool = True

                        if hasattr(space, "show_gizmo_object_translate"):
                            space.show_gizmo_object_translate = True

                        if hasattr(space, "show_gizmo_object_rotate"):
                            space.show_gizmo_object_rotate = True


def _on_lock_edit_transforms(self, context):
    _apply_gizmo_lock(bool(self.lock_edit_transforms))


# -------------------------------------------------------------------
# Properties
# -------------------------------------------------------------------

PALETTE_ITEMS = [
    ('BLUERED',      "Blue–Red (diverging)", ""),
    ('GREYS',        "Greyscale", ""),
    ('JET',          "Jet", ""),
    ('COOL',         "Cool", ""),
    ('HOT',          "Hot", ""),
    ('POLAR',        "Polar (BV)", ""),
    ('ECCENTRICITY', "Eccentricity (BV)", ""),
]

SURFACE_SPACE_ITEMS = [
    (
        'FS_TKRAS',
        "FreeSurfer tkRAS",
        "Use this for recon-all surfaces such as lh.white, rh.white, lh.pial, inflated, etc.",
    ),
    (
        'SCANNER_RAS',
        "Scanner/NIfTI RAS",
        "Use this for meshes whose vertices are already in NIfTI/world/scanner RAS millimetres, e.g. many Workbench/GIFTI/OBJ exports.",
    ),
]


def _on_nifti_path_update(self, context):
    """Detect 3D vs 4D and set TR + N volumes for 4D display."""
    p = self

    p.is_4d_detected = False
    p.tc_tr = 1.0
    p.tc_n_vols = 0

    path = abspath(p.nifti_path)

    if not path or not os.path.isfile(path):
        return

    try:
        nib = get_nib()
        img = nib.load(path)
        nd = img.ndim

        p.is_4d_detected = bool(nd == 4)

        if p.is_4d_detected:
            z = img.header.get_zooms()

            if len(z) > 3 and z[3]:
                p.tc_tr = float(z[3])

            shape = img.shape
            p.tc_n_vols = int(shape[3]) if len(shape) > 3 else 0

    except Exception:
        pass


class NIFTI_PaintProps(bpy.types.PropertyGroup):
    # Painting
    nifti_path: bpy.props.StringProperty(
        name="NIfTI",
        subtype='FILE_PATH',
        default="",
        update=_on_nifti_path_update,
    )

    fs_ref_path: bpy.props.StringProperty(
        name="FS ref (nu.mgz)",
        subtype='FILE_PATH',
        default="",
    )

    surface_space: bpy.props.EnumProperty(
        name="Surface coordinate space",
        items=SURFACE_SPACE_ITEMS,
        default='FS_TKRAS',
        description="Coordinate system used by the selected mesh vertices",
    )

    attribute_name: bpy.props.StringProperty(
        name="Colour attribute base",
        default="Col",
        description="Base name for paint colour attributes. With unique-material mode on, the map name is appended automatically.",
    )

    unique_material_per_paint: bpy.props.BoolProperty(
        name="New material for each paint",
        description=(
            "Create a new colour attribute and a new material for every paint. "
            "This keeps previous maps switchable from the material slots."
        ),
        default=True,
    )

    last_paint_attribute_name: bpy.props.StringProperty(
        name="Last paint attribute",
        default="",
        options={'HIDDEN'},
    )

    last_paint_material_name: bpy.props.StringProperty(
        name="Last paint material",
        default="",
        options={'HIDDEN'},
    )

    palette: bpy.props.EnumProperty(
        name="Palette",
        items=PALETTE_ITEMS,
        default='BLUERED',
    )

    invert_palette: bpy.props.BoolProperty(
        name="Invert palette",
        default=False,
    )

    symmetric: bpy.props.BoolProperty(
        name="Symmetric ±range",
        default=True,
    )

    sample_neg_mm: bpy.props.FloatProperty(
        name="Below (mm)",
        default=0.0,
        min=0.0,
    )

    sample_pos_mm: bpy.props.FloatProperty(
        name="Above (mm)",
        default=0.0,
        min=0.0,
    )

    sample_step_mm: bpy.props.FloatProperty(
        name="Step (mm)",
        default=0.5,
        min=0.01,
    )

    sample_using_linked_white_normals: bpy.props.BoolProperty(
        name="Sample using linked WHITE normals",
        description="When painting/plotting, use normals and points from the linked WHITE mesh",
        default=False,
    )

    use_auto_range: bpy.props.BoolProperty(
        name="Auto range",
        default=True,
    )

    vmin: bpy.props.FloatProperty(
        name="Min",
        default=-1.0,
    )

    vmax: bpy.props.FloatProperty(
        name="Max",
        default=1.0,
    )

    p_lo: bpy.props.FloatProperty(
        name="Low %",
        default=2.0,
        min=0.0,
        max=50.0,
    )

    p_hi: bpy.props.FloatProperty(
        name="High %",
        default=98.0,
        min=50.0,
        max=100.0,
    )

    use_curvature_underlay: bpy.props.BoolProperty(
        name="Curvature underlay",
        default=True,
        update=_update_show_stats_map,
    )

    show_stats_map: bpy.props.BoolProperty(
        name="Show painted map",
        default=True,
        update=_update_show_stats_map,
    )

    # Curvature
    curv_p_hi: bpy.props.FloatProperty(
        name="Scale %",
        default=90.0,
        min=50.0,
        max=100.0,
    )

    curv_contrast_x: bpy.props.FloatProperty(
        name="Contrast",
        default=10.0,
        min=0.1,
        soft_max=20.0,
    )

    curv_trough_lum: bpy.props.FloatProperty(
        name="Trough",
        default=0.95,
        min=0.0,
        max=1.0,
    )

    curv_peak_lum: bpy.props.FloatProperty(
        name="Peak",
        default=0.05,
        min=0.0,
        max=1.0,
    )

    curv_smooth_after: bpy.props.BoolProperty(
        name="Smooth after paint",
        default=True,
    )

    curv_smooth_strength: bpy.props.FloatProperty(
        name="Smooth strength",
        default=0.5,
        min=0.0,
        max=1.0,
    )

    # Inflated link
    inflated_obj_path: bpy.props.StringProperty(
        name="Inflated OBJ",
        subtype='FILE_PATH',
        default="",
        description="Link once; colours auto-copy after paint/smooth",
    )

    # 4D detection + plotting
    is_4d_detected: bpy.props.BoolProperty(
        name="Detected: 4D",
        default=False,
        options={'HIDDEN'},
    )

    tc_tr: bpy.props.FloatProperty(
        name="TR (s)",
        default=1.0,
        min=0.0001,
        options={'HIDDEN'},
    )

    tc_n_vols: bpy.props.IntProperty(
        name="#Volumes",
        default=0,
        min=0,
        options={'HIDDEN'},
    )

    tc_y_mode: bpy.props.EnumProperty(
        name="Y axis",
        items=(
            ('BOLD', "BOLD (a.u.)", ""),
            ('PSC', "% signal change", ""),
        ),
        default='BOLD',
    )

    tc_use_linked_white_normals: bpy.props.BoolProperty(
        name="Sample using linked WHITE normals",
        default=True,
        description="When selected verts are on Inflated, sample at corresponding WHITE verts/normals",
    )

    last_plot_image_name: bpy.props.StringProperty(
        name="Last plot image",
        default="",
        options={'HIDDEN'},
    )

    # Gizmo lock toggle
    lock_edit_transforms: bpy.props.BoolProperty(
        name="Disable transforms in Edit Mode",
        description="Hides Move/Rotate gizmos in all 3D Views while ON",
        default=True,
        update=_on_lock_edit_transforms,
    )


# -------------------------------------------------------------------
# Operators
# -------------------------------------------------------------------

class NIFTI_OT_Paint_FS(bpy.types.Operator):
    bl_idname = "mesh.nifti_paint_fs"
    bl_label = "Paint NIfTI to vertex colours"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        p = ctx.scene.nifti_paint_props
        o = ctx.active_object

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        if not p.nifti_path:
            self.report({'ERROR'}, "Set a NIfTI path.")
            return {'CANCELLED'}

        # Fast path: paint directly on a NIfTI slice mesh.
        ijk_faces = _get_slice_face_ijk_from_mesh(o)
        meta = o.get("nifti_slice_meta") if o else None

        if ijk_faces is not None and meta:
            try:
                _ = _copy_face_color_attr(o.data, "SliceBase", "SliceCol")

                if not _scene_slice_matches_meta(meta):
                    self.report(
                        {'ERROR'},
                        f"Slice mesh is for {meta.get('orientation')} index {meta.get('slice_index')} "
                        f"but Scene shows {getattr(bpy.context.scene.nifti_slice_props, 'orientation', None)} "
                        f"index {getattr(bpy.context.scene.nifti_slice_props, 'slice_index', None)}. "
                        f"Click 'Build slice mesh' first.",
                    )
                    return {'CANCELLED'}

                nib = get_nib()

                stats_img = nib.load(abspath(p.nifti_path))
                stats = stats_img.get_fdata(dtype=np.float32)

                if stats.ndim == 4:
                    stats = stats[..., 0]

                base_path = abspath(meta.get("nifti_path", ""))
                base_img = stats_img if (not base_path or not os.path.isfile(base_path)) else nib.load(base_path)

                M = _vox_to_vox_mat(stats_img.affine, base_img.affine)

                ijk = ijk_faces.astype(np.float64)
                ones = np.ones((ijk.shape[0], 1), dtype=np.float64)
                dst_h = np.concatenate([ijk, ones], axis=1).T
                src_h = M @ dst_h

                x, y, z = src_h[0], src_h[1], src_h[2]

                if np.issubdtype(stats.dtype, np.floating):
                    vals = _sample_trilinear_3d(stats, x, y, z)
                else:
                    vals = _sample_nearest_3d(stats, x, y, z)

                if p.use_auto_range:
                    vmin_auto, vmax_auto = auto_vmin_vmax(
                        vals,
                        symmetric=p.symmetric,
                        p_lo=p.p_lo,
                        p_hi=p.p_hi,
                    )
                    vmin, vmax = vmin_auto, vmax_auto
                else:
                    vmin, vmax = float(p.vmin), float(p.vmax)

                if p.symmetric:
                    a = max(abs(vmin), abs(vmax))
                    vmin, vmax = -a, a

                if vmax <= vmin:
                    vmax = vmin + 1e-6

                rgba_face = map_vals_to_rgba(
                    vals,
                    vmin,
                    vmax,
                    palette=p.palette,
                    invert=p.invert_palette,
                )

                _write_face_rgba_slicecol(o.data, rgba_face)
                _apply_slice_material_rules_if_needed(o)

                self.report({'INFO'}, "Painted statistics onto NIfTI slice mesh.")
                return {'FINISHED'}

            except Exception as e:
                self.report({'ERROR'}, f"Slice paint failed: {e}")
                return {'CANCELLED'}

        # Standard cortical mesh path.
        if p.surface_space == 'FS_TKRAS' and not p.fs_ref_path:
            self.report({'ERROR'}, "Set FS ref (nu.mgz) when using FreeSurfer tkRAS surface space.")
            return {'CANCELLED'}

        try:
            vmin = vmax = None

            if not p.use_auto_range:
                vmin, vmax = p.vmin, p.vmax

            paint_attr_name, paint_mat_name = paint_vertex_colours_fs(
                o,
                p.nifti_path,
                p.fs_ref_path,
                attr_name=p.attribute_name,
                surface_space=p.surface_space,
                symmetric=p.symmetric,
                vmin=vmin,
                vmax=vmax,
                p_lo=p.p_lo,
                p_hi=p.p_hi,
                palette=p.palette,
                invert=p.invert_palette,
                sample_neg_mm=p.sample_neg_mm,
                sample_pos_mm=p.sample_pos_mm,
                sample_step_mm=p.sample_step_mm,
                use_curvature_underlay=p.use_curvature_underlay,
                show_stats_map=p.show_stats_map,
                sample_using_linked_white_normals=p.sample_using_linked_white_normals,
                unique_material_per_paint=p.unique_material_per_paint,
            )

            p.last_paint_attribute_name = paint_attr_name
            p.last_paint_material_name = paint_mat_name

            infl = _get_linked_inflated(o)

            if infl:
                _copy_nifti_material_slots(o, infl)

            _stamp_paint_metadata(o, p)

        except Exception as e:
            self.report({'ERROR'}, f"{e}")
            return {'CANCELLED'}

        if p.last_paint_material_name:
            self.report(
                {'INFO'},
                f"Painted. Material: {p.last_paint_material_name}; attribute: {p.last_paint_attribute_name}",
            )
        else:
            self.report({'INFO'}, "Painted.")

        return {'FINISHED'}


class NIFTI_OT_PaintCurvature(bpy.types.Operator):
    bl_idname = "mesh.nifti_paint_curvature"
    bl_label = "Paint curvature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.active_object

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        if _is_slice_mesh(o):
            self.report({'WARNING'}, "Curvature underlay is disabled for slice meshes.")
            return {'CANCELLED'}

        sp = ctx.scene.nifti_paint_props

        try:
            paint_curvature(
                o,
                "Curv",
                sp.curv_p_hi,
                sp.curv_trough_lum,
                sp.curv_peak_lum,
                sp.curv_contrast_x,
            )

            if sp.curv_smooth_after and sp.curv_smooth_strength > 0.0:
                smooth_vertex_colour_step(
                    o,
                    attr_name="Curv",
                    lam=float(sp.curv_smooth_strength),
                )

            infl = _get_linked_inflated(o)

            if infl:
                _copy_attr_by_index_or_nearest(o, infl, attr_name="Curv")

            for obj_ in (o, infl):
                if obj_:
                    _rebuild_nifti_materials_for_obj(obj_, sp)

        except Exception as e:
            self.report({'ERROR'}, f"{e}")
            return {'CANCELLED'}

        self.report({'INFO'}, "Curvature painted.")
        return {'FINISHED'}


class NIFTI_OT_SmoothCurvature(bpy.types.Operator):
    bl_idname = "mesh.nifti_smooth_curvature_step"
    bl_label = "Smooth curvature (1 step)"
    bl_options = {'REGISTER', 'UNDO'}

    strength: bpy.props.FloatProperty(
        name="Strength",
        default=0.5,
        min=0.0,
        max=1.0,
    )

    def execute(self, ctx):
        o = ctx.active_object

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        if _is_slice_mesh(o):
            self.report({'WARNING'}, "Curvature layer not applicable to slice meshes.")
            return {'CANCELLED'}

        try:
            smooth_vertex_colour_step(
                o,
                attr_name="Curv",
                lam=self.strength,
            )

            infl = _get_linked_inflated(o)

            if infl:
                _copy_attr_by_index_or_nearest(o, infl, attr_name="Curv")

            sp = ctx.scene.nifti_paint_props

            for obj_ in (o, infl):
                if obj_:
                    _rebuild_nifti_materials_for_obj(obj_, sp)

        except Exception as e:
            self.report({'ERROR'}, f"{e}")
            return {'CANCELLED'}

        self.report({'INFO'}, "Smoothed curvature by one step.")
        return {'FINISHED'}


class NIFTI_OT_RemoveCurvature(bpy.types.Operator):
    bl_idname = "mesh.nifti_remove_curvature_layer"
    bl_label = "Remove curvature layer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.active_object

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        me = o.data

        if hasattr(me, "color_attributes"):
            ca = me.color_attributes.get("Curv")

            if ca:
                me.color_attributes.remove(ca)

        else:
            layer = me.vertex_colors.get("Curv") if getattr(me, "vertex_colors", None) else None

            if layer:
                me.vertex_colors.remove(layer)

        try:
            sp = ctx.scene.nifti_paint_props
            _rebuild_nifti_materials_for_obj(o, sp)

            infl = _get_linked_inflated(o)

            if infl:
                _rebuild_nifti_materials_for_obj(infl, sp)

        except Exception:
            pass

        self.report({'INFO'}, "Curvature layer removed.")
        return {'FINISHED'}


class NIFTI_OT_LinkInflatedAndCopy(bpy.types.Operator):
    bl_idname = "mesh.nifti_link_inflated_and_copy"
    bl_label = "Link Inflated"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        base = ctx.active_object

        if not base or base.type != 'MESH':
            self.report({'ERROR'}, "Select your WHITE mesh first.")
            return {'CANCELLED'}

        if _is_slice_mesh(base):
            self.report({'WARNING'}, "Link Inflated is for cortical surfaces, not slice meshes.")
            return {'CANCELLED'}

        p = ctx.scene.nifti_paint_props
        path = abspath(p.inflated_obj_path)

        if not path:
            self.report({'ERROR'}, "Choose an Inflated OBJ path.")
            return {'CANCELLED'}

        try:
            infl, created = link_inflated_once(base, path)

            _copy_nifti_material_slots(base, infl)

            for obj_ in (base, infl):
                if obj_:
                    _rebuild_nifti_materials_for_obj(obj_, p)

        except Exception as e:
            self.report({'ERROR'}, f"{e}")
            return {'CANCELLED'}

        self.report({'INFO'}, "Linked Inflated (new)." if created else "Inflated already linked; updated.")
        return {'FINISHED'}


class NIFTI_OT_UnlinkInflated(bpy.types.Operator):
    bl_idname = "mesh.nifti_unlink_inflated"
    bl_label = "Unlink Inflated"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        base = ctx.active_object

        if not base or base.type != 'MESH':
            self.report({'ERROR'}, "Select your WHITE mesh first.")
            return {'CANCELLED'}

        if _is_slice_mesh(base):
            self.report({'WARNING'}, "No linked Inflated expected for slice meshes.")
            return {'CANCELLED'}

        infl = _get_linked_inflated(base)

        if not infl:
            self.report({'WARNING'}, "No linked Inflated found.")
            return {'CANCELLED'}

        try:
            del base["inflated_link_name"]
        except Exception:
            pass

        try:
            del infl["linked_white_name"]
        except Exception:
            pass

        infl.parent = None

        self.report({'INFO'}, f"Unlinked Inflated: {infl.name}")
        return {'FINISHED'}


class NIFTI_OT_ShowActivePaintMaterial(bpy.types.Operator):
    bl_idname = "mesh.nifti_show_active_paint_material"
    bl_label = "Show active paint material"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.active_object

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        if not o.data.materials:
            self.report({'ERROR'}, "The selected mesh has no material slots.")
            return {'CANCELLED'}

        slot = int(o.active_material_index)
        mat = o.data.materials[slot]

        if not mat:
            self.report({'ERROR'}, "The active material slot is empty.")
            return {'CANCELLED'}

        _assign_material_to_all_faces(o, slot)

        infl = _get_linked_inflated(o)

        if infl:
            target_slot = None

            for i, m in enumerate(infl.data.materials):
                if m == mat:
                    target_slot = i
                    break

            if target_slot is None:
                target_slot = _ensure_material_slot(infl, mat)

            _assign_material_to_all_faces(infl, target_slot)

        self.report({'INFO'}, f"Showing material: {mat.name}")

        return {'FINISHED'}


class NIFTI_OT_PlotTimecourseToImage(bpy.types.Operator):
    bl_idname = "mesh.nifti_plot_timecourse_image"
    bl_label = "Plot avg timecourse (Image Editor)"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        p = getattr(ctx.scene, "nifti_paint_props", None)

        return (
            o is not None and
            o.type == 'MESH' and
            ctx.mode == 'EDIT_MESH' and
            p and
            p.is_4d_detected and
            p.nifti_path and
            (p.surface_space == 'SCANNER_RAS' or p.fs_ref_path)
        )

    def execute(self, ctx):
        p = ctx.scene.nifti_paint_props
        obj = ctx.active_object

        win = ctx.window
        wm = ctx.window_manager

        try:
            if win:
                win.cursor_set("WAIT")
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            wm.progress_begin(0, 100)

        except Exception:
            pass

        try:
            use_white = bool(p.tc_use_linked_white_normals)

            if use_white:
                white = _get_linked_white(obj)

                if not white:
                    self.report({'WARNING'}, "No linked WHITE mesh found; sampling with active mesh normals.")
                    use_white = False

            if use_white:
                P_w, N_w, ids, _ = _selection_in_white_tkras(obj, white)

                if not len(ids):
                    self.report({'ERROR'}, "No vertices selected in Edit Mode.")
                    return {'CANCELLED'}

                t, y, n_sel, _ = _compute_avg_timecourse_4d_from_points(
                    P_w,
                    N_w,
                    p.fs_ref_path,
                    p.nifti_path,
                    surface_space=p.surface_space,
                    neg_mm=p.sample_neg_mm,
                    pos_mm=p.sample_pos_mm,
                    step_mm=p.sample_step_mm,
                )

            else:
                t, y, n_sel, _ = _compute_avg_timecourse_4d(
                    obj,
                    p.fs_ref_path,
                    p.nifti_path,
                    surface_space=p.surface_space,
                    use_linked_white_normals=False,
                    neg_mm=p.sample_neg_mm,
                    pos_mm=p.sample_pos_mm,
                    step_mm=p.sample_step_mm,
                )

            if y.size == 0:
                self.report({'ERROR'}, "No selection or no valid samples.")
                return {'CANCELLED'}

            if p.tc_y_mode == 'PSC':
                m = np.mean(y) if np.isfinite(y).any() else 1.0

                if abs(m) < 1e-8:
                    m = 1.0

                y = (y / m - 1.0) * 100.0

            img = _plot_to_image("Selection timecourse", t, y)
            _ = _show_image_in_any_image_editor(img)

            p.last_plot_image_name = img.name

            self.report(
                {'INFO'},
                f"Plotted to Image: {img.name} (n={n_sel}, TR={p.tc_tr:g}s, Vols={p.tc_n_vols})",
            )

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        finally:
            try:
                wm.progress_end()

                if win:
                    win.cursor_set("DEFAULT")

                ctx.workspace.status_text_set(None)

            except Exception:
                pass


# -------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------

class NIFTI_PT_Panel(bpy.types.Panel):
    bl_label = "NIfTI Paint"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NIfTI Paint'

    def draw(self, ctx):
        l = self.layout
        p = ctx.scene.nifti_paint_props

        obj = ctx.active_object if ctx.active_object and ctx.active_object.type == 'MESH' else None
        is_slice = _is_slice_mesh(obj)

        l.prop(p, "nifti_path")
        l.prop(p, "surface_space")

        row = l.row()
        row.enabled = (p.surface_space == 'FS_TKRAS')
        row.prop(p, "fs_ref_path")

        l.prop(p, "attribute_name")
        l.prop(p, "unique_material_per_paint")
        l.prop(p, "palette")
        l.prop(p, "invert_palette")
        l.prop(p, "symmetric")

        box = l.box()
        box.label(text="Sample along normal")

        col = box.column(align=True)
        col.prop(p, "sample_neg_mm")
        col.prop(p, "sample_pos_mm")
        col.prop(p, "sample_step_mm")

        l.prop(p, "sample_using_linked_white_normals")

        col = l.column(align=True)
        col.prop(p, "use_auto_range")

        if p.use_auto_range:
            col.prop(p, "p_lo")
            col.prop(p, "p_hi")
        else:
            col.prop(p, "vmin")
            col.prop(p, "vmax")

        l.operator("mesh.nifti_paint_fs", icon='BRUSH_DATA')

        if obj and obj.data.materials:
            boxM = l.box()
            boxM.label(text="Paint materials")

            boxM.template_list(
                "MATERIAL_UL_matslots",
                "",
                obj,
                "material_slots",
                obj,
                "active_material_index",
                rows=4,
            )

            boxM.operator("mesh.nifti_show_active_paint_material", icon='MATERIAL')

        l.prop(p, "show_stats_map")
        l.prop(p, "use_curvature_underlay")

        l.separator()

        box = l.box()
        row_title = box.row()
        row_title.label(text="Curvature underlay")
        box.enabled = not is_slice

        box.operator("mesh.nifti_paint_curvature", icon='BRUSH_DATA')

        grid = box.column(align=True)

        row1 = grid.row(align=True)
        row1.prop(p, "curv_p_hi", text="Scale%")
        row1.prop(p, "curv_contrast_x", text="Contrast")

        row2 = grid.row(align=True)
        row2.prop(p, "curv_trough_lum", text="Trough")
        row2.prop(p, "curv_peak_lum", text="Peak")

        box.operator("mesh.nifti_smooth_curvature_step", text="Smooth 1-step", icon='SMOOTHCURVE')

        rowS = box.row(align=True)
        rowS.prop(p, "curv_smooth_after")
        rowS.prop(p, "curv_smooth_strength", text="Strength")

        l.separator()

        boxI = l.box()
        boxI.enabled = not is_slice
        boxI.label(text="Inflated surface (link once, then auto-updates)")
        boxI.prop(p, "inflated_obj_path", text="Inflated OBJ")

        row = boxI.row(align=True)
        row.operator("mesh.nifti_link_inflated_and_copy", text="Link Inflated", icon='LINKED')
        row.operator("mesh.nifti_unlink_inflated", text="Unlink", icon='X')

        if p.is_4d_detected:
            l.separator()

            box4 = l.box()
            box4.label(text="4D timecourse")

            row2 = box4.row(align=True)
            row2.enabled = False
            row2.label(text=f"TR: {p.tc_tr:g}s")
            row2.separator()
            row2.label(text=f"Vols: {p.tc_n_vols}")

            row3 = box4.row(align=True)
            row3.prop(p, "tc_y_mode", text="Y axis")

            box4.prop(p, "tc_use_linked_white_normals")
            box4.prop(p, "lock_edit_transforms", text="Disable transforms in Edit Mode")
            box4.operator("mesh.nifti_plot_timecourse_image", icon='IMAGE')


# -------------------------------------------------------------------
# Register
# -------------------------------------------------------------------

_classes = (
    NIFTI_PaintProps,
    NIFTI_OT_Paint_FS,
    NIFTI_OT_PaintCurvature,
    NIFTI_OT_SmoothCurvature,
    NIFTI_OT_RemoveCurvature,
    NIFTI_OT_LinkInflatedAndCopy,
    NIFTI_OT_UnlinkInflated,
    NIFTI_OT_ShowActivePaintMaterial,
    NIFTI_OT_PlotTimecourseToImage,
    NIFTI_PT_Panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.nifti_paint_props = bpy.props.PointerProperty(type=NIFTI_PaintProps)

    try:
        _apply_gizmo_lock(bpy.context.scene.nifti_paint_props.lock_edit_transforms)
    except Exception:
        pass


def unregister():
    try:
        _apply_gizmo_lock(False)
    except Exception:
        pass

    for c in reversed(_classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.nifti_paint_props