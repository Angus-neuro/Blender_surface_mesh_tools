# tab_vertex_select.py — Vertex Select + ROI export + Auto-ROIs from Peaks
#
# Updated for mixed surface coordinate workflows:
#
#   Surface coordinate space:
#       1) FreeSurfer tkRAS
#       2) Scanner/NIfTI RAS
#
# Key fix:
#   The ROI exporter can now combine:
#
#       selected inflated mesh vertex indices
#       → same indices on anatomical/reference mesh
#       → Scanner/NIfTI RAS voxel mapping
#
#   This is the mode needed for workflows such as:
#
#       mri_tessellate → GIFTI → Workbench smoothing/inflation → OBJ/surface import
#
# where inflated and uninflated surfaces preserve vertex order, but the surface
# coordinates are not FreeSurfer recon-all tkRAS coordinates.
#
# Recommended for Workbench-derived inflated surfaces:
#
#   Active mesh:
#       wm_extrainflated
#
#   Reference anatomical mesh:
#       wm_smoothed
#
#   Surface coordinate space:
#       Scanner/NIfTI RAS
#
#   Volume method:
#       Normal offsets
#
#   This maps the selected inflated vertex indices back onto wm_smoothed before
#   stamping the ROI into the target volume.
#
# Notes:
#   - WHITE→PIAL column fill is now available for both coordinate spaces, provided
#     the anatomical/white and pial meshes have matching vertex counts.
#   - Auto-ROIs from Peaks now respects the nifti_surface_space metadata stamped
#     by the updated NIfTI Paint script.

import bpy
import bmesh
import numpy as np
import heapq
import os

from mathutils import Vector
from mathutils.kdtree import KDTree

from .common import (
    abspath,
    get_nib,
    apply_affine_np,
    sample_trilinear_with_mask,
)

# -------------------------------------------------------------------
# Local helpers: mode, normals, selections
# -------------------------------------------------------------------

def ensure_edit_mode(obj):
    if bpy.context.object != obj:
        bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.mode_set(mode='EDIT')


def get_selected_vert_indices(obj):
    ensure_edit_mode(obj)
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    return [v.index for v in bm.verts if v.select]


def _safe_calc_normals(me):
    """Robust normals calculation across Blender versions."""
    if hasattr(me, "calc_normals"):
        me.calc_normals()
    elif hasattr(me, "calc_normals_split"):
        me.calc_normals_split()


def _mesh_local_vertices_and_normals(obj):
    me = obj.data
    _safe_calc_normals(me)

    verts = np.array([tuple(v.co) for v in me.vertices], dtype=np.float64)
    norms = np.array([tuple(v.normal) for v in me.vertices], dtype=np.float64)
    norms = norms / np.maximum(1e-12, np.linalg.norm(norms, axis=1, keepdims=True))

    return verts, norms


def _selected_local_vertices_and_normals(obj):
    """
    Return selected vertices/normals from obj local coordinate space.

    For imported neuroimaging surfaces, local mesh coordinates are usually the
    meaningful anatomical coordinates. Avoid moving/scaling/rotating the object
    in Blender before exporting ROIs.
    """
    idxs = get_selected_vert_indices(obj)
    if not idxs:
        raise RuntimeError("No vertices selected.")

    verts, norms = _mesh_local_vertices_and_normals(obj)
    return verts[idxs, :], norms[idxs, :], idxs


def _selected_world_vertices_and_normals(obj):
    """
    Return selected vertices/normals from obj world coordinate space.

    This is retained for diagnostics/backward compatibility, but the main ROI
    path below uses local coordinates so that it matches the NIfTI Paint script.
    """
    idxs = get_selected_vert_indices(obj)
    if not idxs:
        raise RuntimeError("No vertices selected.")

    me = obj.data
    _safe_calc_normals(me)

    MW = obj.matrix_world
    Rw = MW.to_3x3()

    pts = []
    nrm = []

    for i in idxs:
        p = MW @ me.vertices[i].co
        n = Rw @ me.vertices[i].normal
        pts.append((p.x, p.y, p.z))
        nrm.append((n.x, n.y, n.z))

    pts = np.asarray(pts, dtype=np.float64)
    nrm = np.asarray(nrm, dtype=np.float64)
    nrm = nrm / np.maximum(1e-12, np.linalg.norm(nrm, axis=1, keepdims=True))

    return pts, nrm, idxs


# -------------------------------------------------------------------
# Coordinate transforms
# -------------------------------------------------------------------

def build_tkras_to_target_vox(fs_ref_path, target_img):
    """
    scanner vox(target) <- tkRAS(ref) affine.

    Use this for FreeSurfer recon-all surfaces whose vertex coordinates are
    FreeSurfer surface/tkRAS coordinates.
    """
    nib = get_nib()
    ref = nib.load(abspath(fs_ref_path))

    M_tkr = ref.header.get_vox2ras_tkr()   # tkRAS <- vox(ref)
    A_ref = ref.header.get_vox2ras()       # scannerRAS <- vox(ref)
    A_tgt = target_img.affine              # scannerRAS <- vox(target)

    return np.linalg.inv(A_tgt) @ A_ref @ np.linalg.inv(M_tkr)


def build_surface_to_target_vox(surface_space, fs_ref_path, target_img):
    """
    Return a 4x4 matrix mapping surface coordinates into target_img voxel ijk.

    surface_space:
      FS_TKRAS:
          mesh vertices are FreeSurfer surface/tkRAS coordinates.

      SCANNER_RAS:
          mesh vertices are scanner/world/NIfTI RAS millimetres and can be
          mapped directly with inv(target_img.affine).
    """
    surface_space = str(surface_space or "FS_TKRAS").upper()

    if surface_space == "FS_TKRAS":
        if not fs_ref_path:
            raise RuntimeError("FS ref (nu.mgz) is required for FreeSurfer tkRAS mode.")
        return build_tkras_to_target_vox(fs_ref_path, target_img)

    if surface_space == "SCANNER_RAS":
        return np.linalg.inv(target_img.affine)

    raise RuntimeError(f"Unknown surface coordinate space: {surface_space}")


# -------------------------------------------------------------------
# Inflated/active selection -> anatomical/reference mesh mapping
# -------------------------------------------------------------------

def selection_indices_on_active(sel_obj):
    """
    Return selected vertex indices from the active/selected mesh.
    """
    idxs = get_selected_vert_indices(sel_obj)
    if not idxs:
        raise RuntimeError("No vertices selected.")
    return idxs


def selection_on_reference_by_index(sel_obj, reference_obj):
    """
    Map the current selection on sel_obj onto reference_obj by vertex index.

    This is the important Workbench-compatible path:

        selected indices on inflated mesh
        -> same indices on wm_smoothed/anatomical mesh

    It requires matching vertex counts. This is expected for Workbench smoothing
    and inflation if the surfaces are generated from the same mesh topology.
    """
    sel_idx = selection_indices_on_active(sel_obj)

    if reference_obj is None:
        return _selected_local_vertices_and_normals(sel_obj)

    if len(sel_obj.data.vertices) != len(reference_obj.data.vertices):
        raise RuntimeError(
            "Active mesh and reference anatomical mesh have different vertex counts. "
            "Index-based inflated→anatomical mapping is not safe. "
            "Use a matching-topology reference mesh or select directly on the anatomical mesh."
        )

    verts_ref, norms_ref = _mesh_local_vertices_and_normals(reference_obj)

    return verts_ref[sel_idx, :], norms_ref[sel_idx, :], sel_idx


def selection_white_indices(sel_obj, white_obj):
    """
    Return WHITE/anatomical vertex indices corresponding to the current selection.

    If vertex counts match, this uses direct index mapping.

    If they do not match, this raises rather than using nearest-neighbour in 3D.
    Nearest-neighbour is not anatomically meaningful for inflated surfaces because
    inflated and folded surfaces are intentionally far apart.
    """
    sel_idx = selection_indices_on_active(sel_obj)

    if white_obj is None:
        return sel_idx

    if len(sel_obj.data.vertices) != len(white_obj.data.vertices):
        raise RuntimeError(
            "Active mesh and reference mesh have different vertex counts. "
            "Cannot safely map inflated selection to anatomical surface by index."
        )

    return sel_idx


# -------------------------------------------------------------------
# Legacy-compatible selection helper
# -------------------------------------------------------------------

def selection_in_white_tkras(sel_obj, white_obj):
    """
    Return selected points/normals expressed in the reference WHITE/anatomical
    object's local space.

    Despite the historical name, this now simply means:
        active selected indices -> reference object local coords

    Whether those local coords are tkRAS or Scanner/NIfTI RAS is decided later by
    the surface coordinate space transform.
    """
    P, N, _ = selection_on_reference_by_index(sel_obj, white_obj)
    return P, N


# -------------------------------------------------------------------
# Selection -> NIfTI ROI
# -------------------------------------------------------------------

def roi_from_selection(sel_obj, fs_ref_mgz_path, target_nifti_path, out_path,
                       neg_mm, pos_mm, step_mm, *,
                       voxel_pad=0,
                       surface_space='FS_TKRAS',
                       reference_anatomical_obj=None,
                       volume_method='NORMAL',
                       pial_ref_obj=None):
    """
    Stamp a selected surface region into a NIfTI mask.

    Parameters
    ----------
    sel_obj:
        The currently selected/active mesh. This may be inflated.

    reference_anatomical_obj:
        Optional anatomical/folded reference mesh.

        If provided and vertex counts match, selected vertex indices from sel_obj
        are mapped to this reference mesh before volume stamping.

        This is required when selecting on an inflated Workbench mesh but exporting
        the ROI in folded anatomical volume space.

    surface_space:
        FS_TKRAS:
            reference/anatomical mesh coordinates are FreeSurfer tkRAS.
            Requires fs_ref_mgz_path.

        SCANNER_RAS:
            reference/anatomical mesh coordinates are scanner/NIfTI RAS mm.
            Uses inv(target_img.affine).

    volume_method:
        NORMAL:
            Stamp selected vertices plus offsets along normals.

        WHITE_PIAL:
            Fill along anatomical/white -> pial columns. Requires pial_ref_obj
            with identical vertex count to reference_anatomical_obj.
    """
    nib = get_nib()

    fs_ref_mgz_path = abspath(fs_ref_mgz_path)
    target_nifti_path = abspath(target_nifti_path)
    out_path = abspath(out_path)

    surface_space = str(surface_space or "FS_TKRAS").upper()
    volume_method = str(volume_method or "NORMAL").upper()

    tgt_img = nib.load(target_nifti_path)

    if len(tgt_img.shape) != 3:
        raise RuntimeError("Target NIfTI must be 3D.")

    vol_shape = tgt_img.shape
    mask = np.zeros(vol_shape, dtype=np.uint8)

    T_surface_to_ijk = build_surface_to_target_vox(surface_space, fs_ref_mgz_path, tgt_img)

    def to_ijk(points_surface):
        return apply_affine_np(T_surface_to_ijk, points_surface)

    def stamp_ijk(i, j, k):
        if 0 <= i < vol_shape[0] and 0 <= j < vol_shape[1] and 0 <= k < vol_shape[2]:
            mask[i, j, k] = 1

            if voxel_pad > 0:
                for di in range(-voxel_pad, voxel_pad + 1):
                    for dj in range(-voxel_pad, voxel_pad + 1):
                        for dk in range(-voxel_pad, voxel_pad + 1):
                            ii = i + di
                            jj = j + dj
                            kk = k + dk

                            if 0 <= ii < vol_shape[0] and 0 <= jj < vol_shape[1] and 0 <= kk < vol_shape[2]:
                                mask[ii, jj, kk] = 1

    neg = float(max(0.0, neg_mm))
    pos = float(max(0.0, pos_mm))
    step = float(max(1e-6, step_mm))

    # ----------------------------------------------------------------
    # Method 1: WHITE/anatomical -> PIAL column fill
    # ----------------------------------------------------------------
    if volume_method == 'WHITE_PIAL':
        if reference_anatomical_obj is None:
            raise RuntimeError("Pick a reference anatomical/WHITE mesh for WHITE→PIAL method.")

        if pial_ref_obj is None:
            raise RuntimeError("Pick a PIAL reference object for WHITE→PIAL method.")

        if len(reference_anatomical_obj.data.vertices) != len(pial_ref_obj.data.vertices):
            raise RuntimeError("Reference anatomical/WHITE and PIAL meshes must have identical vertex counts.")

        if len(sel_obj.data.vertices) != len(reference_anatomical_obj.data.vertices):
            raise RuntimeError(
                "Active mesh and reference anatomical/WHITE mesh must have identical vertex counts "
                "for selected-index WHITE→PIAL column filling."
            )

        white_idx = selection_white_indices(sel_obj, reference_anatomical_obj)
        if not white_idx:
            raise RuntimeError("No vertices selected.")

        Vw, _ = _mesh_local_vertices_and_normals(reference_anatomical_obj)
        Vp, _ = _mesh_local_vertices_and_normals(pial_ref_obj)

        Pw = Vw[white_idx, :]
        Pp = Vp[white_idx, :]

        vec = Pp - Pw
        L = np.linalg.norm(vec, axis=1)

        good = L > 1e-6
        if not np.any(good):
            raise RuntimeError("WHITE→PIAL vectors are degenerate for the current selection.")

        u = np.zeros_like(vec)
        u[good, :] = vec[good, :] / L[good, None]

        for ii in np.where(good)[0]:
            Li = float(L[ii])

            # Start below the anatomical/white point, pass through the cortical
            # column, and optionally extend past pial by pos_mm.
            s = np.arange(-neg, Li + pos + 1e-9, step, dtype=np.float64)

            P = Pw[ii][None, :] + u[ii][None, :] * s[:, None]
            ijk = to_ijk(P)

            for i, j, k in np.rint(ijk).astype(int):
                stamp_ijk(int(i), int(j), int(k))

    # ----------------------------------------------------------------
    # Method 2: normal offsets
    # ----------------------------------------------------------------
    else:
        verts_local, norms_local, _ = selection_on_reference_by_index(
            sel_obj,
            reference_anatomical_obj,
        )

        if (neg > 0) or (pos > 0):
            offs_neg = np.arange(-neg, 0.0, step) if neg > 0 else np.array([], dtype=np.float64)
            offs_pos = np.arange(step, pos + 1e-9, step) if pos > 0 else np.array([], dtype=np.float64)
            offsets = np.concatenate([offs_neg, np.array([0.0], dtype=np.float64), offs_pos], axis=0)
        else:
            offsets = np.array([0.0], dtype=np.float64)

        for d in offsets:
            P = verts_local + norms_local * d
            ijk = to_ijk(P)

            for i, j, k in np.rint(ijk).astype(int):
                stamp_ijk(int(i), int(j), int(k))

    nib.save(
        nib.Nifti1Image(mask, affine=tgt_img.affine, header=tgt_img.header),
        out_path,
    )


# -------------------------------------------------------------------
# Stats sampling for Peak ROIs
# -------------------------------------------------------------------

def compute_stats_values_from_metadata(white_obj):
    """
    Re-sample painted stats back onto the reference/anatomical mesh vertices.

    This now respects the metadata stamped by the updated NIfTI Paint script:

        nifti_surface_space = FS_TKRAS or SCANNER_RAS

    Without this, Auto-ROIs from Peaks would always assume FreeSurfer tkRAS and
    would be wrong for Workbench/scanner-RAS surfaces.
    """
    nib = get_nib()

    nifti = white_obj.get("nifti_paint_path", "")
    fsref = white_obj.get("fs_ref_path", "")
    surface_space = str(white_obj.get("nifti_surface_space", "FS_TKRAS")).upper()

    if not nifti:
        raise RuntimeError("No painted stats metadata found on mesh. Paint a stats map first.")

    if surface_space == "FS_TKRAS" and not fsref:
        raise RuntimeError("No FS reference metadata found on mesh for FreeSurfer tkRAS sampling.")

    neg = float(white_obj.get("nifti_sample_neg_mm", 0.0))
    pos = float(white_obj.get("nifti_sample_pos_mm", 0.0))
    step = float(white_obj.get("nifti_sample_step_mm", 0.5))
    step = float(max(1e-6, step))

    img_tgt = nib.load(abspath(nifti))
    vol = img_tgt.get_fdata().astype(np.float64, copy=False)

    if img_tgt.ndim == 4:
        vol = vol[..., 0]

    T = build_surface_to_target_vox(surface_space, abspath(fsref), img_tgt)

    verts, norms = _mesh_local_vertices_and_normals(white_obj)

    if (neg > 0.0) or (pos > 0.0):
        offs_neg = np.arange(-neg, 0.0, step) if neg > 0 else np.array([], dtype=np.float64)
        offs_pos = np.arange(step, pos + 1e-9, step) if pos > 0 else np.array([], dtype=np.float64)
        offsets = np.concatenate([offs_neg, np.array([0.0], dtype=np.float64), offs_pos], axis=0)

        acc = np.zeros(verts.shape[0], dtype=np.float64)
        cnt = np.zeros(verts.shape[0], dtype=np.int32)

        for d in offsets:
            P = verts + norms * d
            ijk = apply_affine_np(T, P)
            vals, valid = sample_trilinear_with_mask(vol, ijk)

            good = valid & np.isfinite(vals)
            acc[good] += vals[good]
            cnt[good] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0).astype(np.float32)

    else:
        ijk = apply_affine_np(T, verts)
        vals, _ = sample_trilinear_with_mask(vol, ijk)

    return np.asarray(vals, dtype=np.float32)


# -------------------------------------------------------------------
# Graph helpers and peak-based ROIs
# -------------------------------------------------------------------

def build_adjacency_with_lengths(me):
    n = len(me.vertices)
    adj = [[] for _ in range(n)]
    wts = [[] for _ in range(n)]

    for e in me.edges:
        i, j = e.vertices
        p0 = me.vertices[i].co
        p1 = me.vertices[j].co
        d = (p1 - p0).length

        adj[i].append(j)
        wts[i].append(d)

        adj[j].append(i)
        wts[j].append(d)

    return adj, wts


def find_local_maxima(vals, adj, strict_eps=1e-12, mask=None):
    n = len(vals)

    if mask is None:
        mask = np.ones(n, dtype=bool)

    peaks = []

    for i in range(n):
        if not mask[i]:
            continue

        vi = vals[i]
        better = False

        for nb in adj[i]:
            if mask[nb] and vi < vals[nb] - strict_eps:
                better = True
                break

        if not better:
            peaks.append(i)

    return peaks


def geodesic_grow(seed, adj, wts, allowed_mask, max_radius):
    assigned = set()
    pq = [(0.0, seed)]
    dist = {seed: 0.0}

    while pq:
        d, v = heapq.heappop(pq)

        if d > max_radius:
            continue

        if not allowed_mask[v]:
            continue

        if v in assigned:
            continue

        assigned.add(v)

        for nb, w in zip(adj[v], wts[v]):
            nd = d + float(w)

            if (nb not in dist or nd < dist[nb]) and nd <= max_radius:
                dist[nb] = nd
                heapq.heappush(pq, (nd, nb))

    return assigned


def _white_owner(obj):
    """
    Historical helper.

    If obj is an inflated object linked to a white/anatomical mesh by the NIfTI
    Paint tab, return that anatomical mesh. Otherwise return obj.

    For Workbench workflows, Auto-ROIs from Peaks should ideally be run with the
    anatomical/reference mesh active, or with a valid linked_white_name.
    """
    name = obj.get("linked_white_name", "")
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    return obj


# -------------------------------------------------------------------
# Properties
# -------------------------------------------------------------------

SURFACE_SPACE_ITEMS = [
    (
        'FS_TKRAS',
        "FreeSurfer tkRAS",
        "Use for recon-all surfaces whose local coordinates are FreeSurfer surface/tkRAS.",
    ),
    (
        'SCANNER_RAS',
        "Scanner/NIfTI RAS",
        "Use for Workbench/GIFTI/OBJ surfaces whose local coordinates are scanner/NIfTI RAS mm.",
    ),
]


class VertexSelectProps(bpy.types.PropertyGroup):
    circle_radius_px: bpy.props.IntProperty(
        name="Circle radius (px)",
        default=30,
        min=2,
        soft_max=300,
    )

    vertex_group_name: bpy.props.StringProperty(
        name="Vertex Group",
        default="ROI",
    )

    # ROI export
    roi_target_nifti: bpy.props.StringProperty(
        name="Target NIfTI",
        subtype='FILE_PATH',
        default="",
    )

    roi_output_path: bpy.props.StringProperty(
        name="Output NIfTI",
        subtype='FILE_PATH',
        default="",
    )

    roi_surface_space: bpy.props.EnumProperty(
        name="Surface coordinate space",
        items=SURFACE_SPACE_ITEMS,
        default='FS_TKRAS',
        description="Coordinate space of the reference anatomical mesh vertices.",
    )

    roi_reference_anatomical: bpy.props.PointerProperty(
        name="Reference anatomical mesh",
        type=bpy.types.Object,
        description=(
            "Folded anatomical/reference mesh. "
            "When selecting on inflated, selected vertex indices are mapped to this mesh before ROI export."
        ),
    )

    roi_volume_method: bpy.props.EnumProperty(
        name="Volume method",
        items=(
            (
                'NORMAL',
                "Normal offsets",
                "Stamp selected vertices plus Below/Above offsets along reference mesh normals.",
            ),
            (
                'WHITE_PIAL',
                "WHITE→PIAL column",
                "Fill along reference anatomical/WHITE to PIAL columns. Requires matching vertex order.",
            ),
        ),
        default='NORMAL',
    )

    roi_pial_ref: bpy.props.PointerProperty(
        name="Reference PIAL mesh",
        type=bpy.types.Object,
        description="PIAL mesh with vertex count/order matching the reference anatomical/WHITE mesh.",
    )

    sample_neg_mm: bpy.props.FloatProperty(
        name="Below (mm)",
        default=0.0,
        min=0.0,
    )

    sample_pos_mm: bpy.props.FloatProperty(
        name="Above (mm)",
        default=3.0,
        min=0.0,
    )

    sample_step_mm: bpy.props.FloatProperty(
        name="Step (mm)",
        default=0.5,
        min=0.01,
    )

    voxel_pad: bpy.props.IntProperty(
        name="Voxel pad",
        default=0,
        min=0,
        soft_max=3,
    )

    # Auto-ROIs from Peaks
    peaks_group_prefix: bpy.props.StringProperty(
        name="Prefix",
        default="Peak",
    )

    peaks_use_abs: bpy.props.BoolProperty(
        name="Use |value|",
        default=True,
    )

    peaks_min_value: bpy.props.FloatProperty(
        name="Min value (thr)",
        default=0.2,
    )

    peaks_min_separation_mm: bpy.props.FloatProperty(
        name="Min peak separation (mm)",
        default=10.0,
        min=0.0,
    )

    peaks_radius_mm: bpy.props.FloatProperty(
        name="Grow radius (mm)",
        default=15.0,
        min=0.0,
    )

    peaks_max_count: bpy.props.IntProperty(
        name="Max peaks",
        default=12,
        min=1,
        soft_max=64,
    )

    peaks_clear_old: bpy.props.BoolProperty(
        name="Clear existing with prefix",
        default=True,
    )


# -------------------------------------------------------------------
# Operators
# -------------------------------------------------------------------

class VERTSEL_OT_activate_circle(bpy.types.Operator):
    bl_idname = "mesh.activate_circle_select_tool"
    bl_label = "Activate Circle Select (C)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.active_object

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        ensure_edit_mode(o)

        try:
            ctx.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.wm.tool_set_by_id(name="builtin.select_circle")

            tool = ctx.workspace.tools.from_space_view3d_mode(ctx.mode, create=False)
            if tool and tool.idname == "builtin.select_circle":
                op = tool.operator_properties("view3d.select_circle")
                op.radius = int(ctx.scene.vertex_select_props.circle_radius_px)

        except Exception:
            pass

        self.report({'INFO'}, "Circle Select active.")
        return {'FINISHED'}


class VERTSEL_OT_deactivate_circle(bpy.types.Operator):
    bl_idname = "mesh.deactivate_circle_select_tool"
    bl_label = "Deactivate Circle Select"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        try:
            bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
            self.report({'INFO'}, "Circle Select deactivated.")
        except Exception:
            self.report({'WARNING'}, "Could not switch tool; press Esc.")

        return {'FINISHED'}


class VERTSEL_OT_deselect_all(bpy.types.Operator):
    bl_idname = "mesh.vertex_deselect_all"
    bl_label = "Deselect All"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.active_object

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        ensure_edit_mode(o)
        bpy.ops.mesh.select_all(action='DESELECT')

        return {'FINISHED'}


class VERTSEL_OT_save_vgroup(bpy.types.Operator):
    bl_idname = "mesh.save_selection_to_vgroup"
    bl_label = "Save selection to Vertex Group"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.active_object
        p = ctx.scene.vertex_select_props

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        ensure_edit_mode(o)
        idxs = get_selected_vert_indices(o)

        if not idxs:
            self.report({'WARNING'}, "No vertices selected.")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')

        vg = o.vertex_groups.get(p.vertex_group_name) or o.vertex_groups.new(name=p.vertex_group_name)
        vg.add(idxs, 1.0, 'REPLACE')

        bpy.ops.object.mode_set(mode='EDIT')

        self.report({'INFO'}, f"Saved {len(idxs)} vertices to '{vg.name}'.")
        return {'FINISHED'}


class VERTSEL_OT_convert_to_roi(bpy.types.Operator):
    bl_idname = "mesh.selection_to_nifti_roi"
    bl_label = "Selection → ROI (NIfTI)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.active_object
        main = ctx.scene.nifti_paint_props if hasattr(ctx.scene, "nifti_paint_props") else None
        vp = ctx.scene.vertex_select_props

        if not o or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        fs_ref = ""
        if main is not None:
            fs_ref = abspath(main.fs_ref_path)

        surface_space = str(vp.roi_surface_space or "FS_TKRAS").upper()

        if surface_space == "FS_TKRAS":
            if not fs_ref or not os.path.isfile(fs_ref):
                self.report(
                    {'ERROR'},
                    "FS ref (nu.mgz) not set or not found. "
                    "Set it in the NIfTI Paint tab or use Scanner/NIfTI RAS mode.",
                )
                return {'CANCELLED'}

        target = abspath(vp.roi_target_nifti)
        out = abspath(vp.roi_output_path)

        if not target or not os.path.splitext(target)[1]:
            self.report({'ERROR'}, "Target NIfTI path is empty or invalid.")
            return {'CANCELLED'}

        if not os.path.isfile(target):
            self.report({'ERROR'}, f"Target NIfTI not found: {target}")
            return {'CANCELLED'}

        if not out or not os.path.splitext(out)[1]:
            self.report({'ERROR'}, "Output NIfTI path is empty or invalid.")
            return {'CANCELLED'}

        vol_method = str(vp.roi_volume_method or "NORMAL").upper()
        ref_obj = vp.roi_reference_anatomical
        pial_ref = vp.roi_pial_ref if vol_method == 'WHITE_PIAL' else None

        if ref_obj is not None:
            if ref_obj.type != 'MESH':
                self.report({'ERROR'}, "Reference anatomical object must be a mesh.")
                return {'CANCELLED'}

            if len(o.data.vertices) != len(ref_obj.data.vertices):
                self.report(
                    {'ERROR'},
                    "Active mesh and reference anatomical mesh have different vertex counts. "
                    "Cannot safely map inflated selection to anatomical mesh by index.",
                )
                return {'CANCELLED'}

        if vol_method == 'WHITE_PIAL':
            if ref_obj is None:
                self.report({'ERROR'}, "Pick a reference anatomical/WHITE mesh for WHITE→PIAL method.")
                return {'CANCELLED'}

            if pial_ref is None:
                self.report({'ERROR'}, "Pick a reference PIAL mesh for WHITE→PIAL method.")
                return {'CANCELLED'}

            if pial_ref.type != 'MESH':
                self.report({'ERROR'}, "Reference PIAL object must be a mesh.")
                return {'CANCELLED'}

            if len(pial_ref.data.vertices) != len(ref_obj.data.vertices):
                self.report({'ERROR'}, "Reference anatomical/WHITE and PIAL meshes must have identical vertex counts.")
                return {'CANCELLED'}

        try:
            roi_from_selection(
                o,
                fs_ref,
                target,
                out,
                vp.sample_neg_mm,
                vp.sample_pos_mm,
                vp.sample_step_mm,
                voxel_pad=vp.voxel_pad,
                surface_space=surface_space,
                reference_anatomical_obj=ref_obj,
                volume_method=vol_method,
                pial_ref_obj=pial_ref,
            )

        except Exception as e:
            self.report({'ERROR'}, f"{e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"ROI saved: {out}")
        return {'FINISHED'}


class VERTSEL_OT_AutoPeakROIs(bpy.types.Operator):
    """
    Detect peaks in the painted stats on the anatomical/reference mesh and grow
    non-overlapping ROIs by geodesic radius.
    """
    bl_idname = "mesh.auto_peak_rois"
    bl_label = "Auto-ROIs from Peaks"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        obj_active = ctx.active_object

        if not obj_active or obj_active.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh.")
            return {'CANCELLED'}

        vp = ctx.scene.vertex_select_props

        try:
            white = _white_owner(obj_active)
            me = white.data

            vals = compute_stats_values_from_metadata(white)

            verts = np.array([tuple(v.co) for v in me.vertices], dtype=np.float64)
            adj, wts = build_adjacency_with_lengths(me)

            work_vals = np.abs(vals) if vp.peaks_use_abs else vals
            finite = np.isfinite(work_vals)
            mask = finite & (work_vals >= float(vp.peaks_min_value))

            cand = find_local_maxima(work_vals, adj, strict_eps=1e-12, mask=mask)

            if not cand:
                self.report({'WARNING'}, "No peaks found above threshold.")
                return {'CANCELLED'}

            cand.sort(key=lambda i: work_vals[i], reverse=True)

            kept = []
            for i in cand:
                if len(kept) >= int(vp.peaks_max_count):
                    break

                if all(np.linalg.norm(verts[i] - verts[j]) >= float(vp.peaks_min_separation_mm) for j in kept):
                    kept.append(i)

            if not kept:
                self.report({'WARNING'}, "All peaks suppressed by min separation.")
                return {'CANCELLED'}

            label = -np.ones(len(vals), dtype=np.int32)
            regions = []

            for pi, seed in enumerate(kept, start=1):
                allowed = mask & (label < 0)
                grown = geodesic_grow(seed, adj, wts, allowed, float(vp.peaks_radius_mm))

                if not grown:
                    continue

                for v in grown:
                    label[v] = pi

                regions.append((pi, sorted(grown)))

            if not regions:
                self.report({'WARNING'}, "No vertices added to any ROI.")
                return {'CANCELLED'}

            prefix = (vp.peaks_group_prefix or "Peak").strip()

            def clear_prefix(obj):
                if not vp.peaks_clear_old:
                    return

                for vg in list(obj.vertex_groups):
                    if vg.name.startswith(prefix):
                        obj.vertex_groups.remove(vg)

            infl_name = white.get("inflated_link_name", "")
            infl = bpy.data.objects.get(infl_name) if infl_name else None
            same = infl is not None and len(infl.data.vertices) == len(white.data.vertices)

            clear_prefix(white)
            if same:
                clear_prefix(infl)

            bpy.ops.object.mode_set(mode='OBJECT')

            for pi, idxs in regions:
                name = f"{prefix}_{pi:02d}"

                vg = white.vertex_groups.get(name) or white.vertex_groups.new(name=name)
                vg.add(idxs, 1.0, 'REPLACE')

                if same:
                    vg2 = infl.vertex_groups.get(name) or infl.vertex_groups.new(name=name)
                    vg2.add(idxs, 1.0, 'REPLACE')

            bpy.ops.object.mode_set(mode='EDIT')

            self.report({'INFO'}, f"Created {len(regions)} peak ROI group(s) with prefix '{prefix}'.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"{e}")
            return {'CANCELLED'}


# -------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------

class VERTSEL_PT_Panel(bpy.types.Panel):
    bl_label = "Vertex Select"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Vertex Select'

    def draw(self, ctx):
        l = self.layout
        p = ctx.scene.vertex_select_props

        box = l.box()
        box.label(text="Circle Select (built-in)")
        box.prop(p, "circle_radius_px")

        row = box.row(align=True)
        row.operator("mesh.activate_circle_select_tool", text="Activate (C)", icon='SELECT_SET')
        row.operator("mesh.deactivate_circle_select_tool", text="Deactivate", icon='CANCEL')
        row.operator("mesh.vertex_deselect_all", text="Deselect All", icon='PANEL_CLOSE')

        l.separator()

        boxP = l.box()
        boxP.label(text="Auto-ROIs from Peaks (painted stats)")

        col = boxP.column(align=True)
        col.prop(p, "peaks_group_prefix")
        col.prop(p, "peaks_use_abs")
        col.prop(p, "peaks_min_value")
        col.prop(p, "peaks_min_separation_mm")
        col.prop(p, "peaks_radius_mm")
        col.prop(p, "peaks_max_count")
        col.prop(p, "peaks_clear_old")

        boxP.operator("mesh.auto_peak_rois", icon='MOD_VERTEX_WEIGHT')

        l.separator()

        box2 = l.box()
        box2.label(text="Vertex group")
        box2.prop(p, "vertex_group_name")
        box2.operator("mesh.save_selection_to_vgroup", icon='GROUP_VERTEX')

        l.separator()

        box3 = l.box()
        box3.label(text="Selection → Volume ROI (NIfTI)")

        box3.prop(p, "roi_target_nifti")
        box3.prop(p, "roi_output_path")

        box3.separator()
        box3.prop(p, "roi_surface_space")
        box3.prop(p, "roi_reference_anatomical")

        row_info = box3.row()
        row_info.enabled = False

        if p.roi_reference_anatomical:
            row_info.label(text="Inflated selection maps to reference by vertex index.")
        else:
            row_info.label(text="No reference mesh: active selected mesh is used directly.")

        box3.separator()
        box3.prop(p, "roi_volume_method")

        if str(p.roi_volume_method).upper() == 'WHITE_PIAL':
            box3.prop(p, "roi_pial_ref")

        col = box3.column(align=True)
        col.prop(p, "sample_neg_mm")
        col.prop(p, "sample_pos_mm")
        col.prop(p, "sample_step_mm")
        col.prop(p, "voxel_pad")

        box3.operator("mesh.selection_to_nifti_roi", icon='OUTLINER_OB_VOLUME')


# -------------------------------------------------------------------
# Register
# -------------------------------------------------------------------

_classes = (
    VertexSelectProps,
    VERTSEL_OT_activate_circle,
    VERTSEL_OT_deactivate_circle,
    VERTSEL_OT_deselect_all,
    VERTSEL_OT_save_vgroup,
    VERTSEL_OT_convert_to_roi,
    VERTSEL_OT_AutoPeakROIs,
    VERTSEL_PT_Panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.vertex_select_props = bpy.props.PointerProperty(type=VertexSelectProps)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.vertex_select_props