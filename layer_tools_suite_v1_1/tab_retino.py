# tab_retino_v123.py  —  V1/V2/V3 fitter (hue bands → clusters → centerlines → V2 midline)
# Writes vertex groups (with _d or _v suffix): V13_candidates*, V1_cluster*, V3_cluster*, V1_line*, V3_line*, V2_line*

import bpy, heapq
import numpy as np
from collections import deque
from mathutils.kdtree import KDTree

# Set the panel name in the N-panel sidebar
PANEL_CATEGORY = "Retino"      # change to "NIfTI Paint" if you want it under the same tab as your painter

# ------------------------- color access & mapping -------------------------

def _find_point_attr(me, preferred=None):
    ca = getattr(me, "color_attributes", None)
    if ca:
        if preferred:
            lay = ca.get(preferred)
            if lay and lay.domain == 'POINT':
                return preferred
        act = getattr(ca, "active_color", None)
        if act and act.domain == 'POINT':
            return act.name
        for lay in ca:
            if lay.domain == 'POINT':
                return lay.name
        return None
    vc = getattr(me, "vertex_colors", None)
    if vc and len(vc) > 0:
        return vc.active.name if vc.active else vc[0].name
    return None

def _get_rgba_point(me, name):
    ca = getattr(me, "color_attributes", None)
    if ca:
        layer = ca.get(name)
        if not layer or layer.domain != 'POINT':
            return None
        n = min(len(layer.data), len(me.vertices))
        out = np.zeros((len(me.vertices), 4), np.float32)
        for i in range(n):
            out[i] = layer.data[i].color[:]
        return out
    vc = getattr(me, "vertex_colors", None)
    if not vc:
        return None
    layer = vc.get(name)
    if not layer:
        return None
    sums = np.zeros((len(me.vertices), 4), np.float64)
    cnt  = np.zeros(len(me.vertices), np.int32)
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            r,g,b,a = layer.data[li].color
            sums[vi] += (r,g,b,a)
            cnt[vi]  += 1
    cnt = np.maximum(cnt, 1)
    return (sums / cnt[:, None]).astype(np.float32)

def _kd_map_rgba_world(src_obj, tgt_obj, src_attr):
    src_rgba = _get_rgba_point(src_obj.data, src_attr)
    if src_rgba is None:
        return None
    pts = [src_obj.matrix_world @ v.co for v in src_obj.data.vertices]
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    out = np.zeros((len(tgt_obj.data.vertices), 4), np.float32)
    for i, v in enumerate(tgt_obj.data.vertices):
        _, j, _ = kd.find(tgt_obj.matrix_world @ v.co)
        out[i] = src_rgba[j]
    return out

def _get_rgba_for_object(obj, preferred="", allow_linked_white=True):
    me = obj.data
    attr = _find_point_attr(me, preferred or None)
    rgba = _get_rgba_point(me, attr) if attr else None
    if rgba is not None:
        return rgba, attr, obj
    if allow_linked_white:
        # your “Link Inflated” tool sets this when you attach Inflated to WHITE
        wname = obj.get("linked_white_name", "")
        if wname and wname in bpy.data.objects:
            white = bpy.data.objects[wname]
            a2 = _find_point_attr(white.data, preferred or None)
            if a2:
                mapped = _kd_map_rgba_world(white, obj, a2)
                if mapped is not None:
                    return mapped, f"{a2} (mapped from {white.name})", white
    return None, None, None

# ------------------------- hue / adjacency / gradients -------------------------

def _rgb_to_hue_deg(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    c = maxc - minc
    c_safe = np.where(c > 1e-12, c, 1.0)
    hr = np.where(maxc == r, (g - b) / c_safe, 0.0)
    hg = np.where(maxc == g, (b - r) / c_safe + 2.0, hr)
    hb = np.where(maxc == b, (r - g) / c_safe + 4.0, hg)
    h6 = np.where(c > 1e-12, hb, 0.0)
    hue = (h6 / 6.0) % 1.0
    return (hue * 360.0).astype(np.float32)

def _rgb_to_hue_deg_sat(rgb):
    """Return hue in degrees and HSV saturation for each RGB triple."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    c = maxc - minc

    # Hue (same logic as _rgb_to_hue_deg)
    c_safe = np.where(c > 1e-12, c, 1.0)
    hr = np.where(maxc == r, (g - b) / c_safe, 0.0)
    hg = np.where(maxc == g, (b - r) / c_safe + 2.0, hr)
    hb = np.where(maxc == b, (r - g) / c_safe + 4.0, hg)
    h6 = np.where(c > 1e-12, hb, 0.0)
    hue = (h6 / 6.0) % 1.0
    hue_deg = (hue * 360.0).astype(np.float32)

    # HSV saturation: c / maxc (0 when maxc==0)
    sat = np.zeros_like(maxc, dtype=np.float32)
    nz = maxc > 1e-6
    sat[nz] = (c[nz] / maxc[nz]).astype(np.float32)
    return hue_deg, sat

def _circ_diff_deg(a, b):
    d = np.abs(a - b)
    return np.where(d > 180.0, 360.0 - d, d)

def _build_adj(obj, weighted=False):
    me = obj.data
    n = len(me.vertices)
    adj = [[] for _ in range(n)]
    wts = [[] for _ in range(n)] if weighted else None
    co  = np.array([(obj.matrix_world @ v.co)[:] for v in me.vertices], np.float64) if weighted else None
    for e in me.edges:
        i, j = e.vertices
        adj[i].append(j); adj[j].append(i)
        if weighted:
            d = float(np.linalg.norm(co[i] - co[j]))
            wts[i].append(d); wts[j].append(d)
    return adj, wts

def _build_adj_weights(obj):
    me = obj.data
    n = len(me.vertices)
    adj = [[] for _ in range(n)]
    wts = [[] for _ in range(n)]
    co  = np.array([(obj.matrix_world @ v.co)[:] for v in me.vertices], np.float64)
    for e in me.edges:
        i, j = e.vertices
        d = float(np.linalg.norm(co[i] - co[j]))
        adj[i].append(j); wts[i].append(d)
        adj[j].append(i); wts[j].append(d)
    return adj, wts

def _phase_grad_deg(adj, hue_deg):
    n = len(hue_deg)
    g = np.zeros(n, np.float32)
    for i in range(n):
        nbs = adj[i]
        if not nbs: continue
        di = _circ_diff_deg(hue_deg[i], hue_deg[nbs])
        g[i] = float(np.mean(di))
    return g

# ------------------------- morphology / components -------------------------

def _support_prune(mask, adj, min_support):
    if min_support <= 0:
        return mask
    m = mask.copy()
    idx = np.nonzero(m)[0]
    kill = []
    for i in idx:
        s = sum(1 for j in adj[i] if m[j])
        if s < min_support:
            kill.append(i)
    if kill:
        m[np.array(kill, dtype=np.int32)] = False
    return m

def _components(mask, adj, min_size):
    n = len(mask)
    seen = np.zeros(n, bool)
    out = []
    for i in range(n):
        if not mask[i] or seen[i]:
            continue
        q = deque([i]); seen[i] = True; cur = []
        while q:
            u = q.popleft(); cur.append(u)
            for v in adj[u]:
                if mask[v] and not seen[v]:
                    seen[v] = True; q.append(v)
        if len(cur) >= min_size:
            out.append(np.array(cur, np.int32))
    return out

# ------------------------- longest path / line -------------------------

def _two_bfs_longest_path(adj, nodes):
    S = set(int(x) for x in nodes)
    def bfs(start):
        parent = {start: -1}
        q = deque([start])
        last = start
        while q:
            u = q.popleft(); last = u
            for v in adj[u]:
                if v in S and v not in parent:
                    parent[v] = u; q.append(v)
        return last, parent
    a0 = int(nodes[0])
    a, _ = bfs(a0)
    b, parent = bfs(a)
    path = []
    cur = b
    while cur != -1:
        path.append(cur); cur = parent[cur]
    return path

def _mask_from_path(path, n, widen, adj):
    m = np.zeros(n, bool)
    for i in path:
        m[i] = True
    for _ in range(int(max(0, widen))):
        add = m.copy()
        idx = np.nonzero(m)[0]
        for i in idx:
            for j in adj[i]:
                add[j] = True
        m = add
    return m

# ------------------------- geodesic & Voronoi helpers -------------------------

def _dijkstra_multi(adj, wts, sources):
    n = len(adj); INF = 1e30
    dist = np.full(n, INF, np.float64)
    h = []
    for s in sources:
        s = int(s); dist[s] = 0.0; heapq.heappush(h, (0.0, s))
    while h:
        d, u = heapq.heappop(h)
        if d != dist[u]:
            continue
        for v, w in zip(adj[u], wts[u]):
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(h, (nd, v))
    return dist

def _geodesic_ball(adj, wts, seeds, radius_mm):
    """Vertices within geodesic radius_mm of any seed."""
    n = len(adj); INF = 1e30
    dist = np.full(n, INF, np.float64)
    h = []
    for s in seeds:
        s = int(s); dist[s] = 0.0; heapq.heappush(h, (0.0, s))
    while h:
        d, u = heapq.heappop(h)
        if d != dist[u] or d > radius_mm:
            continue
        for v, w in zip(adj[u], wts[u]):
            nd = d + float(w)
            if nd < dist[v] and nd <= radius_mm:
                dist[v] = nd; heapq.heappush(h, (nd, v))
    return dist <= radius_mm

def _dijkstra_multi_masked(adj, wts, sources, mask):
    """Geodesic distance constrained to vertices where mask==True."""
    n = len(adj); INF = 1e30
    dist = np.full(n, INF, np.float64)
    h = []
    for s in sources:
        s = int(s)
        if not mask[s]:
            continue
        dist[s] = 0.0; heapq.heappush(h, (0.0, s))
    while h:
        d, u = heapq.heappop(h)
        if d != dist[u]:
            continue
        for v, w in zip(adj[u], wts[u]):
            if not mask[v]:
                continue
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(h, (nd, v))
    return dist

def _boundary_from_labels(adj, band_mask, label):
    """
    band_mask: candidate vertices; label[i] in {1,3} for 'closer to V1'/'closer to V3'.
    Return vertices that sit on the label-change boundary inside band_mask.
    """
    n = len(band_mask)
    out = np.zeros(n, bool)
    idx = np.nonzero(band_mask)[0]
    for i in idx:
        li = label[i]
        if li == 0:
            continue
        if any(band_mask[j] and label[j] != li for j in adj[i]):
            out[i] = True
    return out

# ------------------------- I/O: vertex groups -------------------------

def _write_group(obj, name, mask):
    vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    vg.remove(range(len(obj.data.vertices)))
    if mask is None:
        return 0
    idx = np.nonzero(mask)[0]
    if len(idx):
        vg.add(idx.tolist(), 1.0, 'REPLACE')
    return len(idx)

def _read_group_mask(obj, name):
    vg = obj.vertex_groups.get(name)
    n  = len(obj.data.vertices)
    if not vg:
        return np.zeros(n, bool)
    m = np.zeros(n, bool)
    for i, dv in enumerate(obj.data.vertices):
        for g in dv.groups:
            if g.group == vg.index and g.weight > 0.5:
                m[i] = True; break
    return m

# ------------------------- utilities: suffix / side -------------------------

def _circ(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def _decide_suffix(h1, use_h2, h2, override):
    if override == 'DORSAL':
        return "_d"
    if override == 'VENTRAL':
        return "_v"
    targets_d = [0.0, 60.0]
    targets_v = [180.0, 240.0]
    hues = [h1] + ([h2] if use_h2 else [])
    d_score = min(_circ(h, t) for h in hues for t in targets_d)
    v_score = min(_circ(h, t) for h in hues for t in targets_v)
    return "_d" if d_score <= v_score else "_v"

# ------------------------- core pipeline steps -------------------------

def _get_phase_and_masks(obj, preferred_attr, allow_linked_white, alpha_thr, sat_min):
    rgba, src_attr, _ = _get_rgba_for_object(obj, preferred=preferred_attr, allow_linked_white=allow_linked_white)
    if rgba is None:
        raise RuntimeError("No vertex colours found and no linked WHITE to map from.")
    rgb   = np.clip(rgba[:, :3], 0.0, 1.0)
    alpha = rgba[:, 3]
    # Use hue + saturation; drop low-alpha or low-saturation verts
    hue, sat = _rgb_to_hue_deg_sat(rgb)
    valid = (alpha > float(alpha_thr)) & (sat >= float(sat_min))
    return hue, valid, src_attr

def step1_candidates(obj, p, suffix):
    hue, valid, src_attr = _get_phase_and_masks(obj, p.attr_name, p.allow_linked_white, p.alpha_thr, p.sat_min)
    adj, _ = _build_adj(obj, weighted=False)
    mask = (_circ_diff_deg(hue, p.v13_hue1_deg) <= p.tol_deg)
    if p.use_second_hue:
        mask |= (_circ_diff_deg(hue, p.v13_hue2_deg) <= p.tol_deg)
    mask &= valid
    if p.use_grad_gate:
        grad = _phase_grad_deg(adj, hue)
        thr  = float(np.percentile(grad[np.isfinite(grad)], p.grad_percentile))
        mask &= (grad >= thr)
    mask = _support_prune(mask, adj, p.min_support)
    _write_group(obj, f"V13_candidates{suffix}", mask)
    return mask, hue, valid, adj, src_attr

def step2_clusters(obj, p, suffix, mask, hue, valid, adj):
    roi = np.ones(len(mask), bool)
    base = mask & roi
    comps = _components(base, adj, p.min_comp_size)

    tol = float(p.tol_deg)
    ms  = int(p.min_support)
    gp  = float(p.grad_percentile)

    while len(comps) < 2 and tol >= p.tol_min_deg:
        tol -= p.tol_step_deg
        newmask = (_circ_diff_deg(hue, p.v13_hue1_deg) <= tol)
        if p.use_second_hue:
            newmask |= (_circ_diff_deg(hue, p.v13_hue2_deg) <= tol)
        newmask &= valid & roi
        if p.use_grad_gate:
            grad = _phase_grad_deg(adj, hue)
            gp = min(100.0, gp + p.grad_step)
            thr = float(np.percentile(grad[np.isfinite(grad)], gp))
            newmask &= (grad >= thr)
        ms = min(ms + 1, 6)
        newmask = _support_prune(newmask, adj, ms)
        comps = _components(newmask, adj, p.min_comp_size)
        base  = newmask

    comps.sort(key=lambda c: len(c), reverse=True)
    n = len(mask)
    if len(comps) == 0:
        _write_group(obj, f"V1_cluster{suffix}", None)
        _write_group(obj, f"V3_cluster{suffix}", None)
        return None, None
    c1 = comps[0]
    c3 = comps[1] if len(comps) > 1 else None
    m1 = np.zeros(n, bool); m1[c1] = True
    m3 = np.zeros(n, bool)
    if c3 is not None: m3[c3] = True
    _write_group(obj, f"V1_cluster{suffix}", m1)
    _write_group(obj, f"V3_cluster{suffix}", m3 if c3 is not None else np.zeros(n, bool))
    return m1, m3

def step3_centerlines(obj, p, suffix, m1, m3):
    adj, _ = _build_adj(obj, weighted=False)
    n = len(obj.data.vertices)
    def do_one(mask, name):
        if mask is None or not np.any(mask):
            return _write_group(obj, f"{name}{suffix}", None)
        nodes = np.nonzero(mask)[0]
        path = _two_bfs_longest_path(adj, nodes)
        line = _mask_from_path(path, n, p.line_widen, adj)
        return _write_group(obj, f"{name}{suffix}", line)
    c1 = do_one(m1, "V1_line")
    c3 = do_one(m3, "V3_line")
    return c1, c3

def step4_midline(obj, p, suffix, valid_mask):
    v1 = _read_group_mask(obj, f"V1_line{suffix}")
    v3 = _read_group_mask(obj, f"V3_line{suffix}")
    if not np.any(v1) or not np.any(v3):
        return _write_group(obj, f"V2_line{suffix}", None)

    adj, wts = _build_adj_weights(obj)
    A = np.nonzero(v1)[0]; B = np.nonzero(v3)[0]

    # Corridor near BOTH lines (intersection of balls)
    near1 = _geodesic_ball(adj, wts, A, p.v2_corridor_mm)
    near3 = _geodesic_ball(adj, wts, B, p.v2_corridor_mm)
    corridor = (near1 & near3) | v1 | v3
    corridor &= valid_mask

    # Distances constrained to the corridor
    d1 = _dijkstra_multi_masked(adj, wts, A, corridor)
    d3 = _dijkstra_multi_masked(adj, wts, B, corridor)

    ok = corridor & np.isfinite(d1) & np.isfinite(d3)
    if not np.any(ok):
        return _write_group(obj, f"V2_line{suffix}", None)

    # Tight equidistance band
    band = ok & (np.abs(d1 - d3) <= float(p.mid_band_mm))

    # Voronoi boundary inside the band
    lab = np.zeros(len(ok), np.int8)
    lab[d1 < d3] = 1
    lab[d3 < d1] = 3
    boundary = _boundary_from_labels(adj, band, lab)
    if not np.any(boundary):
        boundary = band

    # Reduce boundary to one path
    comps = _components(boundary, adj, min_size=8)
    if not comps:
        return _write_group(obj, f"V2_line{suffix}", None)
    best = max(comps, key=lambda c: len(c))
    path = _two_bfs_longest_path(adj, best)
    v2_mask = _mask_from_path(path, len(ok), widen=0, adj=adj)

    return _write_group(obj, f"V2_line{suffix}", v2_mask)

# ------------------------- UI + operators -------------------------

class RETINO_V13_Props(bpy.types.PropertyGroup):
    # source
    attr_name: bpy.props.StringProperty(name="Attr (blank=auto)", default="")
    allow_linked_white: bpy.props.BoolProperty(name="Allow linked WHITE fallback", default=True)

    # side / suffix
    side_mode: bpy.props.EnumProperty(
        name="Side",
        items=[('AUTO',"Auto (_d/_v)",""),
               ('DORSAL',"Dorsal (_d)",""),
               ('VENTRAL',"Ventral (_v)","")],
        default='AUTO'
    )

    # hue targets
    v13_hue1_deg: bpy.props.FloatProperty(name="Hue1°", default=0.0, min=0, max=360)     # dorsal: 0 (red)
    use_second_hue: bpy.props.BoolProperty(name="Use Hue2", default=True)
    v13_hue2_deg: bpy.props.FloatProperty(name="Hue2°", default=60.0, min=0, max=360)    # dorsal: 60 (yellow)

    # candidate gating
    tol_deg: bpy.props.FloatProperty(name="Tol°", default=18.0, min=2.0, max=45.0)
    use_grad_gate: bpy.props.BoolProperty(name="Gradient gate", default=True)
    grad_percentile: bpy.props.FloatProperty(name="Grad %", default=24.0, min=0, max=100)
    min_support: bpy.props.IntProperty(name="Min neighbour support", default=2, min=0, max=8)
    alpha_thr: bpy.props.FloatProperty(name="Alpha >", default=0.02, min=0.0, max=1.0)
    sat_min: bpy.props.FloatProperty(
        name="Min saturation", default=0.1, min=0.0, max=1.0,
        description="Ignore low-saturation (grey/flat) areas when picking hue candidates"
    )

    # cluster step (bridge-break)
    min_comp_size: bpy.props.IntProperty(name="Min cluster size", default=150, min=10, soft_max=10000)
    tol_min_deg: bpy.props.FloatProperty(name="Tol min°", default=6.0, min=1.0, max=45.0)
    tol_step_deg: bpy.props.FloatProperty(name="Tol step°", default=2.0, min=0.5, max=10.0)
    grad_step: bpy.props.FloatProperty(name="Grad tighten %", default=5.0, min=0.0, max=20.0)

    # centerline widen
    line_widen: bpy.props.IntProperty(name="Widen line (rings)", default=0, min=0, max=3)

    # step 4: midline
    v2_corridor_mm: bpy.props.FloatProperty(
        name="V2 corridor (mm)", default=3.0, min=1.0, max=50.0,
        description="Geodesic radius around each line; intersection defines the between-lines corridor")
    mid_band_mm: bpy.props.FloatProperty(
        name="V2 band (mm)", default=2.0, min=0.1, max=10.0,
        description="Equidistance tolerance inside the corridor")

class RETINO_OT_V13_Step1(bpy.types.Operator):
    bl_idname = "retino.v13_step1"
    bl_label  = "Step 1: V1/V3 candidates"
    bl_options = {'REGISTER','UNDO'}
    def execute(self, ctx):
        obj = ctx.active_object; p = ctx.scene.retino_v13
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh."); return {'CANCELLED'}
        suffix = _decide_suffix(p.v13_hue1_deg, p.use_second_hue, p.v13_hue2_deg, p.side_mode)
        try:
            mask, hue, valid, adj, src = step1_candidates(obj, p, suffix)
            self.report({'INFO'}, f"Candidates{suffix}: {int(mask.sum())} (attr: {src})")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}

class RETINO_OT_V13_Step2(bpy.types.Operator):
    bl_idname = "retino.v13_step2"
    bl_label  = "Step 2: Split clusters"
    bl_options = {'REGISTER','UNDO'}
    def execute(self, ctx):
        obj = ctx.active_object; p = ctx.scene.retino_v13
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh."); return {'CANCELLED'}
        suffix = _decide_suffix(p.v13_hue1_deg, p.use_second_hue, p.v13_hue2_deg, p.side_mode)
        cand = _read_group_mask(obj, f"V13_candidates{suffix}")
        if not np.any(cand):
            self.report({'ERROR'}, f"Run Step 1 first (V13_candidates{suffix} empty).")
            return {'CANCELLED'}
        try:
            hue, valid, _ = _get_phase_and_masks(obj, p.attr_name, p.allow_linked_white, p.alpha_thr, p.sat_min)
            adj, _ = _build_adj(obj, weighted=False)
            m1, m3 = step2_clusters(obj, p, suffix, cand, hue, valid, adj)
            n1 = 0 if m1 is None else int(m1.sum()); n3 = 0 if m3 is None else int(m3.sum())
            self.report({'INFO'}, f"Clusters{suffix} — V1:{n1}  V3:{n3}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}

class RETINO_OT_V13_Step3(bpy.types.Operator):
    bl_idname = "retino.v13_step3"
    bl_label  = "Step 3: Centerlines"
    bl_options = {'REGISTER','UNDO'}
    def execute(self, ctx):
        obj = ctx.active_object; p = ctx.scene.retino_v13
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh."); return {'CANCELLED'}
        suffix = _decide_suffix(p.v13_hue1_deg, p.use_second_hue, p.v13_hue2_deg, p.side_mode)
        m1 = _read_group_mask(obj, f"V1_cluster{suffix}")
        m3 = _read_group_mask(obj, f"V3_cluster{suffix}")
        if not np.any(m1) and not np.any(m3):
            self.report({'ERROR'}, f"Run Step 2 first (clusters{suffix} empty).")
            return {'CANCELLED'}
        c1, c3 = step3_centerlines(obj, p, suffix, m1 if np.any(m1) else None, m3 if np.any(m3) else None)
        self.report({'INFO'}, f"Lines{suffix} — V1:{c1}  V3:{c3}")
        return {'FINISHED'}

class RETINO_OT_V13_Step4(bpy.types.Operator):
    bl_idname = "retino.v13_step4"
    bl_label  = "Step 4: V2 midline"
    bl_options = {'REGISTER','UNDO'}
    def execute(self, ctx):
        obj = ctx.active_object; p = ctx.scene.retino_v13
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh."); return {'CANCELLED'}
        suffix = _decide_suffix(p.v13_hue1_deg, p.use_second_hue, p.v13_hue2_deg, p.side_mode)
        try:
            _, valid, _ = _get_phase_and_masks(obj, p.attr_name, p.allow_linked_white, p.alpha_thr, p.sat_min)
            n = step4_midline(obj, p, suffix, valid)
            self.report({'INFO'}, f"V2_line{suffix}: {n} verts")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}

class RETINO_OT_V13_RunAll(bpy.types.Operator):
    bl_idname = "retino.v13_run_all"
    bl_label  = "Run All (1→4)"
    bl_options = {'REGISTER','UNDO'}
    def execute(self, ctx):
        obj = ctx.active_object; p = ctx.scene.retino_v13
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh."); return {'CANCELLED'}
        suffix = _decide_suffix(p.v13_hue1_deg, p.use_second_hue, p.v13_hue2_deg, p.side_mode)
        try:
            mask, hue, valid, adj, _ = step1_candidates(obj, p, suffix)
            m1, m3 = step2_clusters(obj, p, suffix, mask, hue, valid, adj)
            c1, c3 = step3_centerlines(obj, p, suffix, m1, m3)
            n  = step4_midline(obj, p, suffix, valid)
            self.report({'INFO'}, f"Done{suffix}. Cand={int(mask.sum())}  V1={c1}  V3={c3}  V2={n}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}

class VIEW3D_PT_RetinoV13(bpy.types.Panel):
    bl_label = "Retino"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = PANEL_CATEGORY
    def draw(self, ctx):
        l = self.layout; p = ctx.scene.retino_v13

        box = l.box(); box.label(text="Source & Side")
        box.prop(p, "attr_name")
        box.prop(p, "allow_linked_white")
        box.prop(p, "side_mode")

        box = l.box(); box.label(text="Step 1 — V1/V3 hue candidates")
        row = box.row(align=True)
        row.prop(p, "v13_hue1_deg"); row.prop(p, "use_second_hue")
        if p.use_second_hue:
            box.prop(p, "v13_hue2_deg")
        box.prop(p, "tol_deg")
        row = box.row(align=True)
        row.prop(p, "use_grad_gate"); row.prop(p, "grad_percentile")
        box.prop(p, "min_support")
        box.prop(p, "alpha_thr")
        box.prop(p, "sat_min")
        box.operator("retino.v13_step1", icon='COLOR')

        box = l.box(); box.label(text="Step 2 — Cluster & bridge-break")
        row = box.row(align=True)
        row.prop(p, "min_comp_size"); row.prop(p, "tol_min_deg")
        row = box.row(align=True)
        row.prop(p, "tol_step_deg"); row.prop(p, "grad_step")
        box.operator("retino.v13_step2", icon='MOD_DECIM')

        box = l.box(); box.label(text="Step 3 — Centerlines")
        box.prop(p, "line_widen")
        box.operator("retino.v13_step3", icon='IPO_LINEAR')

        box = l.box(); box.label(text="Step 4 — V2 midline (equidistant)")
        box.prop(p, "v2_corridor_mm")
        box.prop(p, "mid_band_mm")
        box.operator("retino.v13_step4", icon='IPO_QUAD')

        l.separator()
        l.operator("retino.v13_run_all", icon='SEQ_SEQUENCER')

# ------------------------- register -------------------------

_classes = (
    RETINO_V13_Props,
    RETINO_OT_V13_Step1, RETINO_OT_V13_Step2, RETINO_OT_V13_Step3, RETINO_OT_V13_Step4,
    RETINO_OT_V13_RunAll,
    VIEW3D_PT_RetinoV13,
)

def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.retino_v13 = bpy.props.PointerProperty(type=RETINO_V13_Props)

def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.retino_v13
