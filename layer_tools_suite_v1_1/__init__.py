# Init.py

bl_info = {
    "name": "Layer Tools Suite (NIfTI Paint, Vertex Select, Retino)",
    "author": "",
    "version": (1, 0, 3),
    "blender": (2, 93, 0),
    "location": "3D Viewport > Sidebar (N)",
    "description": "Split tabs: NIfTI painting, Vertex selection+ROIs+Peaks, and Retino V1/V2/V3 fitter.",
    "category": "Mesh",
}

# ---------------- Dependency bootstrap (local 'pydeps') ----------------

import os, sys, importlib, subprocess, shutil

# Blender API (loaded lazily by Blender, safe to import here)
import bpy
from bpy.types import AddonPreferences, Operator

# Add-on dir + local deps path
_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_DEP_DIR   = os.path.join(_ADDON_DIR, "pydeps")

# Make sure our local deps are importable before loading submodules
if os.path.isdir(_DEP_DIR) and _DEP_DIR not in sys.path:
    sys.path.insert(0, _DEP_DIR)

_REQUIRED_PY_DEPS = [
    # keep this short and stable; Blender ships numpy already
    "nibabel>=5.0.0",
]

def _have_module(mod_name: str) -> bool:
    try:
        import importlib.util as _util
        return _util.find_spec(mod_name) is not None
    except Exception:
        return False

def _missing_deps():
    missing = []
    for spec in _REQUIRED_PY_DEPS:
        # the top-level import name is the bit before any comparison operator
        top = spec.split("[")[0]
        for tok in ("==", ">=", "<=", "~=", ">", "<"):
            if tok in top:
                top = top.split(tok, 1)[0]
        top = top.strip()
        name = spec.split("[", 1)[0].split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("~=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip()
        if not _have_module(top or name):
            missing.append(spec)
    return missing

def _ensure_pip_available():
    try:
        import pip  # noqa: F401
        return True
    except Exception:
        try:
            import ensurepip
            ensurepip.bootstrap()  # installs pip into Blender's Python
            return True
        except Exception:
            return False

def _pip_install(packages, target_dir, verbose=False):
    os.makedirs(target_dir, exist_ok=True)

    # Upgrade pip silently (best effort)
    if _ensure_pip_available():
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            pass
    else:
        raise RuntimeError("Could not bootstrap pip in Blender's Python (no ensurepip available).")

    base_cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", target_dir,
        "--upgrade",
        "--prefer-binary",
        "--no-warn-script-location",
    ]

    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    for pkg in packages:
        cmd = base_cmd + [pkg]
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=(None if verbose else subprocess.PIPE),
            stderr=(None if verbose else subprocess.PIPE),
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pip failed for '{pkg}'. "
                f"Try again from Preferences or install manually."
            )

def _try_auto_install(verbose=False):
    missing = _missing_deps()
    if not missing:
        return True, "All dependencies present."

    try:
        _pip_install(missing, _DEP_DIR, verbose=verbose)

        if _DEP_DIR not in sys.path:
            sys.path.insert(0, _DEP_DIR)

        # Re-check
        still = _missing_deps()
        if still:
            return False, f"Installed some, but still missing: {', '.join(still)}"

        return True, "Dependencies installed."

    except Exception as e:
        return False, f"Auto-install failed: {e}"

# ---------------- Preferences UI (Install/Repair button) ----------------

class LTS_AddonPreferences(AddonPreferences):
    bl_idname = __name__

    auto_install: bpy.props.BoolProperty(
        name="Try auto-install on enable",
        default=True,
        description=(
            "When the add-on is enabled and a dependency is missing, "
            "attempt to install it into this add-on's 'pydeps' folder."
        ),
    )

    verbose_pip: bpy.props.BoolProperty(
        name="Verbose pip output (console)",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Python Dependencies", icon='SCRIPT')

        # status line
        missing = _missing_deps()
        if not missing:
            row = box.row()
            row.label(text="OK: nibabel is available.", icon='CHECKMARK')
        else:
            row = box.row()
            row.label(text=f"Missing: {', '.join(missing)}", icon='ERROR')

        col = box.column(align=True)
        col.prop(self, "auto_install")
        col.prop(self, "verbose_pip")
        col.separator()

        row = col.row(align=True)

        op = row.operator(
            "layer_tools.install_python_deps",
            text="Install/Repair Python Dependencies",
            icon='CONSOLE',
        )
        op.repair = False
        op.verbose = self.verbose_pip

        op2 = row.operator(
            "layer_tools.install_python_deps",
            text="Repair (clean & reinstall)",
            icon='FILE_REFRESH',
        )
        op2.repair = True
        op2.verbose = self.verbose_pip

        if os.path.isdir(_DEP_DIR):
            col.label(text=f"Local site-packages: {os.path.relpath(_DEP_DIR, _ADDON_DIR)}")

class LTS_OT_InstallDeps(Operator):
    bl_idname = "layer_tools.install_python_deps"
    bl_label = "Install or Repair Python Dependencies"
    bl_options = {'INTERNAL'}

    repair: bpy.props.BoolProperty(default=False)
    verbose: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        try:
            if self.repair and os.path.isdir(_DEP_DIR):
                shutil.rmtree(_DEP_DIR, ignore_errors=True)

            _pip_install(_REQUIRED_PY_DEPS, _DEP_DIR, verbose=self.verbose)

            if _DEP_DIR not in sys.path:
                sys.path.insert(0, _DEP_DIR)

        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        self.report({'INFO'}, "Dependencies installed.")
        return {'FINISHED'}

# ---------------- Module wiring ----------------

# Support both packaged and flat installs
if __package__ in (None, ""):
    _dir = os.path.dirname(__file__)
    if _dir and _dir not in sys.path:
        sys.path.append(_dir)

    common            = importlib.import_module("common")
    tab_nifti_paint   = importlib.import_module("tab_nifti_paint")
    tab_vertex_select = importlib.import_module("tab_vertex_select")
    tab_retino        = importlib.import_module("tab_retino")

else:
    from . import (
        common,
        tab_nifti_paint,
        tab_vertex_select,
        tab_retino,
    )

_submods = [
    common,
    tab_nifti_paint,
    tab_vertex_select,
    tab_retino,
]

# ---------------- Register / Unregister ----------------

_classes = (
    LTS_AddonPreferences,
    LTS_OT_InstallDeps,
)

def register():
    # Register prefs + operator first
    for c in _classes:
        bpy.utils.register_class(c)

    # Try auto install if requested and something is missing
    prefs = bpy.context.preferences.addons.get(__name__)
    auto = True
    verbose = False

    if prefs and hasattr(prefs, "preferences"):
        auto = prefs.preferences.auto_install
        verbose = prefs.preferences.verbose_pip

    if auto:
        ok, msg = _try_auto_install(verbose=verbose)
        if not ok:
            print(f"[Layer Tools Suite] {msg}")

    # Allow hot reloading from Blender’s Text Editor
    for m in _submods:
        importlib.reload(m)

    # Register submodules
    tab_nifti_paint.register()
    tab_vertex_select.register()
    tab_retino.register()

def unregister():
    # Unregister submodules in reverse order
    tab_retino.unregister()
    tab_vertex_select.unregister()
    tab_nifti_paint.unregister()

    for c in reversed(_classes):
        bpy.utils.unregister_class(c)