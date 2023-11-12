#!/usr/bin/env python3
from pathlib import Path
import mpot_rearmour

def get_root_path() -> Path:
    return Path(mpot_rearmour.__file__).resolve().parents[1]

def get_urdf_path() -> Path:
    return get_root_path() / "assets"
