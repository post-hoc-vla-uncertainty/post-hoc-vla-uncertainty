#!/usr/bin/env python3
"""Build the isolated extension checkout without changing the reproduction checkout."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'upstream/uq_vla'
TARGET=ROOT/'upstream/uq_vla_extensions'
PATCH=ROOT/'patches/extensions-upstream.patch'
PIN='daffea8dd1b7b865a773d52ca0cb910d9259f6b6'


def main():
    if not TARGET.exists():
        subprocess.run(['git','clone','--no-hardlinks',str(SOURCE),str(TARGET)],check=True)
        subprocess.run(['git','checkout','--detach',PIN],cwd=TARGET,check=True)
        subprocess.run(['git','apply',str(PATCH)],cwd=TARGET,check=True)
        subprocess.run(['git','add','--intent-to-add','src/lerobot/utils/gpu_utils.py'],cwd=TARGET,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TARGET,text=True).strip()
    actual=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=TARGET)
    if commit!=PIN or actual!=PATCH.read_bytes():
        raise SystemExit('Extension checkout differs from the pinned extension patch; review local changes before running.')
    print('Isolated extension checkout verified; existing uv environment is reused.')


if __name__=='__main__':main()
