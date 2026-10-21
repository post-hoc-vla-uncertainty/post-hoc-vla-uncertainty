#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VLA_BUILD_ROOT="$PWD"
export UV_CACHE_DIR="$VLA_BUILD_ROOT/.cache/uv"
mkdir -p upstream .local/build
if [[ ! -d upstream/uq_vla/.git ]]; then
  git clone https://github.com/learnsyslab/uq_vla.git upstream/uq_vla
  git -C upstream/uq_vla checkout daffea8dd1b7b865a773d52ca0cb910d9259f6b6
fi
if [[ "$(git -C upstream/uq_vla rev-parse HEAD)" != daffea8dd1b7b865a773d52ca0cb910d9259f6b6 ]]; then
  echo 'Upstream revision differs from the pinned commit.' >&2
  exit 1
fi
if git -C upstream/uq_vla apply --reverse --check "$VLA_BUILD_ROOT/patches/upstream-compat.patch" 2>/dev/null; then
  : # Already applied.
else
  git -C upstream/uq_vla apply --check "$VLA_BUILD_ROOT/patches/upstream-compat.patch"
  git -C upstream/uq_vla apply "$VLA_BUILD_ROOT/patches/upstream-compat.patch"
fi
# Include the patch's new helper in the working-tree diff checked by the launcher.
git -C upstream/uq_vla add --intent-to-add src/lerobot/utils/gpu_utils.py
uv sync --locked
if [[ ! -d upstream/fiper/.git ]]; then
  git clone https://github.com/learnsyslab/fiper.git upstream/fiper
  git -C upstream/fiper checkout 13d79c5c3069def843e454787ff128defc249838
fi
if [[ "$(git -C upstream/fiper rev-parse HEAD)" != 13d79c5c3069def843e454787ff128defc249838 ]]; then
  echo 'FIPER revision differs from the pinned commit.' >&2
  exit 1
fi
if [[ ! -f .local/ffmpeg-7.1.1/lib/libavcodec.so.61 ]]; then
  curl -fL https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz -o .local/build/ffmpeg-7.1.1.tar.xz
  echo '733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1  .local/build/ffmpeg-7.1.1.tar.xz' | sha256sum -c -
  tar -xf .local/build/ffmpeg-7.1.1.tar.xz -C .local/build
  (
    cd .local/build/ffmpeg-7.1.1
    ./configure --prefix="$VLA_BUILD_ROOT/.local/ffmpeg-7.1.1" --enable-shared --disable-static \
      --disable-doc --disable-programs --disable-autodetect --disable-x86asm
    make -j8
    make install
  )
fi
source scripts/env.sh
python scripts/prepare_assets.py
python scripts/prepare_simulator.py
python scripts/configure_reproduction.py
echo 'Setup complete. Run: bash scripts/reproduce.sh verify --gpus 0,1'
