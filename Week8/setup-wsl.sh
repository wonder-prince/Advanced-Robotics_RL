#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

case "${1:-}" in
  '') assets_only=false ;;
  --assets-only) assets_only=true ;;
  *) echo 'Usage: bash setup-wsl.sh [--assets-only]' >&2; exit 2 ;;
esac

if ! $assets_only; then
  if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]] || ! command -v apt-get >/dev/null; then
    echo 'Run the full setup in x86-64 Ubuntu on WSL 2.' >&2
    exit 1
  fi

  apt=(apt-get)
  if (( EUID != 0 )); then
    apt=(sudo apt-get)
  fi
  "${apt[@]}" update
  unity_libraries=(libgl1 libnss3 libx11-6 libxcursor1 libxi6 libxrandr2)
  for library in libasound2 libglib2.0-0 libgtk-3-0; do
    if apt-cache show "${library}t64" >/dev/null 2>&1; then
      unity_libraries+=("${library}t64")
    else
      unity_libraries+=("$library")
    fi
  done
  "${apt[@]}" install -y --no-install-recommends ca-certificates curl unzip "${unity_libraries[@]}"
fi

mkdir -p builds/archives builds/linux results
scratch=$(mktemp -d builds/archives/.setup.XXXXXXXX)
trap 'rm -r -- "$scratch"' EXIT

verify_archive() {
  local archive=$1 digest=$2
  if command -v sha256sum >/dev/null; then
    printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
  else
    printf '%s  %s\n' "$digest" "$archive" | shasum -a 256 -c -
  fi
}

prepare_player() {
  local name=$1 digest=$2 url=$3 executable=$4
  local archive="builds/archives/$name.zip"
  local destination=builds/linux
  if [[ $name == SoccerTwos ]]; then
    destination+=/SoccerTwos
  fi
  if [[ ! -f $archive ]]; then
    curl -fL --retry 3 -o "$scratch/$name.zip" "$url"
    verify_archive "$scratch/$name.zip" "$digest"
    mv -- "$scratch/$name.zip" "$archive"
  else
    verify_archive "$archive" "$digest"
  fi
  unzip -oq "$archive" -d "$destination"
  chmod +x "builds/linux/$executable"
}

prepare_player SnowballTarget 7cc64fe0e7fc7f6f27e4ce1e5a7cd90c765b73b49bab1e97ad3f44b6b24b737d \
  'https://media.githubusercontent.com/media/huggingface/Snowball-Target/1bc2647d32c4f83dbfacf53af57cb5e44896386a/SnowballTarget.zip' \
  SnowballTarget/SnowballTarget.x86_64
prepare_player Pyramids 1e0f1cfa88a380b42644d974525a7dbfe144089883a401526b8341c5441f7cae \
  'https://huggingface.co/spaces/unity/ML-Agents-Pyramids/resolve/9b7a0bb012f66e9ae0ead016e6d3151990058c64/Pyramids.zip' \
  Pyramids/Pyramids
prepare_player SoccerTwos 5f808039ff3f69f1d0b96b43f2df115a13b7af6cdb2b85276d0cf66172073b13 \
  'https://drive.usercontent.google.com/download?id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL&export=download&confirm=t' \
  SoccerTwos/SoccerTwos.x86_64

if $assets_only; then
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
if ! command -v uv >/dev/null; then
  python3 -m pip install --index-url https://mirrors.aliyun.com/pypi/simple 'uv==0.8.16'
fi
if [[ $(python3 -c 'import platform; print(platform.python_version())') == 3.10.12 ]]; then
  uv sync --locked --python "$(command -v python3)"
else
  uv python install 3.10.12
  uv sync --locked
fi
for player in builds/linux/{SnowballTarget,SoccerTwos}/UnityPlayer.so builds/linux/Pyramids/UnityPlayer.so; do
  dependencies=$(ldd "$player")
  if [[ $dependencies == *'not found'* ]]; then
    printf 'Missing shared library for %s\n%s\n' "$player" "$dependencies" >&2
    exit 1
  fi
done
uv run --no-sync mlagents-learn --help >/dev/null
for experiment in snowball pyramids soccer; do
  uv run --no-sync train.py "$experiment" --check
done

echo 'Week8 environment is ready.'
