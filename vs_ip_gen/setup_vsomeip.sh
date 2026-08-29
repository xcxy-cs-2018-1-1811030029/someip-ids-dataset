#!/usr/bin/env bash
# =============================================================================
# Setup vSomeIP inside WSL2 Ubuntu-22.04 for SOME/IP IDS data generation
# Run:  bash setup_vsomeip.sh      (will prompt for sudo password)
#
# NOTE: Ubuntu 22.04 ships Boost 1.74, but vSomeIP >= 3.7 requires Boost >= 1.75.
# This script therefore builds Boost 1.83.0 from source into $HOME/boost_1_83_0
# and points vSomeIP's cmake at it. Compiling Boost takes ~15-30 min on a fast
# CPU; vSomeIP itself builds in a few minutes.
# =============================================================================
set -euo pipefail
JP="$(nproc)"
BOOST_VER="1.83.0"
BOOST_SRC="$HOME/boost_1_83_0"

echo ">> [1/6] apt update + build tooling (idempotent)"
sudo apt-get update -y
sudo apt-get install -y \
  build-essential cmake git \
  libboost-system-dev libboost-thread-dev libboost-log-dev \
  libasio-dev libssl-dev \
  python3 python3-pip python3-venv \
  tcpdump iproute2 net-tools

echo ">> [2/6] ensure Boost >= 1.75 is available"
if [ ! -f "$BOOST_SRC/include/boost/version.hpp" ]; then
  echo "   Building Boost ${BOOST_VER} from source into ${BOOST_SRC} (this takes a while)..."
  cd "$HOME"
  wget -c -q "https://archives.boost.io/release/${BOOST_VER}/source/boost_${BOOST_VER//./_}.tar.gz" -O "boost_${BOOST_VER//./_}.tar.gz"
  tar xzf "boost_${BOOST_VER//./_}.tar.gz"
  cd "boost_${BOOST_VER//./_}"
  ./bootstrap.sh --prefix="$BOOST_SRC" \
     --with-libraries=system,filesystem,thread,log,date_time,regex,serialization,atomic,chrono
  ./b2 -j"$JP" threading=multi link=shared --prefix="$BOOST_SRC" install
else
  echo "   Boost ${BOOST_VER} already built. Skipping."
fi

echo ">> [3/6] clone COVESA vSomeIP"
cd "$HOME"
if [ ! -d vsomeip ]; then
  git clone https://github.com/COVESA/vsomeip.git
fi
cd vsomeip

echo ">> [4/6] configure + build vSomeIP (Release, using Boost from ${BOOST_SRC})"
rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="$BOOST_SRC" \
  -DBOOST_LIBRARYDIR="$BOOST_SRC/lib" \
  -DWITH_KOMORI_PLUGIN=OFF -DWITH_DLT=OFF ..
make -j"$JP"
sudo make install
sudo ldconfig

echo ">> [5/6] python deps (feature/analysis scripts)"
python3 -m pip install --user --upgrade pip 2>/dev/null || true
python3 -m pip install --user numpy pandas scikit-learn scapy 2>/dev/null || true
# Torch with CUDA (WSL2 GPU passthrough). CPU fallback if it fails.
python3 -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
  || python3 -m pip install --user torch torchvision

echo ">> [6/6] verify"
echo "   Boost: $($BOOST_SRC/include/boost/version.hpp 2>/dev/null | grep -m1 BOOST_LIB_VERSION || echo n/a)"
echo "   nvidia-smi (WSL2):"
nvidia-smi 2>/dev/null | head -15 || echo "   (GPU not exposed to WSL; generation still works, only training needs GPU)"
echo "   vSomeIP examples:"
ls "$HOME/vsomeip/build/examples" 2>/dev/null || ls "$HOME/vsomeip/examples" 2>/dev/null || echo "   (no examples dir found)"
echo
echo "SUGGESTED NEXT:"
echo "  export LD_LIBRARY_PATH=$BOOST_SRC/lib:\$LD_LIBRARY_PATH"
echo "  Run a hello_world publish-subscribe demo to confirm the stack, e.g.:"
echo "    ./hello_world_service &  ./hello_world_client"
echo "  Capture traffic:  sudo tcpdump -i eth0 -w normal.pcap 'udp port 30500 or 30490'"
