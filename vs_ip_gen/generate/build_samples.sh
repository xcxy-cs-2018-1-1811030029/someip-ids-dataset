#!/usr/bin/env bash
# Build the SOME/IP IDS dataset generator (signal_service + signal_client).
# Run inside WSL, from this directory. Requires vSomeIP installed (setup_vsomeip.sh).
set -euo pipefail
BOOST="$HOME/boost_1_83_0"
CXX="g++"
FLAGS="-std=c++17 -O2"

echo ">> building signal_service ..."
$CXX $FLAGS -o signal_service signal_service.cpp \
  -I/usr/local/include -I"$BOOST/include" \
  -L/usr/local/lib -L"$BOOST/lib" \
  -lvsomeip3 \
  -lboost_system -lboost_thread -lboost_log -lboost_filesystem \
  -lboost_regex -lboost_date_time -lboost_serialization -lboost_chrono -lboost_atomic \
  -lpthread -ldl

echo ">> building signal_client ..."
$CXX $FLAGS -o signal_client signal_client.cpp \
  -I/usr/local/include -I"$BOOST/include" \
  -L/usr/local/lib -L"$BOOST/lib" \
  -lvsomeip3 \
  -lboost_system -lboost_thread -lboost_log -lboost_filesystem \
  -lboost_regex -lboost_date_time -lboost_serialization -lboost_chrono -lboost_atomic \
  -lpthread -ldl

echo ">> ok. binaries:"; ls -lh signal_service signal_client
echo ">> run with:"
echo "   export LD_LIBRARY_PATH=/usr/local/lib:$BOOST/lib:\$LD_LIBRARY_PATH"
