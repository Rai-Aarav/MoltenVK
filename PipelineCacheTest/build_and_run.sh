#!/bin/bash
# Builds pipeline_cache_test and runs it against a given libMoltenVK.dylib.
#
# Usage: build_and_run.sh <dir containing libMoltenVK.dylib> <dir containing vulkan/vulkan.h> [shaderCount] [threadCount]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
libdir="$(cd "$1" && pwd)"
incdir="$(cd "$2" && pwd)"
shift 2
out="$(mktemp -d)/pipeline_cache_test"
clang++ -std=c++17 -O2 -Wall -I "$incdir" "$here/pipeline_cache_test.cpp" -L "$libdir" -lMoltenVK -Wl,-rpath,"$libdir" -o "$out"
MVK_CONFIG_LOG_LEVEL="${MVK_CONFIG_LOG_LEVEL:-2}" DYLD_LIBRARY_PATH="$libdir" "$out" "$@"
