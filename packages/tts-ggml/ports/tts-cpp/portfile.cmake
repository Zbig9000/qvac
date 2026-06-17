# tts-cpp — LOCAL OVERLAY PORT.
#
# *** QVAC-20616 DEVICE-FARM TEST PIN — DO NOT MERGE ***
# Temporarily repoints this overlay at Zbig9000/qvac-ext-lib-whisper.cpp@
# ea51e37b (branch QVAC-20616-devicefarm-test) so the AWS Device Farm
# (Android OpenCL + iOS Metal) exercises the QVAC-20616 end-of-speech fix
# (Phase 1 heuristic stop + Phase 2 alignment EOS + improvements #1-#5) on
# real mobile GPUs. That branch == the Android dlopen-fix baseline f7d4d6c
# (described below) with the 9 QVAC-20616 commits rebased on top, so the only
# delta vs the known-good pin is QVAC-20616 itself. Revert this overlay to the
# registry `tts-cpp` once the fix is published.
#
# Baseline f7d4d6c (now the parent of the test pin) replaces the registry
# `tts-cpp` port so the tts-ggml prebuild builds an upstream fix that is NOT
# yet published to qvac-registry-vcpkg. It is exactly one commit on top of the
# published 2026-06-05 pin (128dae42) which reroutes Supertonic's direct
# CPU-backend calls that are unlinkable under `GGML_BACKEND_DL=ON`:
#   - `ggml_get_type_traits_cpu(...)->from_float` -> `ggml_quantize_chunk()`
#     (ggml-base, always linked)
#   - `ggml_backend_is_cpu()` -> `tts_cpp::detail::backend_is_cpu()` registry
#     shim (the pattern Chatterbox / parakeet already use)
#
# Why this exists: tts-cpp@2026-06-05 (QVAC-19254 sched + cpu_backend
# refactor) left `ggml_backend_is_cpu` / `ggml_get_type_traits_cpu` as
# undefined symbols in `libqvac__tts-ggml.*.so`, so the addon fails to
# `dlopen` on Android (per-arch CPU backends are dlopen'd lazily, not
# linked) -> SIGABRT at bootstrap -> all Android e2e dead.
#
# How to verify the fix: build the Android prebuild from this overlay, then
#   llvm-readelf --dyn-syms prebuilds/android-arm64/qvac__tts-ggml.bare \
#     | grep -E 'UND.*(ggml_backend_is_cpu|ggml_get_type_traits_cpu)'
# The CPU symbols should no longer appear as UND.
#
# TEMPORARY: remove this overlay (and the `overlay-ports` entry in
# vcpkg-configuration.json) once the fix is published to the registry and
# consumed via vcpkg.json. Tracked as a QVAC-19254 follow-up.
#
# Base pin 128dae42 brought PR #31 (supertonic_optimizations): QVAC-18605
# Supertonic Vulkan/Metal optimisations, QVAC-19254 sched + cpu_backend
# refactor, QVAC-19213 Adreno-generation parse fix.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO Zbig9000/qvac-ext-lib-whisper.cpp
    REF ea51e37b07ba24140145c371f9a0452b74e47126
    SHA512 cab29c9119e3738753703ed403bb3e22fe02be0c5c830c506675aa26f4c8b558825978934e5ca3cd89521702a5a5e4f2188dddc08641bbc87a8f2203b7344c42
    HEAD_REF QVAC-20616-devicefarm-test
)

set(SOURCE_PATH "${WHISPER_CPP_SRC}/tts-cpp")
if (NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "tts-cpp: ${SOURCE_PATH}/CMakeLists.txt missing; the tts-cpp/ "
        "subfolder layout in qvac-ext-lib-whisper.cpp may have changed.")
endif()

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        metal   GGML_METAL
        vulkan  GGML_VULKAN
        cuda    GGML_CUDA
        opencl  GGML_OPENCL
)

set(PLATFORM_OPTIONS)

if(NOT VCPKG_TARGET_IS_OSX)
    list(APPEND PLATFORM_OPTIONS
        -DGGML_BLAS=OFF
        -DGGML_ACCELERATE=OFF
        -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DTTS_CPP_BUILD_LIBRARY=ON
        -DTTS_CPP_BUILD_SHARED=OFF
        -DTTS_CPP_BUILD_EXECUTABLES=OFF
        -DTTS_CPP_BUILD_TESTS=OFF
        -DTTS_CPP_INSTALL=ON
        -DTTS_CPP_USE_SYSTEM_GGML=ON
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF
        -DTTS_CPP_OPENMP=OFF
        -DGGML_CCACHE=OFF
        -DTTS_CPP_CCACHE=OFF
        ${FEATURE_OPTIONS}
        ${PLATFORM_OPTIONS}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME tts-cpp CONFIG_PATH share/tts-cpp)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if (VCPKG_LIBRARY_LINKAGE MATCHES "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
