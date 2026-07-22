# parakeet-cpp: NVIDIA Parakeet ASR + Sortformer diarization in pure C++/ggml.
# Sourced from the parakeet-cpp/ subfolder of tetherto/qvac-ext-lib-whisper.cpp;
# consumes the ggml-speech port.
#
# Long-audio memory fix: bound offline-transcription memory. transcribe_samples
# / transcribe_samples_stream previously ran the conformer encoder over the whole
# input in a single graph (O(T_enc^2) self-attention), OOMing on multi-hour files
# (~100 GB for a 90 min file, SIGKILL). This pin computes the mel once (global
# CMVN) and slides the encoder over it in overlapping windows, trimming the
# shared context at the interior seams; inputs that fit one window keep the
# bit-identical single-pass path. Layered on top of master ecac5bb7 (PR #85, EOU
# RNN-T decoder as ggml graphs on GPU). Requires ggml-speech >= 2026-07-15
# (unchanged from the previous pin).
#
# WIP: REPO/REF/HEAD_REF point at the Zbig9000 fork branch pending the upstream
# PR merge (SHA512 is the hash of that fork tarball). NOTE: master has since
# moved parakeet-cpp/ to engines/parakeet/ (reorg PR #95); this pin deliberately
# stays on the pre-reorg ecac5bb7 line so the bugfix does not also adopt the
# layout migration. On upstream merge, repoint REPO -> tetherto, REF -> the merge
# commit, HEAD_REF -> master, update SOURCE_PATH to engines/parakeet if the fix
# lands on the reorged tree, and recompute SHA512.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO Zbig9000/qvac-ext-lib-whisper.cpp
    REF e5f22556bffb791be0c230281edb8e1a06564ee4
    SHA512 894078f9e6f96e95193d28552100b29b03e077358bb1aa8a8f4d694dcd78fe85ca863c680cd36560a4cab6d2d9f1eb1d49161be4f320a3bbcd7c5f4baac2e8d1
    HEAD_REF QVAC-22367-parakeet-long-audio
)

set(SOURCE_PATH "${WHISPER_CPP_SRC}/parakeet-cpp")
if (NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "parakeet-cpp: ${SOURCE_PATH}/CMakeLists.txt missing; the parakeet-cpp/ "
        "subfolder layout in qvac-ext-lib-whisper.cpp may have changed.")
endif()

set(GGML_METAL  OFF)
set(GGML_VULKAN OFF)
set(GGML_CUDA   OFF)
set(GGML_OPENCL OFF)
if("metal" IN_LIST FEATURES)
    set(GGML_METAL ON)
endif()
if("vulkan" IN_LIST FEATURES)
    set(GGML_VULKAN ON)
endif()
if("cuda" IN_LIST FEATURES)
    set(GGML_CUDA ON)
endif()
if("opencl" IN_LIST FEATURES)
    set(GGML_OPENCL ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DPARAKEET_BUILD_LIBRARY=ON
        -DPARAKEET_BUILD_EXECUTABLES=OFF
        -DPARAKEET_BUILD_TESTS=OFF
        -DPARAKEET_BUILD_EXAMPLES=OFF
        -DPARAKEET_INSTALL=ON
        -DPARAKEET_USE_SYSTEM_GGML=ON
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF
        -DPARAKEET_OPENMP=OFF
        -DGGML_CCACHE=OFF
        -DPARAKEET_CCACHE=OFF
        -DGGML_METAL=${GGML_METAL}
        -DGGML_VULKAN=${GGML_VULKAN}
        -DGGML_CUDA=${GGML_CUDA}
        -DGGML_OPENCL=${GGML_OPENCL}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME parakeet-cpp CONFIG_PATH share/parakeet-cpp)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if (VCPKG_LIBRARY_LINKAGE MATCHES "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
