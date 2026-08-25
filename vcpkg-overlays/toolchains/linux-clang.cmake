set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

# nvcc otherwise defaults to g++, which rejects the -stdlib=libc++ the linux
# triplets put in VCPKG_CXX_FLAGS / VCPKG_LINKER_FLAGS, so enabling the CUDA
# language fails at the compiler ABI check. Inert for ports without CUDA.
set(CMAKE_CUDA_HOST_COMPILER "clang++")

include("$ENV{VCPKG_ROOT}/scripts/toolchains/linux.cmake")
