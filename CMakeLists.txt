cmake_minimum_required(VERSION 3.18)
project(dorypony C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(CONCAT CMAKE_CXX_FLAGS_COMMON_STR ""
                        "-Werror -Wall -Wconversion -Wfloat-equal "
                        "-Wpedantic -Wpointer-arith -Wswitch-default "
                        "-Wpacked -Wextra -Winvalid-pch "
                        "-Wmissing-field-initializers "
                        "-Wunreachable-code -Wcast-align -Wcast-qual "
                        "-Wdisabled-optimization -Wformat=2 "
                        "-Wformat-nonliteral -Wuninitialized "
                        "-Wformat-security -Wformat-y2k -Winit-self "
                        "-Wmissing-declarations -Wmissing-include-dirs "
                        "-Wredundant-decls -Wstrict-overflow=5 -Wundef "
                        "-Wno-unused -Wctor-dtor-privacy -Wsign-promo "
                        "-Woverloaded-virtual -Wold-style-cast")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(CONCAT CMAKE_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS_COMMON_STR} "
                        "-Wlogical-op -Wstrict-null-sentinel -Wnoexcept")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_STR}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    string(CONCAT CMAKE_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS_COMMON_STR} ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_STR}")
endif()

string(CONCAT CMAKE_C_FLAGS_COMMON_STR ""
                        "-Werror -Wall -Wconversion -Wfloat-equal "
                        "-Wpedantic -Wpointer-arith -Wswitch-default "
                        "-Wpacked -Wextra -Winvalid-pch "
                        "-Wmissing-field-initializers -Wunreachable-code "
                        "-Wcast-align -Wcast-qual -Wdisabled-optimization "
                        "-Wformat=2 -Wformat-nonliteral -Wuninitialized "
                        "-Wformat-security -Wformat-y2k -Winit-self "
                        "-Wmissing-declarations -Wmissing-include-dirs "
                        "-Wredundant-decls -Wstrict-overflow=5 "
                        "-Wundef -Wno-unused")

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    string(CONCAT CMAKE_C_FLAGS_STR "${CMAKE_C_FLAGS_COMMON_STR} "
                        "-Wlogical-op")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_STR}")
elseif (CMAKE_C_COMPILER_ID MATCHES "Clang")
    string(CONCAT CMAKE_C_FLAGS_STR "${CMAKE_C_FLAGS_COMMON_STR} ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_STR}")
endif()



set(CMAKE_C_FLAGS_DEBUG "-Winline -g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

set(CMAKE_CXX_FLAGS_DEBUG "-Winline -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

set(CMAKE_CUDA_FLAGS_DEBUG "-Winline -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# To bump to a version 17, you need CMake 3.18 or above. You can install it as
# follows: wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc
# 2>/dev/null | sudo apt-key add - sudo apt-add-repository 'deb
# https://apt.kitware.com/ubuntu/ bionic main' sudo apt update && sudo apt
# install cmake

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# message(STATUS "Selecting appropriate gencodes for x86 CPU architectures")
# set(GPU_ARCHS "60") # Default minimum supported CUDA gencode for cuML
#
# if(CUDA_VERSION_MAJOR GREATER_EQUAL 9) set(GPU_ARCHS "${GPU_ARCHS};70")
# endif()
#
# if(CUDA_VERSION_MAJOR GREATER_EQUAL 10) set(GPU_ARCHS "${GPU_ARCHS};75")
# endif()
#
# if(CUDA_VERSION_MAJOR GREATER_EQUAL 11) set(GPU_ARCHS "${GPU_ARCHS};80;86")
# endif()
#
# set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHS}") set(CUDA_ARCHITECTURES
# "${CMAKE_CUDA_ARCHITECTURES}")
#
# message(STATUS "Building for GPU_ARCHS = ${GPU_ARCHS}") foreach(arch
# ${GPU_ARCHS}) set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode
# arch=compute_${arch},code=sm_${arch}") endforeach()

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
  set(CMAKE_CUDA_ARCHITECTURES 52 61 75 86)
endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

string(
  CONCAT CMAKE_CUDA_INTFLAGS_STR
         "-Xcompiler "
         "-Wall,-Werror,-Wconversion,-Wfloat-equal,"
         "-Wpointer-arith,-Wswitch-default,"
         "-Wpacked,-Wextra,-Winvalid-pch,"
         "-Wmissing-field-initializers,"
         "-Wunreachable-code,-Wcast-align,-Wcast-qual,"
         "-Wdisabled-optimization,-Wformat=2,"
         "-Wformat-nonliteral,-Wuninitialized,"
         "-Wformat-security,-Wformat-y2k,-Winit-self,"
         # Commented out, otherwise the separable compilation fails
         # "-Wmissing-declarations,"
         "-Wmissing-include-dirs,"
         "-Wstrict-overflow=5,"
         "-Wno-unused,-Wctor-dtor-privacy,-Wsign-promo,"
         "-Woverloaded-virtual,-Wextra-semi")

string(
  CONCAT CMAKE_CUDA_EXTFLAGS_STR
         ""
         "-Werror all-warnings "
         "-Wreorder "
         # "-Wdefault-stream-launch " "-Wmissing-launch-bounds "
         "-Wext-lambda-captures-this "
         "-Wno-deprecated-declarations -Wno-deprecated-gpu-targets")

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_EXTFLAGS_STR} ${CMAKE_CUDA_INTFLAGS_STR}")

#MESSAGE( STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS} )
#MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )
message(STATUS "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

set(SOURCE src/main.cu src/check.cu src/siphash.cu src/braid.cu)

add_executable(test ${SOURCE})
target_link_libraries(
  test
  cuda
  cudadevrt
  cudart_static
  rt
  pthread
  dl)
set_property(TARGET test PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
