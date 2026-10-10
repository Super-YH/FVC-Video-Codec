cmake_minimum_required(VERSION 3.16)
project(fvc CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  # 復号の決定性 (§13.3): 浮動小数点の縮約 (FMA 融合) と高速数学を禁止し、IEEE 倍精度の演算順を固定する
  add_compile_options(-ffp-contract=off -fno-fast-math)
endif()

add_library(fvc src/color.cpp src/pqmf.cpp src/quant.cpp src/transform.cpp src/entropy.cpp
  src/frame.cpp src/codec.cpp src/block_coder.cpp
  src/inter.cpp src/dictionary.cpp src/shapes.cpp src/loop_filter.cpp src/alf.cpp src/cdef.cpp src/gs_coder.cpp)
target_include_directories(fvc PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(fvc PUBLIC Threads::Threads)

enable_testing()
add_executable(fvc_tests tests/test_main.cpp)
target_link_libraries(fvc_tests PRIVATE fvc)
add_test(NAME fvc_tests COMMAND fvc_tests)

add_executable(fvc_cli tools/fvc.cpp)
target_link_libraries(fvc_cli PRIVATE fvc)
set_target_properties(fvc_cli PROPERTIES OUTPUT_NAME fvc)
add_executable(bandviz tools/bandviz.cpp)
target_link_libraries(bandviz PRIVATE fvc)
