cmake_minimum_required(VERSION 3.16)
project(top-stencil LANGUAGES C DESCRIPTION "3D stencil with MPI")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_C_STANDARD 17)

# set(CMAKE_BUILD_TYPE Debug)

# if(CMAKE_BUILD_TYPE EQUAL Debug)
#     add_compile_options(-Wall -Werror -Wconversion)
# endif()

# if(CMAKE_COMPILER_IS_GNUCXX)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -O3 -g")
# elseif(CMAKE_COMPILER_IS_GNUC)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic -O3 -g")
# endif()

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_subdirectory(src)

add_executable(top-stencil src/main.c)
target_include_directories(top-stencil PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(top-stencil PRIVATE stencil::stencil stencil::utils ${MPI_C_LIBRARIES})
