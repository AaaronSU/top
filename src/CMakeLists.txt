cmake_minimum_required(VERSION 3.10)
project(TOP-project)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_library(stencil SHARED stencil/config.c stencil/comm_handler.c stencil/mesh.c stencil/init.c stencil/solve.c)
target_include_directories(stencil PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${MPI_C_INCLUDE_PATH}
)
target_link_libraries(stencil PUBLIC 
    m
    ${MPI_C_LIBRARIES}
    OpenMP::OpenMP_C
)

add_library(utils SHARED chrono.c)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_library(stencil::stencil ALIAS stencil)
add_library(stencil::utils ALIAS utils)

add_executable(my_executable main.c)

target_link_libraries(my_executable stencil::stencil stencil::utils)

if(OPENMP_FOUND AND MPI_FOUND)
    # Add AVX compiler flags
    if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # For GCC/G++
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -O3 -Wall -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough \
            -Werror=format-security \
            -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 \
            -D_GLIBCXX_ASSERTIONS \
            -fstack-clash-protection -fstack-protector-strong \
            -Wl,-z,nodlopen -Wl,-z,noexecstack \
            -Wl,-z,relro -Wl,-z,now")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -O3")
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # For Clang/Clang++
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -O3 -Wall -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough \
            -Werror=format-security \
            -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 \
            -D_GLIBCXX_ASSERTIONS \
            -fstack-clash-protection -fstack-protector-strong \
            -Wl,-z,nodlopen -Wl,-z,noexecstack \
            -Wl,-z,relro -Wl,-z,now")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -O3")
    elseif(MSVC)
        # MSVC doesn't use the same flag, but AVX support is enabled by default
    endif()
    target_compile_options(stencil PRIVATE ${OpenMP_C_FLAGS} -mavx -fopenmp)
    target_compile_options(utils PRIVATE ${OpenMP_C_FLAGS})
    target_compile_options(my_executable PRIVATE ${OpenMP_C_FLAGS})
endif()