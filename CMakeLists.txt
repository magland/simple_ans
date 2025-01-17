cmake_minimum_required(VERSION 3.14)
project(simple_ans VERSION 0.1.0 LANGUAGES CXX)

# Find pybind11
find_package(pybind11 REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add source files
set(SOURCES
    simple_ans/cpp/simple_ans.cpp
    simple_ans_bind.cpp
)

# Add header files
set(HEADERS
    simple_ans/cpp/simple_ans.hpp
)

# Create Python module
pybind11_add_module(_simple_ans ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(_simple_ans
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/simple_ans/cpp
)

# Set compiler flags
if(MSVC)
    target_compile_options(_simple_ans PRIVATE /W4)
else()
    target_compile_options(_simple_ans PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install rules
include(GNUInstallDirs)
install(TARGETS _simple_ans
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simple_ans
)
