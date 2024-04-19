#
# Created by Daniel on 29/01/2024.
#
cmake_minimum_required(VERSION 3.18)
project(ABM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_FILES ABM.hpp)
set(SOURCE_FILES ABM.cpp)

add_library(ABM STATIC ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(ABM PRIVATE "${TORCH_LIBRARIES}" OrderBook API DBHandler)
set_property(TARGET ABM PROPERTY CXX_STANDARD 17)