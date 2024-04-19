project(HeaderWidget)
add_definitions(-std=c++14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(HeaderWidget STATIC HeaderWidget.cpp HeaderWidget.hpp)
target_link_libraries(HeaderWidget GAN ABM Qt${QT_VERSION_MAJOR}::Widgets)