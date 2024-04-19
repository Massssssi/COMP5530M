cmake_minimum_required(VERSION 3.18)
project(OrderBookDemo)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        httplib
        URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.0.zip
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)

FetchContent_Declare(
        libtorch
        # URL https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu118.zip
        URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcpu.zip
)
FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(httplib)
FetchContent_GetProperties(libtorch)

if(NOT libtorch_POPULATED)
    FetchContent_Populate(libtorch)
    set(Torch_DIR ${libtorch_SOURCE_DIR}/share/cmake/Torch)
endif()
message(STATUS "Torch_DIR is set to ${Torch_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(SQLite3 REQUIRED)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Charts)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Charts)
message("Qt${QT_VERSION_MAJOR}")

include_directories(${httplib_SOURCE_DIR})
include_directories(${torch_SOURCE_DIR})
include_directories("${CMAKE_SOURCE_DIR}/Logic/OrderBook")
include_directories("${CMAKE_SOURCE_DIR}/Logic/API")
include_directories("${CMAKE_SOURCE_DIR}/Logic/ABM")
include_directories("${CMAKE_SOURCE_DIR}/Logic/GAN")
include_directories("${CMAKE_SOURCE_DIR}/DummyInput/DBHandler")
include_directories("${CMAKE_SOURCE_DIR}/UI/OrderBookWidget")
include_directories("${CMAKE_SOURCE_DIR}/UI/CandlestickWidget")
include_directories("${CMAKE_SOURCE_DIR}/UI/HeaderWidget")
include_directories("${CMAKE_SOURCE_DIR}/UI/CentralWidget")
include_directories(${torch_INCLUDE_DIRS})

message("Going into subdirectories...")
add_subdirectory(Logic/OrderBook)
add_subdirectory(Logic/API)
add_subdirectory(Logic/GAN)
add_subdirectory(Logic/ABM)
add_subdirectory(DummyInput/DBHandler)
add_subdirectory(Logic/OrderBookTests)

add_subdirectory(UI/OrderBookWidget)
add_subdirectory(UI/CandlestickWidget)
add_subdirectory(UI/HeaderWidget)

add_subdirectory(UI/CentralWidget)
message("added subdirectories")

set(SOURCE_FILES main.cpp)
add_executable(OrderBookDemo ${SOURCE_FILES})

target_link_libraries(OrderBookDemo PUBLIC ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json httplib::httplib OrderBook API DBHandler GAN ABM OrderBookWidget CandlestickWidget CentralWidget HeaderWidget Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Charts SQLite::SQLite3)
set_property(TARGET OrderBookDemo PROPERTY CXX_STANDARD 17)