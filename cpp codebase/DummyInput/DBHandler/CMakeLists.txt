cmake_minimum_required(VERSION 3.5)
project (DBHandler)
message("In DBHandler subdirectory")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_file(SQLITE3_DLL_DEBUG NAMES sqlite3d.dll PATHS ...)
set(HEADER_FILES DBHandler.hpp)
set(SOURCE_FILES DBHandler.cpp)

include_directories("${CMAKE_SOURCE_DIR}/Logic/OrderBook")

message("adding DBHandler library...")
add_library(DBHandler STATIC ${SOURCE_FILES} ${HEADER_FILES})
include_directories(${SQLite3_INCLUDE_DIRS})
target_include_directories(DBHandler PUBLIC)
target_link_libraries(DBHandler PUBLIC Qt${QT_VERSION_MAJOR}::Widgets SQLite::SQLite3)