project(OrderBookTests)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
add_definitions(-std=c++17)
find_package(SQLite3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GoogleTest)

message("adding runtests executable")
add_executable(RunTests SomeTests.cpp)
message("Linking libraries")
target_link_libraries(RunTests gtest gtest_main OrderBook DBHandler SQLite::SQLite3)


gtest_discover_tests(RunTests DISCOVERY_MODE PRE_TEST)
