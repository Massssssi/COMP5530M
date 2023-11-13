cmake_minimum_required(VERSION 3.3)
project(OrderBookDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(SOURCE_FILES main.cpp)
add_executable(OrderBookDemo ${SOURCE_FILES})


include_directories(OrderBook)
message("Going into Orderbook subdirectory...")
add_subdirectory(OrderBook)
message("Going into OrderbookTests subdirectory...")
add_subdirectory(OrderBookTests)


target_link_libraries(OrderBookDemo OrderBook)