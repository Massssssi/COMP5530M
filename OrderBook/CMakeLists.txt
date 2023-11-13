project (OrderBook)
message("In orderbook subdirectory")
add_definitions(-std=c++17)

set(HEADER_FILES OrderBook.hpp)
set(SOURCE_FILES OrderBook.cpp)

message("adding orderbook library...")
add_library(OrderBook STATIC ${SOURCE_FILES} ${HEADER_FILES})