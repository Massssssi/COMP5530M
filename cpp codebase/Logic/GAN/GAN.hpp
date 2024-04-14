//
// Created by Daniel on 05/04/2024.
//

#ifndef GAN_HPP
#define GAN_HPP

#include "OrderBook.hpp"
#include "DBHandler.hpp"
#include "API.hpp"
#include <filesystem>
#include <cstdio>
#include <iostream>
#include <algorithm>

class Gan {
 private:
  static bool flag;

 public:
  static std::thread startGenerate;
  static std::atomic<bool> cancelRequested;
  static void updateOrderBook(float *values, OrderBook *globalOrderBook);
  static void generateQuantity(DBHandler *handler, OrderBook *globalOrderBook);
  static void startServer(DBHandler *handler, OrderBook *globalOrderBook);
  static std::string getProjectSourceDirectory();
};

#endif