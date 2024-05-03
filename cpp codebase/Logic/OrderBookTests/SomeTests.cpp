#include <gtest/gtest.h>
#include <OrderBook.hpp>
#include <iostream>
#include <sqlite3.h>
#include <filesystem>
#include <string>
using namespace std;
static int lastProcessedId = 1;
OrderBook globalOrderBook;

namespace {
class OrderBookTests : public testing::Test {
 protected:
  sqlite3 *db;
  char *zErrMsg = nullptr;

  OrderBookTests() {}

  ~OrderBookTests() override {}

  string getProjectSourceDirectory() {
    std::string currFilePath = __FILE__;
    std::filesystem::path fullPath(currFilePath);
    cout << fullPath.parent_path().parent_path().string();
    return fullPath.parent_path().parent_path().string();
  }

  static int callback(void *notUsed, int argc, char **argv, char **azColName) {
    if (argc == 4) {
      int index = std::stoi(argv[0]);
      // Check if this row has already been processed
      if (index <= lastProcessedId) return 0;

      double price_level = std::stod(argv[1]);
      double new_quantity = std::stod(argv[2]);
      std::string id = argv[3];

      // Add the order to the OrderBook
      bool is_bid = id.find("_ask") == std::string::npos; // Assuming "_ask" is part of the ask IDs
      globalOrderBook.add_order(index, price_level, new_quantity, is_bid);

      // Update the last processed index
      lastProcessedId = index;
    }
    return 0;
  }

  void SetUp() override {
    string dbFilePath = getProjectSourceDirectory() + "/prime_orderbook.db";
    cout << dbFilePath.c_str();
    sqlite3_open(dbFilePath.c_str(), &db);
    std::string baseSQLStatement = "SELECT * FROM book";
    std::string modifiedSQLStatement =
        baseSQLStatement + " WHERE \"index\" > " + std::to_string(lastProcessedId) + " ORDER BY \"index\" ASC";
    sqlite3_exec(db, modifiedSQLStatement.c_str(), callback, nullptr, nullptr);
  }

  bool connectToDB(const string &dbFilePath) {
    int exit = sqlite3_open(dbFilePath.c_str(), &db);
    return exit == SQLITE_OK;
  }

  void TearDown() override {
  }
};

TEST_F(OrderBookTests, OrderBookEmptyByDefault) {
  OrderBook book;
  EXPECT_TRUE(book.is_empty());
}

// Tests that a connection is established in the database
TEST_F(OrderBookTests, CheckDBConnectionOK) {
  string dbFilePath = getProjectSourceDirectory() + "/prime_orderbook.db";
  EXPECT_TRUE(connectToDB(dbFilePath));
}

// Ensures that data is read from the DB correctly
TEST_F(OrderBookTests, ReadFromDB) {
  EXPECT_TRUE(db);
  EXPECT_FALSE(globalOrderBook.is_empty());
}

// Tests that we can read the correct amount of bids from the database
TEST_F(OrderBookTests, BidAmount) {
  EXPECT_TRUE(db);
  ASSERT_EQ(globalOrderBook.getBids().size(), 236);
}

// Tests that we read the correct amount of asks from the database
TEST_F(OrderBookTests, AsksAmount) {
  EXPECT_TRUE(db);
  ASSERT_EQ(globalOrderBook.getAsks().size(), 204);
}

// Tests that we can find the highest price level from the database
TEST_F(OrderBookTests, HighestAskPriceLevel) {
  EXPECT_TRUE(db);
  std::map asks = globalOrderBook.getAsks();
  double result = 0.0;
  for (auto const &x : asks) {
    if (x.first > result) {
      result = x.first;
    }
  }
  ASSERT_EQ(result, 2072.4);
}

// Tests that we can find the highest price level from the database
TEST_F(OrderBookTests, HighestBidPriceLevel) {
  EXPECT_TRUE(db);
  std::map asks = globalOrderBook.getBids();
  double result = 0.0;
  for (auto const &x : asks) {
    if (x.first > result) {
      result = x.first;
    }
  }
  ASSERT_EQ(result, 2038.1);
}

// Test additional bid and ask
TEST_F(OrderBookTests, AddBidAsk) {
  EXPECT_TRUE(db);
  globalOrderBook.add_order(500, 2040.45, 10, true);
  globalOrderBook.add_order(500, 2075.01, 10, false);

  std::map asks = globalOrderBook.getAsks();
  double result = 0.0;
  for (auto const &x : asks) {
    if (x.first > result) {
      result = x.first;
    }
  }
  ASSERT_EQ(result, 2075.01);
  ASSERT_EQ(globalOrderBook.getAsks().size(), 205);

  std::map bids = globalOrderBook.getBids();
  result = 0.0;
  for (auto const &y : bids) {
    if (y.first > result) {
      result = y.first;
    }
  }
  ASSERT_EQ(result, 2040.45);
  ASSERT_EQ(globalOrderBook.getBids().size(), 237);

}

// Test removal of bid and ask
TEST_F(OrderBookTests, RemoveBidAsk) {
  EXPECT_TRUE(db);
  globalOrderBook.add_order(500, 2040.45, 10, true);
  globalOrderBook.add_order(500, 2075.01, 10, false);

  std::map asks = globalOrderBook.getAsks();
  double result = 0.0;
  for (auto const &x : asks) {
    if (x.first > result) {
      result = x.first;
    }
  }
  ASSERT_EQ(result, 2075.01);

  std::map bids = globalOrderBook.getBids();
  result = 0.0;
  for (auto const &y : bids) {
    if (y.first > result) {
      result = y.first;
    }
  }
  ASSERT_EQ(result, 2040.45);

  globalOrderBook.remove_order(500, 2040.45, true);
  globalOrderBook.remove_order(500, 2075.01, false);

  result = 0.0;
  asks = globalOrderBook.getAsks();
  for (auto const &x : asks) {
    if (x.first > result)
      result = x.first;
  }
  ASSERT_EQ(result, 2072.4);

  result = 0.0;
  bids = globalOrderBook.getBids();
  for (auto const &y : bids) {
    if (y.first > result)
      result = y.first;
  }
  ASSERT_EQ(result, 2038.1);

}
// Test modification of bid and ask
TEST_F(OrderBookTests, ModifyBidAsk) {
  EXPECT_TRUE(db);
  globalOrderBook.add_order(500, 2040.45, 10, true);
  globalOrderBook.add_order(501, 2075.01, 10, false);

  std::map asks = globalOrderBook.getAsks();
  double result = 0.0;
  for (auto const &x : asks) {
    if (x.first > result)
      result = x.first;
  }
  ASSERT_EQ(result, 2075.01);

  std::map bids = globalOrderBook.getBids();
  result = 0.0;
  for (auto const &y : bids) {
    if (y.first > result) {
      result = y.first;
    }
  }
  ASSERT_EQ(result, 2040.45);

  globalOrderBook.modify_order(500, 2040.45, 2045.45, 20, true);
  globalOrderBook.modify_order(501, 2075.01, 2080.01, 20, false);

  result = 0.0;
  asks = globalOrderBook.getAsks();
  for (auto const &x : asks) {
    if (x.first > result) {
      result = x.first;
    }
  }
  ASSERT_EQ(result, 2080.01);

  result = 0.0;
  bids = globalOrderBook.getBids();
  for (auto const &y : bids) {
    if (y.first > result) {
      result = y.first;
    }
  }
  ASSERT_EQ(result, 2045.45);
}
}

