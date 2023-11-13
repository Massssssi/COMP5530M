#include <gtest/gtest.h>
#include <OrderBook.hpp>
TEST(OrderBookTests, OrderBookEmptyByDefault)
{
    OrderBook book;
    EXPECT_TRUE(book.is_empty());
}

TEST(OrderBookTests, OrderAddition)
{
    OrderBook book;
    book.add_bid(123,700);
    auto bidask = book.get_bid_ask();
    EXPECT_TRUE(bidask.bid.has_value());
    auto bid = bidask.bid.value();
    EXPECT_EQ(123, bid.first);
    EXPECT_EQ(700,bid.second);
}

// also test removal for price levels which don't exist yet
TEST(OrderBookTests, OrderRemoval)
{
    OrderBook book;
    book.add_bid(123,700);
    book.add_ask(124,500);

    book.remove_ask(124,300);
    book.remove_bid(123,600);
    auto bidask = book.get_bid_ask();

    EXPECT_TRUE(bidask.bid.has_value());
    EXPECT_TRUE(bidask.ask.has_value());

    // Get the bid and ask from the Order Book (they are optional)
    auto bid = bidask.bid.value();
    auto ask = bidask.ask.value();

    EXPECT_EQ(123, bid.first);
    EXPECT_EQ(700-600, bid.second);

    EXPECT_EQ(124, ask.first);
    EXPECT_EQ(500-300, ask.second);
}

// also test removal for price levels which don't exist yet
TEST(OrderBookTests, CalculateSpread)
{
    OrderBook book;
    book.add_bid(123,700);
    book.add_ask(124,500);

    auto diff = book.get_bid_ask().spread();

    EXPECT_TRUE(diff.has_value());
    EXPECT_EQ(1, diff);
}

// Tests that we can't calculate the spread if we don't have a bid/ask already
TEST(OrderBookTests, CalculateSpreadInvalid)
{
    OrderBook book;
    auto diff = book.get_bid_ask().spread();
    EXPECT_FALSE(diff.has_value());
}