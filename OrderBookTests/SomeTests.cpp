#include <gtest/gtest.h>
#include <OrderBook.hpp>
TEST(OrderBookTets, OrderBookEmptyByDefault)
{
    OrderBook book;
    EXPECT_TRUE(book.is_empty());
}

TEST(OrderBookTets, OrderAddition)
{
    OrderBook book;
    book.add_bid(123,700);
    auto bidask = book.get_bid_ask();
    EXPECT_TRUE(bidask.bid);
    auto bid = bidask.bid.value();
    EXPECT_EQ(123, bid.first);
    EXPECT_EQ(700,bid.second);
}

// also test removal for price levels which don't exist yet
TEST(OrderBookTets, OrderRemoval)
{
    OrderBook book;
    book.add_bid(123,700);
    book.add_ask(124,500);

    book.remove_ask(124,300);
    book.remove_bid(123,600);
    auto bidask = book.get_bid_ask();

    EXPECT_TRUE(bidask.bid);
    EXPECT_TRUE(bidask.ask);

    // Get the bid and ask from the Order Book (they are optional)
    auto bid = bidask.bid.value();
    auto ask = bidask.ask.value();

    EXPECT_EQ(123, bid.first);
    EXPECT_EQ(700-600, bid.second);

    EXPECT_EQ(124, ask.first);
    EXPECT_EQ(500-300, ask.second);
}

// also test removal for price levels which don't exist yet
TEST(OrderBookTets, CalculateSpread)
{
    OrderBook book;
    book.add_bid(123,700);
    book.add_ask(124,500);

    auto diff = book.get_bid_ask().spread();

    EXPECT_TRUE(diff);
    EXPECT_EQ(1, diff);
}