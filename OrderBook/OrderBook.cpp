//
//  OrderBook.cpp
//  TradingEngine
//
//  Created by Shreyas Honnalli on 06/11/2023.
//

#include <iostream>
#include "OrderBook.hpp"

// Check if the order book is empty
bool OrderBook::is_empty() const {
    return bids.empty() && asks.empty();
}

// Add a bid (buy) order to the order book
void OrderBook::add_bid(int price, int amount) {
    add(price, amount, true);
}

// Add an ask (sell) order to the order book
void OrderBook::add_ask(int price, int amount) {
    add(price, amount, false);
}

// Add an order to the order book
void OrderBook::add(int price, int amount, bool bid) {
    if (bid)
        bids[price] += amount; // If it's a bid, add to the bids side
    else
        asks[price] += amount; // If it's an ask, add to the asks side
}

// Overload the << operator to print the order book
std::ostream& operator<<(std::ostream& os, const OrderBook& book) {
    if (book.is_empty()) {
        os << "ORDER BOOK EMPTY";
        return os;
    }

    // Print the asks side of the order book (from highest price to lowest)
    for (auto it = book.asks.rbegin(); it != book.asks.rend(); ++it)
        os << it->first << "\t" << it->second << std::endl;

    os << std::endl;

    // Print the bids side of the order book (from highest price to lowest)
    for (auto it = book.bids.rbegin(); it != book.bids.rend(); ++it)
        os << it->first << "\t" << it->second << std::endl;

    return os;
}

// Remove a bid/ask from the OrderBook
void OrderBook::remove(int price, int amount, bool bid) {
    // first is the price, second is the quantity
    auto& table = bid? bids: asks;
    auto it = table.find(price);

    // we found a price in the corresponding map
    if (it != table.end()){
        it->second-=amount;
        // Check if the amount is 0, if so then remove it from the order book

        if (it->second == 0)
            table.erase(it);

        // If it's less than 0, then something went wrong
        if (it->second < 0)
            std::cout << "Can't have negative quantity after removal.";
    }
}

// Remove a bid from the Order Book
void OrderBook::remove_bid(int price, int amount) {
    remove(price,amount,true);
}

// Remove an ask from the Order Book
void OrderBook::remove_ask(int price, int amount) {
    remove(price,amount, false);
}

// Returns the best bid & ask from the order book
OrderBook::BidAsk OrderBook::get_bid_ask() const {
    BidAsk result;

    // Gets the bid from the map
    auto best_bid = bids.rbegin();
    if (best_bid != bids.rend())
        // store the reference
        result.bid = *best_bid;

    auto best_ask = asks.begin();
    if (best_ask != asks.end())
        result.ask = *best_ask;

    return result;
}

std::ostream &operator<<(std::ostream &os, const OrderBook::BidAsk &ba) {
    auto print = [&](const OrderBook::BidAsk::Entry& e, const std::string & text){
        if (e){
            auto value = e.value();
            os << value.second << text << "s @" << value.first;
        }else{
            os << "NO" << text;
        }
    };
    print(ba.bid, "bid");
    print(ba.ask, "ask");
    return os;
}

// Calculates the spread from the order book
std::experimental::optional<int> OrderBook::BidAsk::spread() const {
    std::experimental::optional<int> result;
    // If there are bids & asks, then calculate the spread
    if (bid && ask)
        result = ask.value().first - bid.value().first;
    return result;
}
