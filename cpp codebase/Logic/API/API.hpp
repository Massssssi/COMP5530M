//
// Created by Daniel on 29/01/2024.
//
#ifndef API_hpp
#define API_hpp

#include <iostream>
#include <httplib.h>
#include "OrderBook.hpp"
#include "DBHandler.hpp"

class API {
private:
    static httplib::Server s;
    static std::map< int, std::map<std::string, double> > orderHistory;
    static float price;
    static float balance;
    static float pnl;
    static float quantity;

public:
    static void getOrders(OrderBook &o, DBHandler &handler);
    static void deleteOrder(OrderBook &o, DBHandler &handler);
    static void submitOrder(OrderBook &o, DBHandler &handler);
    static void startServer(OrderBook &o, DBHandler &handler);
    static void getPnL(OrderBook &o, DBHandler &handler);
    static void updatePnL();
    static void setPrice(float p);
    static float getPrice();
    static float getBalance();
    static float getQuantity();
    static float getPnl();
    static void IOC_bid(float targetPrice, float targetQuantity, OrderBook &o);
    static void IOC_ask(float targetPrice, float targetQuantity, OrderBook &o);
    static void FOK_ask(float targetPrice, float targetQuantity, OrderBook &o);
    static void FOK_bid(float targetPrice, float targetQuantity, OrderBook &o);

    };

#endif
