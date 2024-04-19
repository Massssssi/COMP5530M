//
// Created by Daniel on 29/01/2024.
//
#include "API.hpp"
#include <nlohmann/json.hpp>

/*
 * Endpoints
 * GET: /pos -> returns the users current positions -> returns a list of positions
 *
 * GET: /pnl -> return the current balance, positions, price and PnL to the user
 *
 * POST: /submit -> attempts to fulfill the users order in the orderbook
 *
 * DELETE: /cancel -> this removes a user's order if it hasn't been fulfilled
 * returns corresponding status code if successful or not
 *
 */

// Variables
httplib::Server API::s;
std::random_device rd;
std::default_random_engine e(rd());
std::map<int, std::map<std::string, double> > API::orderHistory;
float API::price = 2500;
float API::balance = 1000000.00;
float API::quantity = 0.00;
float API::pnl = 0.00;

void API::setPrice(float p) { price = p; }

float API::getPrice() { return price; }

float API::getPnl()  {
  updatePnL();
  return pnl;
}

void API::updatePnL() { pnl = balance + (float) (quantity * price); }

float API::getQuantity(){ return quantity; }

// Since it's a bid, we subtract from balance
void API::IOC_bid(float targetPrice, float targetQuantity, OrderBook &o) {
  o.OB_mutex.lock();
  std::map<double, OrderList> asks = o.getAsks();
  bool flag = false;

  for (const auto &ob : asks) {
    if (flag) { break; }
    std::list<Order> ordersForPL = ob.second;
    for (Order ob2 : ordersForPL) {
      if (targetPrice < ob2.price or balance == 0) {
        flag = true;
        break;
      }
      float q = std::min((float) ob2.quantity, targetQuantity);

      // Check that we have enough balance to buy for the user
      // then only buy enough for the quantity
      if (q * ob2.price > balance) { q = (float) (balance / ob2.price); }
      quantity += q;
      balance -= (float) (q * ob2.price);
      targetQuantity -= q;
    }
  }
  o.OB_mutex.unlock();
}

// Since it's a bid, we subtract from balance
void API::FOK_bid(float targetPrice, float targetQuantity, OrderBook &o) {
  o.OB_mutex.lock();
  std::map<double, OrderList> asks = o.getAsks();
  bool flag = false;

  for (const auto &ob : asks) {
    if (flag) { break; }

    std::list<Order> ordersForPL = ob.second;
    for (Order ob2 : ordersForPL) {
      // 1. Check that we can match the current bid with this price
      // 2. we have enough balance to buy since it's FOK
      // 3. We have enough quantity remaining since it's FOK
      if (targetPrice < ob2.price or ob2.quantity * ob2.price > balance or ob2.quantity > targetQuantity) {
        flag = true;
        break;
      }

      quantity += (float) ob2.quantity;
      balance -= (float) (ob2.quantity * ob2.price);
      targetQuantity -= (float) ob2.quantity;
    }
  }
  o.OB_mutex.unlock();
}

// Since it's an ask, we add to balance
void API::IOC_ask(float targetPrice, float targetQuantity, OrderBook &o) {
  o.OB_mutex.lock();
  bool flag = false;

  std::map<double, OrderList> bids = o.getBids();
  for (auto iter = bids.rbegin(); iter != bids.rend(); ++iter) {
    if (flag) { break; }
    for (auto const &i : iter->second) {
      if (i.price < targetPrice or quantity == 0) {
        flag = true;
        break;
      }

      float q = std::min((float) i.quantity, targetQuantity);

      // Check that we have enough balance to buy for the user
      if (q > quantity) { q = quantity; }

      balance += (float) (q * i.price);
      quantity -= q;
      targetQuantity -= q;
    }
  }
  o.OB_mutex.unlock();
}

// Since it's an ask, we add to balance
void API::FOK_ask(float targetPrice, float targetQuantity, OrderBook &o) {
  o.OB_mutex.lock();
  bool flag = false;
  std::map<double, OrderList> bids = o.getBids();
  for (auto iter = bids.rbegin(); iter != bids.rend(); ++iter) {
    if (flag){ break; }

    for (auto const &i : iter->second) {

      // Check that we have enough balance to buy for the user
      if (i.price < targetPrice or quantity == 0 or i.quantity > quantity or i.quantity > targetQuantity) {
        flag=true;
        break;
      }

      balance += (float) (i.quantity * i.price);
      quantity -= (float) i.quantity;
      targetQuantity -= (float) i.quantity;
    }
  }
  o.OB_mutex.unlock();
}

// Gets the orders that the user has submitted
void API::getOrders(OrderBook &o, DBHandler &handler) {
  s.Get("/orders", [&o](const httplib::Request &req, httplib::Response &res) {
    try {
      nlohmann::json response_json = nlohmann::json::array();
      for (const auto &entry : orderHistory) {
        nlohmann::json item;

        item["id"] = entry.first;
        item["price"] = entry.second.at("price");
        item["bidAsk"] = entry.second.at("bidAsk");
        item["quantity"] = entry.second.at("quantity");
        response_json.push_back(item);
      }
      res.set_content(response_json.dump(), "application/json");
      return;

    } catch (const std::exception &e) {
      std::cerr << "Unable to find one of the keys: " << e.what() << std::endl;
    }
    res.status = 400;
    res.set_content("Unable to find one of the keys", "text/plain");
  });
}

void API::getPnL(OrderBook &o, DBHandler &handler) {
  s.Get("/pnl", [&o](const httplib::Request &req, httplib::Response &res) {
    try {
      API::updatePnL();
      nlohmann::json item;
      item["PnL"] = pnl;
      item["balance"] = balance;
      item["quantity"] = quantity;
      item["price"] = price;
      nlohmann::json response_json = nlohmann::json::array();
      response_json.push_back(item);
      res.set_content(response_json.dump(), "application/json");
      return;

    } catch (const std::exception &e) {
      std::cerr << "Unable to get PnL: " << e.what() << std::endl;
    }
    res.status = 400;
    res.set_content("Unable to calculate PnL", "text/plain");
  });
}

// the thread to read from the db updates the map each time. Account for this when removing from the DB
// if it's no longer in the order book, then assume it's been fulfilled
void API::deleteOrder(OrderBook &o, DBHandler &handler) {
  s.Delete("/delete", [&o](const httplib::Request &req, httplib::Response &res) {

    if (!req.has_header("Content-Type") || req.get_header_value("Content-Type") != "application/json") {
      res.status = 400;
      res.set_content("Invalid request structure", "text/plain");
      return;
    }

    try {
      auto json_data = nlohmann::json::parse(req.body);
      if (!json_data.contains("price") || !json_data.contains("bidAsk") || !json_data.contains("id")) {
        res.status = 400;
        res.set_content("Invalid parameters in request body", "text/plain");
        return;
      }

      int id = json_data["id"].get<int>();
      bool bidAsk = json_data["bidAsk"].get<bool>();
      double price = json_data["price"].get<double>();

      // Check whether the ID is in the orderHistory
      if (orderHistory.find(id) == orderHistory.end()) {
        res.status = 400;
        res.set_content("Order ID doesn't exist!", "text/plain");
        return;
      }

      // If order removed, then give SUCCESS message
      orderHistory.erase(id);

      if (o.remove_order(id, price, bidAsk)) {
        res.set_content("Order removed from the OrderBook", "text/plain");
      } else {
        // Order was fulfilled, so reflect this within the OrderbookHistory
        // will change this once we start working on integrating with trading algorithm
        // essentially, get the price the user submitted, and based on the quantity, bidAsk and price
        // either add/remove from their balance

        // will also need to check the users current positions every second & check whether the limit
        // order can be executed. future feature.
        res.set_content("Order has already been fulfilled!", "text/plain");
      }

    } catch (const std::exception &e) {
      res.status = 400;
      std::cerr << "JSON parsing error: " << e.what() << std::endl;
      res.set_content("Invalid JSON structure", "text/plain");
    }
  });
}

// Check for exisitng quantity first before adding
void API::submitOrder(OrderBook &o, DBHandler &handler) {

  s.Post("/submit", [&o, &handler](const httplib::Request &req, httplib::Response &res) {
    if (!req.has_header("Content-Type") || req.get_header_value("Content-Type") != "application/json") {
      res.status = 400;
      res.set_content("Invalid request structure", "text/plain");
      return;
    }

    try {
      auto json_data = nlohmann::json::parse(req.body);
      if (!json_data.contains("price") || !json_data.contains("bidAsk") || !json_data.contains("quantity")
          || !json_data.contains("orderType")) {
        res.status = 400;
        res.set_content("Invalid parameters in request body", "text/plain");
        return;
      }

      int random = e();
      bool bidAsk = json_data["bidAsk"].get<bool>();
      float price = json_data["price"].get<float>();
      float quantity = json_data["quantity"].get<float>();
      std::string type = json_data["orderType"].get<std::string>();

      if (type == "IOC") {
        if (bidAsk) API::IOC_bid(price, quantity, o);
        else API::IOC_ask(price, quantity, o);
      } else if (type == "FOK") {
        if (bidAsk) API::FOK_bid(price, quantity, o);
        else API::FOK_ask(price, quantity, o);
      } else {
        res.set_content("Invalid Order Type!", "text/plain");
        return;
      }

      std::map<std::string, double> new_entry;
      new_entry["price"] = price;
      new_entry["quantity"] = quantity;
      new_entry["bidAsk"] = bidAsk;
      orderHistory[random] = new_entry;
      handler.emitSuccessfulUpdate();
      res.set_content("Order Submitted", "text/plain");
    } catch (const std::exception &e) {
      res.status = 400;
      res.set_content("Error parsing JSON", "text/plain");
    }
  });
}

void API::startServer(OrderBook &o, DBHandler &handler) {
  submitOrder(o, handler);
  //deleteOrder(o, handler);
  getOrders(o, handler);
  getPnL(o, handler);
  s.listen("localhost", 8080);
}

float API::getBalance() { return balance; }


