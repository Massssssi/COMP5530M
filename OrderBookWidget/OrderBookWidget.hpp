//
//  OrderBookWidget.hpp
//  DBHandler
//
//  Created by Shreyas Honnalli on 11/12/2023.
//

#ifndef OrderBookWidget_hpp
#define OrderBookWidget_hpp

#include <stdio.h>
#include <QApplication>
#include <QWidget>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QLabel>
#include "OrderBook.hpp"
#include "DBHandler.hpp"
#include <mutex>
#include <list>
#include <thread>
#include <chrono>

class OrderBookWidget : public QWidget
{
    Q_OBJECT
private:
    OrderBook *orderBook;

    QLabel *bidsLabel; // Label for Bids table
    QTableWidget *bidsTableWidget;
    QLabel *asksLabel; // Label for Asks table
    QTableWidget *asksTableWidget;
    
    
    void updateTable(std::vector<Order>& newOrders, QTableWidget* tableWidget);
    std::vector<Order> getNewOrdersFromOrderbook(bool is_bid);

public:
    OrderBookWidget(DBHandler *handler, OrderBook *orderBook);
    ~OrderBookWidget();
    void initializeTable(QTableWidget *tableWidget, const QStringList &headers);

public slots:
    // Slot to update the table when the order book is updated
    void updateBothTables();
};

#endif /* OrderBookWidget_hpp */
