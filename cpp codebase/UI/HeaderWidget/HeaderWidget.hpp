#ifndef HeaderWidget_hpp
#define HeaderWidget_hpp

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QStringList>
#include <QTimer>
#include "DBHandler.hpp"
#include "OrderBook.hpp"

#include <stdio.h>
#include <iostream>
#include <QApplication>
#include <list>
#include <thread>
#include <chrono>
#include "GAN.hpp"

#include <QTimer>
#include <QStringList>

class HeaderWidget : public QWidget {
    Q_OBJECT

private:
    DBHandler *handler;
    OrderBook *orderBook;

    QLabel *titleLabel, *imageLabel, *newsTicker;
    QPushButton *startSimulation, *cancelSimulation;
    QComboBox *modelType;

    QStringList newsItems;
    int currentNewsIndex = 0;
    QTimer *newsTimer;

    void initialiseLayoutAndAddButtons();
    void setupButtons();
    void setupLogoAndCompany();
    void setupModelType();
    void setupNewsTicker();
    void connectButtons();

public:
    HeaderWidget(DBHandler *handler, OrderBook *orderBook, QWidget *parent = nullptr);
    virtual ~HeaderWidget();

    
public slots:
    void startSim();
    void cancelSim();
    void updateNewsTicker();
};

#endif // HeaderWidget_hpp
