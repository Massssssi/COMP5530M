//
//  OrderBookWidget.cpp
//
//  Created by Shreyas Honnalli on 11/12/2023.
//

#include "OrderBookWidget.hpp"
#include <nlohmann/json.hpp>

OrderBookWidget::OrderBookWidget(DBHandler *handler, OrderBook *orderBookParam) : orderBook(orderBookParam) {
  setupObjects();
  initializeTable(asksTableWidget, {"ID", "Price", "Quantity"});
  initializeTable(bidsTableWidget, {"ID", "Price", "Quantity"});
  addToLayout();

  this->setLayout(mainLayout);
  setupSignalsSlots(handler);
  // Apply a metallic silver background to the entire widget
  this->setStyleSheet("background-color: rgba(48, 58, 46, 30);");
  // Adjust the RGBA values for desired transparency and shade
}

void OrderBookWidget::addToLayout() {
  mainLayout->addWidget(asksLabel);
  mainLayout->addWidget(asksTableWidget);
  mainLayout->addWidget(bidsLabel);
  mainLayout->addWidget(bidsTableWidget);
  mainLayout->addWidget(currentQuantity);
  mainLayout->addWidget(currentBalance);
  mainLayout->addWidget(pnl);

  priceLayout->addWidget(price);
  priceLayout->addWidget(selectPrice);

  quantityLayout->addWidget(quantity);
  quantityLayout->addWidget(selectAllQuantity);

  mainLayout->addLayout(priceLayout);
  mainLayout->addLayout(quantityLayout);

  mainLayout->addWidget(orderType);
  mainLayout->addWidget(bidAsk);
  mainLayout->addWidget(apiButton);
  mainLayout->addWidget(apiResponse);
  mainLayout->addWidget(apiResponse);
}

void OrderBookWidget::setupObjects() {
  mainLayout = new QVBoxLayout;
  quantityLayout = new QHBoxLayout;
  priceLayout = new QHBoxLayout;
  QString whiteTextStyle = "color: white;";
  QString baseFontStyle = "color: white; font-family: 'Orbitron', sans-serif; font-size: 12pt;";

  bidsLabel = new QLabel;
  asksLabel = new QLabel;
  asksTableWidget = new QTableWidget;
  bidsTableWidget = new QTableWidget;

  // Applying white text style to labels
  bidsLabel->setText("Bid Prices");
  asksLabel->setText("Ask Prices");
  bidsLabel->setStyleSheet(baseFontStyle);
  asksLabel->setStyleSheet(baseFontStyle);

  bidsLabel->setText("Bid Prices");
  asksLabel->setText("Ask Prices");

  apiButton = new QPushButton("Submit Order", this);
  apiButton->setGeometry(QRect(QPoint(100, 100), QSize(200, 50)));

  selectPrice = new QPushButton("Current Price", this);
  selectPrice->setGeometry(QRect(QPoint(100, 100), QSize(200, 50)));

  QString buttonStyle = "QPushButton { " + baseFontStyle + "background-color: #333; border: 1px solid #555; }"
                                                           "QPushButton:hover { background-color: #555; }";

  selectAllQuantity = new QPushButton("Max Quantity", this);
  selectAllQuantity->setGeometry(QRect(QPoint(100, 100), QSize(200, 50)));

  // Applying style to buttons
  apiButton->setStyleSheet(buttonStyle);
  selectPrice->setStyleSheet(buttonStyle);
  selectAllQuantity->setStyleSheet(buttonStyle);

  currentBalance = new QLabel(this);
  currentBalance->setText("Current Balance = 100000");
  currentBalance->setGeometry(QRect(10, 10, 200, 20));

  currentQuantity = new QLabel(this);
  currentQuantity->setText("Current Quantity = 0");
  currentQuantity->setGeometry(QRect(50, 50, 200, 20));

  apiResponse = new QLabel(this);
  apiResponse->setGeometry(QRect(10, 10, 200, 20));

  pnl = new QLabel(this);
  pnl->setText("Overall Position = 0");
  pnl->setGeometry(QRect(100, 100, 200, 20));

  // Applying text style to labels
  currentBalance->setStyleSheet(baseFontStyle);
  currentQuantity->setStyleSheet(baseFontStyle);
  pnl->setStyleSheet(baseFontStyle);
  apiResponse->setStyleSheet(baseFontStyle);

  QString comboBoxStyle = "QComboBox { " + baseFontStyle + "background-color: #333; border: 1px solid #555; }"
                                                           "QComboBox::drop-down { border-left: 1px solid #555; }";

  orderType = new QComboBox(this);
  orderType->addItem("Immediate or Cancel (IOC)");
  orderType->addItem("Fill or Kill (FOK)");
  orderType->setGeometry(QRect(QPoint(200, 50), QSize(200, 40)));

  bidAsk = new QComboBox(this);
  bidAsk->addItem("Buy");
  bidAsk->addItem("Sell");
  bidAsk->setGeometry(QRect(QPoint(200, 50), QSize(200, 40)));

  // Applying style to Combo Boxes
  orderType->setStyleSheet(comboBoxStyle);
  bidAsk->setStyleSheet(comboBoxStyle);

  price = new QLineEdit(this);
  price->setPlaceholderText("Price");
  price->setGeometry(QRect(QPoint(100, 50), QSize(200, 40)));
  price->setValidator(new QDoubleValidator(price));

  quantity = new QLineEdit(this);
  quantity->setPlaceholderText("Quantity");
  quantity->setGeometry(QRect(QPoint(200, 50), QSize(200, 40)));
  quantity->setValidator(new QDoubleValidator(quantity));

  // Applying white text style to Line Edits
  price->setStyleSheet("QLineEdit { color: white; }");
  quantity->setStyleSheet("QLineEdit { color: white; }");
}

void OrderBookWidget::setupSignalsSlots(DBHandler *handler) {
  connect(apiButton, &QPushButton::clicked, this, &OrderBookWidget::invokeAPI);
  connect(handler, &DBHandler::orderBookUpdated, this, &OrderBookWidget::updateBothTables);
  connect(handler, &DBHandler::orderBookUpdated, this, &OrderBookWidget::updateBalance);
  connect(handler, &DBHandler::orderBookUpdated, this, &OrderBookWidget::updateQuantity);
  connect(handler, &DBHandler::orderBookUpdated, this, &OrderBookWidget::updatePnl);
  connect(selectAllQuantity, &QPushButton::clicked, this, &OrderBookWidget::setQuantity);
  connect(selectPrice, &QPushButton::clicked, this, &OrderBookWidget::setPrice);
}

void OrderBookWidget::initializeTable(QTableWidget *tableWidget, const QStringList &headers) {
  tableWidget->verticalHeader()->setVisible(false);
  tableWidget->setRowCount(0);
  tableWidget->setColumnCount(headers.size());
  tableWidget->setHorizontalHeaderLabels(headers);
  tableWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  tableWidget->setFixedHeight(118);

  tableWidget->horizontalHeader()->setStretchLastSection(true);
  tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

  // Updated font style
  QString futuristicFontStyle = "QTableWidget { font-family: 'Orbitron', sans-serif; font-size: 12pt; color: white; }"
                                "QHeaderView::section { font-family: 'Orbitron', "
                                "sans-serif; font-size: 12pt; color: white; background-color: #1e1e1e; }";

  tableWidget->setStyleSheet(futuristicFontStyle);
}

void OrderBookWidget::invokeAPI() {
  httplib::Client cli("localhost:8080");

  nlohmann::json body;
  body["price"] = price->text().toFloat();
  body["quantity"] = quantity->text().toFloat();

  if (orderType->currentText().toStdString() == "Immediate or Cancel (IOC)") {
    body["orderType"] = "IOC";
  } else { body["orderType"] = "FOK"; }

  if (bidAsk->currentText().toStdString() == "Buy") { body["bidAsk"] = true; } else { body["bidAsk"] = false; }

  auto res = cli.Post("/submit", body.dump(), "application/json");

  apiResponse->setText(res->body.c_str());
  QPalette sample_palette;

  if (res && res->status == 200) {
    sample_palette.setColor(QPalette::WindowText, Qt::darkGreen);
    apiResponse->setPalette(sample_palette);
    std::cout << "Response from API: " << res->body.c_str() << std::endl;
  } else {
    std::cout << "Failed to call API" << std::endl;
    sample_palette.setColor(QPalette::WindowText, Qt::darkRed);
    apiResponse->setPalette(sample_palette);
  }
}

OrderBookWidget::~OrderBookWidget() {}

void OrderBookWidget::updateTable(std::vector<Order> &newOrders, QTableWidget *tableWidget) {
  tableWidget->clearContents();
  tableWidget->setRowCount((int) newOrders.size());

  for (int i = 0; i < newOrders.size(); ++i) {
    QTableWidgetItem *idItem = new QTableWidgetItem(QString::number(newOrders[i].id));
    QTableWidgetItem *priceItem = new QTableWidgetItem(QString::number(newOrders[i].price));
    QTableWidgetItem *quantityItem = new QTableWidgetItem(QString::number(newOrders[i].quantity));
    tableWidget->setItem(i, 0, idItem);
    tableWidget->setItem(i, 1, priceItem);
    tableWidget->setItem(i, 2, quantityItem);
  }
  addColoursToTables();
}

std::vector<Order> OrderBookWidget::getNewOrdersFromOrderbook(bool is_bid) {
  std::vector<Order> orderList;

  std::map<double, OrderList> mapStructure;
  if (is_bid)
    mapStructure = orderBook->getBids();
  else
    mapStructure = orderBook->getAsks();
  for (auto priceLevel : mapStructure) {
    std::list<Order> ordersForPL = priceLevel.second;
    for (Order o : ordersForPL)
      orderList.push_back(o);
  }
  return orderList;
}

void OrderBookWidget::updateBothTables() {
  std::lock_guard<std::mutex> lock(orderBook->OB_mutex);
  if (orderBook->empty())
    return;
  std::vector<Order> bids = getNewOrdersFromOrderbook(true);
  std::vector<Order> asks = getNewOrdersFromOrderbook(false);
  reverse(bids.begin(), bids.end());
  reverse(asks.begin(), asks.end());
  updateTable(bids, bidsTableWidget);
  updateTable(asks, asksTableWidget);
}

void OrderBookWidget::addColoursToTables() {
  QBrush redBrush(Qt::red);

  for (int row = 0; row < asksTableWidget->rowCount(); ++row) {
    for (int col = 0; col < asksTableWidget->columnCount(); ++col) {
      QTableWidgetItem *item = asksTableWidget->item(row, col);
      if (item) {
        item->setForeground(redBrush);
      }
    }
  }

  QBrush greenBrush(Qt::green);

  for (int row = 0; row < bidsTableWidget->rowCount(); ++row) {
    for (int col = 0; col < bidsTableWidget->columnCount(); ++col) {
      QTableWidgetItem *item = bidsTableWidget->item(row, col);
      if (item) {
        item->setForeground(greenBrush);
      }
    }
  }
}

void OrderBookWidget::updateBalance() {
  currentBalance->setText("Current Balance = £" + QString::number(API::getBalance(), 'f', 2));
}

void OrderBookWidget::updateQuantity() {
  currentQuantity->setText("Current Quantity = " + QString::number(API::getQuantity(), 'f') + " ETH");
}

void OrderBookWidget::updatePnl() { pnl->setText("Total Position = £" + QString::number(API::getPnl(), 'f', 2)); }

void OrderBookWidget::setQuantity() { quantity->setText(QString::number(API::getBalance() / API::getPrice(), 'f')); }

void OrderBookWidget::setPrice() { price->setText(QString::number(API::getPrice())); }
