#include "HeaderWidget.hpp"
#include <QFont>
#include <QPixmap>
#include <QDir>

HeaderWidget::HeaderWidget(DBHandler *handler, OrderBook *orderBook, QWidget *parent)
    : QWidget(parent), handler(handler), orderBook(orderBook) {
    initialiseLayoutAndAddButtons();
    this->setStyleSheet("background-color: rgba(12, 83, 45, 90); border-radius: 20px;");
}

HeaderWidget::~HeaderWidget() {
    if (newsTimer) {
        newsTimer->stop();
    }
}

void HeaderWidget::initialiseLayoutAndAddButtons() {
    QVBoxLayout *container = new QVBoxLayout(this);
    QHBoxLayout *buttonContainer = new QHBoxLayout;
    QHBoxLayout *logoAndCompanyContainer = new QHBoxLayout;

    setupLogoAndCompany();
    setupButtons();
    setupModelType();
    setupNewsTicker();

    buttonContainer->addWidget(modelType);
    buttonContainer->addWidget(startSimulation);
    buttonContainer->addWidget(cancelSimulation);

    logoAndCompanyContainer->addWidget(imageLabel);
    logoAndCompanyContainer->addStretch(1);
    logoAndCompanyContainer->addWidget(titleLabel);
    logoAndCompanyContainer->addStretch(1);

    container->addLayout(logoAndCompanyContainer);
    container->addLayout(buttonContainer);
}

void HeaderWidget::setupLogoAndCompany() {
    QFont titleFont("Roboto", 18, QFont::Bold);
    titleLabel = new QLabel(" UNIVERSITY OF LEEDS STOCK EXCHANGE ");
    titleLabel->setFont(titleFont);
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet("QLabel { color : white; }");

    QPixmap imagePixmap;
    
    // If running on Xcode, the directory is usually the Debug folder
    if (QDir::currentPath().contains("Debug")) {
        imagePixmap = QPixmap("../../resources/UoLSE_Logo.png");
    } else {
        imagePixmap = QPixmap("COMP5530M/cpp codebase/resources/UoLSE_Logo.png");
    }
    
//    QPixmap imagePixmap("COMP5530M/cpp codebase/resources/UoLSE_Logo.png");
    imageLabel = new QLabel;
    imageLabel->setPixmap(imagePixmap.scaled(150, 90, Qt::KeepAspectRatio));
}

void HeaderWidget::setupButtons() {
    QFont buttonFont("Arial", 10);
    QString buttonStyle = "QPushButton { color: white; background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #555555, stop: 1 #333333); border: 1px solid #555; border-radius: 10px; padding: 5px; min-width: 80px; min-height: 24px; } QPushButton:hover { background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #666666, stop: 1 #444444); } QPushButton:pressed { background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #555555, stop: 1 #222222); }";
    startSimulation = new QPushButton("Start Simulation");
    cancelSimulation = new QPushButton("Cancel Simulation");
    startSimulation->setFont(buttonFont);
    cancelSimulation->setFont(buttonFont);
    startSimulation->setStyleSheet(buttonStyle);
    cancelSimulation->setStyleSheet(buttonStyle);
    connectButtons();
}

void HeaderWidget::setupModelType() {
    modelType = new QComboBox(this);
    modelType->addItem("GAN Model");
    modelType->addItem("Pan's Model");
    modelType->setFont(QFont("Arial", 10));
    modelType->setStyleSheet("QComboBox { color: white; background-color: #444444; border-radius: 5px; padding: 3px; } QComboBox::drop-down { border: none; }");
}


void HeaderWidget::setupNewsTicker() {
    int rightMargin = -800;  // Margin from the right edge
    int widgetWidth = 320;  // Width of the newsTicker
    int widgetHeight = 30;  // Height of the newsTicker
    int yPos = 45;  // Vertical position from the top

    newsTicker = new QLabel(this);
    // Set background color, text color, and round corners directly on the newsTicker
    newsTicker->setStyleSheet("QLabel { color: white; font-size: 10pt; background-color: rgba(12, 83, 45, 90); border-radius: 10px; padding: 5px; }");

    // Calculate x-position from the right edge
    int xPos = this->width() - widgetWidth - rightMargin;

    // Set geometry with the calculated x-position
    newsTicker->setGeometry(QRect(xPos, yPos, widgetWidth, widgetHeight));
    
    newsItems << "Bitcoin hits new all-time high!" << "Major exchange suffers breach, Bitcoin safe." << "Bitcoin suffers major bear run" << "Global regulators propose new rules for Bitcoin transactions." << "Tech giants plan to integrate Bitcoin for payments." << "Environmental concerns rise over Bitcoin mining energy use.";
    newsTicker->setText(newsItems.first());

    newsTimer = new QTimer(this);
    connect(newsTimer, &QTimer::timeout, this, &HeaderWidget::updateNewsTicker);
    newsTimer->start(3000);
}

void HeaderWidget::updateNewsTicker() {
    currentNewsIndex = (currentNewsIndex + 1) % newsItems.size();
    newsTicker->setText(newsItems.at(currentNewsIndex));
    newsTicker->setAlignment(Qt::AlignCenter);

}

void HeaderWidget::connectButtons() {
    connect(startSimulation, &QPushButton::clicked, this, &HeaderWidget::startSim);
    connect(cancelSimulation, &QPushButton::clicked, this, &HeaderWidget::cancelSim);
}

void HeaderWidget::startSim() {
    std::string option = modelType->currentText().toStdString();

    if (option == "GAN Model"){
      cancelSim();
      Gan::startServer(handler, orderBook);
    }
}

void HeaderWidget::cancelSim(){
    Gan::cancelRequested = true;
    if (Gan::startGenerate.joinable()) { Gan::startGenerate.join(); }
}
