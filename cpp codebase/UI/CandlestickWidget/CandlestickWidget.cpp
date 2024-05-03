#include "CandlestickWidget.hpp"
#include <QGuiApplication>
#include <QScreen>

CandlestickWidget::CandlestickWidget(DBHandler *handler, OrderBook *orderBookParam) : orderBook(orderBookParam)
{
    setWidgetProperties();
    addLayoutToThisWidget();
    
    connect(handler, &DBHandler::orderBookUpdated, this, &CandlestickWidget::updateInformation);
}

void CandlestickWidget::setWidgetProperties(){ this->setMinimumSize(900, 550); }

void CandlestickWidget::addLayoutToThisWidget(){
    QVBoxLayout *layout = new QVBoxLayout;
    addChartViewToLayout(layout);
    this->setLayout(layout);
}

void CandlestickWidget::addChartViewToLayout(QLayout *layout){
    setupSeries();
    setupChart();
    setupAxes();
    setupChartView();
    layout->addWidget(chartView);
}

void CandlestickWidget::setupChartView(){
    chartView = new QtCharts::QChartView();
    chartView->setChart(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
}

void CandlestickWidget::setupAxes(){
    xAxesList = QStringList{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20" ,
                            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"};
    QtCharts::QBarCategoryAxis *newX = new QtCharts::QBarCategoryAxis();
    QtCharts::QValueAxis *newY = new QtCharts::QValueAxis();
    newX->setCategories(xAxesList);
    newY->setRange(0, 0.1);
    chart->addAxis(newX, Qt::AlignBottom);
    chart->addAxis(newY, Qt::AlignLeft);
    chart->axes(Qt::Horizontal).first()->setGridLineVisible(false);
    candlestickSeries->attachAxis(newX);
    candlestickSeries->attachAxis(newY);
}

void CandlestickWidget::setupSeries(){
    candlestickSeries = new QtCharts::QCandlestickSeries();
    candlestickSeries->setName("ETH");
    candlestickSeries->setIncreasingColor(QColor(Qt::green));
    candlestickSeries->setDecreasingColor(QColor(Qt::red));
}

void CandlestickWidget::setupChart(){
    chart = new QtCharts::QChart();
    chart->setTitle("Ethereum Price");
    chart->addSeries(candlestickSeries);
    chart->setTheme(QtCharts::QChart::ChartThemeDark);
}

void CandlestickWidget::updateInformation(){
    std::lock_guard<std::mutex> lock(orderBook->OB_mutex);
    if (orderBook->empty())
        return;
    bestAskBidPair abp = getBestAskBid();
    updateCurrentStick(abp);
    if (currentStick.seconds == _updateRate){
        appendCurrentStickToGraph();
        resetCurrentStick();
    }
}

void CandlestickWidget::updateCurrentStick(bestAskBidPair abp){
    double price = (abp.first + abp.second) / 2;
    if (price > currentStick.high) currentStick.high = price;
    if (price < currentStick.low) currentStick.low = price;
    if (currentStick.seconds == 0) currentStick.open = price;
    if (currentStick.seconds == _updateRate-1) currentStick.close = price;
    currentStick.seconds++;
}

void CandlestickWidget::resizeYAxis(){
    double maxi = DBL_MIN, mini = DBL_MAX;
    
    for(int i=0;i<candlestickSeries->count();i++){
        QtCharts::QCandlestickSet *item = candlestickSeries->sets().at(i);
        maxi = fmax(maxi, item->high());
        mini = fmin(mini, item->low());
    }
    QtCharts::QValueAxis *axisY = qobject_cast<QtCharts::QValueAxis *>(chart->axes(Qt::Vertical).at(0));
    axisY->setMax(maxi + 0.5);
    axisY->setMin(mini - 0.5);
}

void CandlestickWidget::appendCurrentStickToGraph(){
    QtCharts::QValueAxis *axisY = qobject_cast<QtCharts::QValueAxis *>(chart->axes(Qt::Vertical).at(0));
    QtCharts::QCandlestickSet *set = new QtCharts::QCandlestickSet(static_cast<qreal>(counter));
    set->setHigh(static_cast<qreal>(currentStick.high));
    set->setLow(static_cast<qreal>(currentStick.low));
    set->setOpen(static_cast<qreal>(currentStick.open));
    set->setClose(static_cast<qreal>(currentStick.close));

    if (candlestickSeries->count() == 30)
        candlestickSeries->remove(candlestickSeries->sets().at(0));
    candlestickSeries->append(set);
    resizeYAxis();
}

void CandlestickWidget::resetCurrentStick(){
    currentStick.high = DBL_MIN;
    currentStick.low = DBL_MAX;
    currentStick.open = -1.0;
    currentStick.close = -1.0;
    currentStick.seconds = 0;
}

bestAskBidPair CandlestickWidget::getBestAskBid(){
    double ask = getBestAsk();
    double bid = getBestBid();
    return bestAskBidPair {ask, bid};
}

double CandlestickWidget::getBestAsk(){
    std::map<double, OrderList> mapStructure;
    mapStructure = orderBook->getAsks();
    return mapStructure.begin()->first;
}

double CandlestickWidget::getBestBid(){
    std::map<double, OrderList> mapStructure;
    mapStructure = orderBook->getBids();
    return mapStructure.rbegin()->first;
}
