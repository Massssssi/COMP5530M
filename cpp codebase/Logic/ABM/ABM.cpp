//
// Created by Daniel on 19/04/2024.
//
#include "ABM.hpp"

std::thread Abm::startGenerate;
std::atomic<bool> Abm::cancelRequested(false);
std::vector<std::vector<double>> Abm::csvData;
int Abm::counter = 0;
bool flag = true;

void Abm::readCsv(const std::string& filename) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
    return;
  }

  std::string line;
  int count = 0;
  while (std::getline(file, line)) {
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    if (line.empty()) { continue; }

    std::vector<double> row;
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {

      if (cell.empty() || cell.find("Bid") != std::string::npos || cell.find("Ask") != std::string::npos) {
        continue;
      }

      try {
        row.push_back(std::stod(cell));
      } catch (const std::invalid_argument& e) {
        std::cerr << "Could not convert the string to double: " << cell << std::endl;
      }
    }

    if (!row.empty()) { csvData.push_back(row); }
    count++;
  }

  file.close();
  return;
}

std::string Abm::getProjectSourceDirectory() {
  std::string currFilePath = __FILE__;
  std::filesystem::path fullPath(currFilePath);
  return fullPath.parent_path().string();
}

void Abm::startServer(DBHandler *handler, OrderBook *globalOrderBook) {
  cancelRequested = false;
  readCsv(getProjectSourceDirectory()+"/abm_lob.csv");
  startGenerate = std::thread(generateQuantity, handler, globalOrderBook);
  std::cout << "Starting thread!\n";
}

void Abm::generateQuantity(DBHandler *handler, OrderBook *globalOrderBook) {
  while (!cancelRequested){
    globalOrderBook->OB_mutex.lock();
    globalOrderBook->clear_order_book();

    if (flag){
      API::setPrice(csvData.at(counter).at(0));
      for (int i = 0; i < 10; i++){
        std::vector<double> current = csvData.at(counter+i);
        globalOrderBook->add_order(i+1, current.at(0), current.at(1), true);
        globalOrderBook->add_order(i+11, current.at(2), current.at(3), false);
      }

      counter +=10;
      if (counter == csvData.size()){ flag = false; }
    }else{
      for (int i = 0; i < 10; i++){
        std::vector<double> current = csvData.at(counter-i-1);
        globalOrderBook->add_order(i+1, current.at(0), current.at(1), true);
        globalOrderBook->add_order(i+11, current.at(2), current.at(3), false);
      }
      counter -=10;
      API::setPrice(csvData.at(counter).at(0));

      if (counter == 0){ flag = true; }
    }

    API::updatePnL();
    handler->emitSuccessfulUpdate();
    globalOrderBook->OB_mutex.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}