//
// Created by Daniel on 05/04/2024.
//
#include <torch/script.h>
#include "GAN.hpp"

bool Gan::flag = false;
std::thread Gan::startGenerate;
std::atomic<bool> Gan::cancelRequested(false);
std::random_device rd1;
std::default_random_engine e1(rd1());

void Gan::updateOrderBook(float *values, OrderBook *globalOrderBook) {
  for (int i = 6; i > 0; i--) {
    double price_level = API::getPrice() + i - 3;
    double new_quantity = abs(values[i - 1]);
    globalOrderBook->add_order(i, price_level, new_quantity, i <= 3);
  }
}

std::string Gan::getProjectSourceDirectory() {
  std::string currFilePath = __FILE__;
  std::filesystem::path fullPath(currFilePath);
  return fullPath.parent_path().string();
}

void getNextCentre(float *values, torch::Tensor &output, OrderBook *globalOrderBook) {
  if (!output.defined() || output.numel() < 12 || !output.is_contiguous()) {
    std::cerr << "Output tensor is not valid or does not have the expected number of elements." << std::endl;
    return;
  }
  // result of how many bids we have (denoted by negative values)
  int balance = 0;

  float temp[12];

  if (output.dim() != 2) {
    std::cerr << "Tensor is not 2D. It has " << output.dim() << " dimensions." << std::endl;
    return;
  }

  for (int64_t j = 0; j < output.size(1); ++j) {
    temp[j] = output[0][j].item<float>();
  }

  int random = e1();
  if (random % 2 == 1) {
    std::reverse(temp, temp + 12);
    for (int i = 0; i < 12; i++) {
      temp[i] *= -1;
    }
  }

  for (int i = 0; i < 12; i++) {
    if (temp[i] < 0) balance++;
  }

  // do balance - 6 to calculate the next best bid for the centre of the OB
  API::setPrice(API::getPrice() + balance - 6);

  // Store the new centre in vals
  for (int i = 2; i <= 6; ++i) {
    if (temp[i] < 0 && temp[i + 1] > 0) {
      for (int j = 0; j < 6; ++j) {
        values[j] = temp[i - 2 + j];
      }
      break;
    }
  }

}

void Gan::startServer(DBHandler *handler, OrderBook *globalOrderBook) {
  cancelRequested = false;
  startGenerate = std::thread(generateQuantity, handler, globalOrderBook);
}

void Gan::generateQuantity(DBHandler *handler, OrderBook *globalOrderBook) {
  try {
    std::string model_path = getProjectSourceDirectory() + "/RecreatingBestModelCPU.pt";
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.eval();

    torch::Tensor output = torch::empty({0});
    float values[6] = {-0.7787362, -0.8905415, -0.6589257, 0.62962157, 1.2157797, 2.0017693};

    while (!cancelRequested) {
      if (flag) { getNextCentre(values, output, globalOrderBook); }

      flag = true;
      torch::Tensor S_t = torch::from_blob(values, torch::IntArrayRef{1, 6});
      torch::Tensor z_t = torch::randn({1, 12});
      output = model.forward({z_t, S_t}).toTensor();

      if (output.dim() != 2) {
        std::cout << "Tensor is not 2D. It has " << output.dim() << " dimensions." << std::endl;
      }

      globalOrderBook->clear_order_book();
      updateOrderBook(values, globalOrderBook);
      API::updatePnL();
      handler->emitSuccessfulUpdate();
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }
  catch (const c10::Error &e) {
    std::cerr << "Error loading the model: " << e.what() << std::endl;
  }
}
