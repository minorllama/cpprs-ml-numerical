#include <tensorflow/c/c_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Status = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;
using Graph = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
using Options = std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
using Tensor = std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>;

void check(TF_Status* status, const std::string& action) {
    if (TF_GetCode(status) != TF_OK) {
        throw std::runtime_error(action + ": " + TF_Message(status));
    }
}

struct SessionDeleter {
    TF_Status* status;  // Owned by main and kept alive until after this session.

    void operator()(TF_Session* session) const noexcept {
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
    }
};

TF_Output endpoint(TF_Graph* graph, const std::string& name) {
    const auto separator = name.rfind(':');
    if (separator == std::string::npos || separator + 1 == name.size()) {
        throw std::runtime_error("Expected a tensor name such as operation:0: " + name);
    }
    const std::string op_name = name.substr(0, separator);
    const std::string port = name.substr(separator + 1);
    std::size_t used = 0;
    const int index = std::stoi(port, &used);
    TF_Operation* op = TF_GraphOperationByName(graph, op_name.c_str());
    if (!op || used != port.size() || index < 0 || index >= TF_OperationNumOutputs(op)) {
        throw std::runtime_error("Invalid graph endpoint: " + name);
    }
    return {op, index};
}

std::vector<float> read_values(int argc, char** argv) {
    std::vector<float> values;
    for (int i = 2; i < argc; ++i) {
        const std::string text = argv[i];
        std::size_t used = 0;
        const float value = std::stof(text, &used);
        if (used != text.size() || !std::isfinite(value)) {
            throw std::runtime_error("Expected a finite temperature: " + text);
        }
        values.push_back(value);
    }
    return values;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIR RAW_C [RAW_C ...]\n";
        return 2;
    }

    try {
        const auto values = read_values(argc, argv);
        const std::filesystem::path directory(argv[1]);
        std::ifstream manifest(directory / "endpoints.txt");
        std::string feed_name, fetch_name;
        if (!(manifest >> feed_name >> fetch_name)) {
            throw std::runtime_error("Could not read " + (directory / "endpoints.txt").string());
        }

        Status status(TF_NewStatus(), TF_DeleteStatus);
        Graph graph(TF_NewGraph(), TF_DeleteGraph);
        Options options(TF_NewSessionOptions(), TF_DeleteSessionOptions);
        const char* tags[] = {"serve"};
        std::unique_ptr<TF_Session, SessionDeleter> session(
            TF_LoadSessionFromSavedModel(options.get(), nullptr, argv[1], tags, 1,
                                         graph.get(), nullptr, status.get()),
            SessionDeleter{status.get()});
        check(status.get(), "Load SavedModel");

        const TF_Output feed = endpoint(graph.get(), feed_name);
        const TF_Output fetch = endpoint(graph.get(), fetch_name);
        const std::int64_t shape[] = {static_cast<std::int64_t>(values.size()), 1};
        const auto bytes = values.size() * sizeof(float);
        Tensor input(TF_AllocateTensor(TF_FLOAT, shape, 2, bytes), TF_DeleteTensor);
        if (!input) {
            throw std::runtime_error("Could not allocate input tensor");
        }
        std::memcpy(TF_TensorData(input.get()), values.data(), bytes);

        TF_Tensor* input_ptr = input.get();
        TF_Tensor* output_ptr = nullptr;
        TF_SessionRun(session.get(), nullptr, &feed, &input_ptr, 1, &fetch, &output_ptr, 1,
                      nullptr, 0, nullptr, status.get());
        Tensor output(output_ptr, TF_DeleteTensor);
        check(status.get(), "Run inference");
        if (!output || TF_TensorType(output.get()) != TF_FLOAT ||
            TF_NumDims(output.get()) != 2 || TF_Dim(output.get(), 0) != shape[0] ||
            TF_Dim(output.get(), 1) != shape[1] || TF_TensorByteSize(output.get()) != bytes) {
            throw std::runtime_error("Expected a float32 output with shape [batch, 1]");
        }

        const auto* prediction = static_cast<const float*>(TF_TensorData(output.get()));
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::cout << prediction[i] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
