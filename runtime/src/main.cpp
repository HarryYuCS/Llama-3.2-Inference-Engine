#include <model/model_config.hpp>
#include <model/weight_store.hpp>

#include <string>
#include <stdexcept>
#include <optional>
#include <span>
#include <iostream>
#include <string_view>
#include <filesystem>

// cli config object
struct LaunchConfig {
    std::string weightsPath; // root of huggingface repo
};

// parses cli args and returns LaunchConfig object
std::optional<LaunchConfig> parse_args(int argc, char** argv) {
    LaunchConfig cfg;

    // wrap launch args in a span to avoid index 0 confusion
    // span takes beginning pointer and count instead of iterators
    std::span<char*> args(argv + 1, static_cast<size_t>(argc - 1));

    // TODO : expand this
    // for now, just parse the weights file
    for (size_t i = 0; i < args.size(); ++i) {
        if (std::string_view{args[i]} == "--weights-dir") {
            if (i + 1 >= args.size()) {
                return std::nullopt;
            }
            else {
                std::string cwd = std::filesystem::current_path().string();

                cfg.weightsPath = cwd + "/" + args[i + 1];
                // std::cout << cfg.weightsFile << std::endl;
            }
        }
    }

    if (cfg.weightsPath.empty()) {
        return std::nullopt;
    }
    return cfg;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    std::optional<LaunchConfig> launchConfigOptional = parse_args(argc, argv);
    if (!launchConfigOptional) {
        std::cerr << "--weights-dir is required." << std::endl;
    }

    LaunchConfig launchConfig = launchConfigOptional.value();
    model::ModelConfig modelConfig = model::ModelConfig(launchConfig.weightsPath + "/config.json");
    model::WeightStore weightStore = model::WeightStore(modelConfig, launchConfig.weightsPath + "/model.safetensors");

    return 0;
}
