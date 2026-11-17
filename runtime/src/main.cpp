#include <model/model_config.hpp>

#include <string>
#include <stdexcept>
#include <optional>
#include <span>
#include <iostream>
#include <string_view>
#include <filesystem>

// cli config
struct LaunchConfig {
    std::string weightsFile; // root of huggingface repo
};

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
                cfg.weightsFile = std::filesystem::current_path().string() + "/" + args[i + 1];
                // std::cout << cfg.weightsFile << std::endl;
            }
        }
    }

    if (cfg.weightsFile.empty()) {
        return std::nullopt;
    }
    return cfg;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    std::optional<LaunchConfig> config = parse_args(argc, argv);
    if (!config) {
        std::cerr << "--weights-dir is required." << std::endl;
    }

    return 0;
}
