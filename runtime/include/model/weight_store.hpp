/**

Owns device tensors for each layer, maintains mapping

*/


#ifndef WEIGHT_STORE_HPP
#define WEIGHT_STORE_HPP

#include <model/model_config.hpp>
#include <core/tensor.hpp>

#include <unordered_map>

/**
scheme:

model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.norm.weight
*/

namespace model {

class WeightStore {
public:
    WeightStore(const ModelConfig &); // parse and load tensors

    const core::DeviceTensor& get(const std::string &name) const;
private:
    std::unordered_map<std::string, core::DeviceTensor> tensors_;

    void parseSafetensors_(const std::string &path); // parse .safetensors file at path
};

} // namespace model

#endif // WEIGHT_STORE_HPP