#pragma once

#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

class Statement;

namespace ir_utils {

template <
    typename FilterType,
    template <typename, typename...> typename Container,
    typename ElementType>
Container<FilterType*> filterVals(const Container<ElementType*>& container) {
  Container<FilterType*> out;
  for (auto& s : container) {
    if (s->getValType() == FilterType::type) {
      out.push_back(s->template as<FilterType>());
    }
  }
  return out;
}

} // namespace ir_utils
} // namespace fuser
} // namespace jit
} // namespace torch
