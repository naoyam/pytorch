#pragma once

#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

namespace ir_utils {

#if 0
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
#else
template <typename FilterType, typename InputIt>
class FilterIterator {
 public:
  FilterIterator(InputIt first, InputIt last):
      input_it_(first), last_(last) {
    advance();
  }

  FilterType* operator*() const {
    return (*input_it_)->template as<FilterType>();
  }

  FilterIterator& operator++() {
    ++input_it_;
    advance();
    return *this;
  }

  bool operator==(const FilterIterator& other) const {
    return input_it_ == other.input_it_ &&
        last_ == other.last_;
  }

  bool operator!=(const FilterIterator& other) const {
    return !(*this == other);
  }

 private:
  InputIt input_it_;
  InputIt last_;

  void advance() {
    while (input_it_ != last_) {
      if ((*input_it_)->getValType() == FilterType::type) {
        break;
      }
      ++input_it_;
    }
  }
};

template <typename FilterType, typename InputIt>
class FilterValContainer {
 public:
  using value_type = FilterType*;
  using const_iterator = FilterIterator<FilterType, InputIt>;

  FilterValContainer(InputIt first, InputIt last):
      input_it_(first), last_(last) {
  }

  const_iterator begin() const {
    return const_iterator(input_it_, last_);
  }

  const_iterator end() const {
    return const_iterator(last_, last_);
  }

 private:
  InputIt input_it_;
  InputIt last_;
};

template <typename FilterType, typename InputIt>
auto filterVals(InputIt first, InputIt last) {
  return FilterValContainer<FilterType, InputIt>(first, last);
}

template <typename FilterType, typename ContainerType>
auto filterVals(const ContainerType& inputs) {
  return filterVals<FilterType>(inputs.begin(), inputs.end());
}
#endif

} // namespace ir_utils
} // namespace fuser
} // namespace jit
} // namespace torch
