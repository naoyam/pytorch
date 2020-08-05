#pragma once

#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

namespace ir_utils {

template <typename FilterType, typename Iterator>
class FilterIterator {
 public:
  FilterIterator(Iterator begin, Iterator end) : current_(begin), end_(end) {
    advance();
  }

  FilterType* operator*() const {
    return (*current_)->template as<FilterType>();
  }

  FilterIterator& operator++() {
    ++current_;
    advance();
    return *this;
  }

  bool operator==(const FilterIterator& other) const {
    TORCH_INTERNAL_ASSERT(
        end_ == other.end_,
        "Comparing two FilteredViews that originate from different containers");
    return current_ == other.current_;
  }

  bool operator!=(const FilterIterator& other) const {
    return !(*this == other);
  }

 private:
  Iterator current_;
  const Iterator end_;

 private:
  void advance() {
    current_ = std::find_if(current_, end_, [](const auto& val) {
      return val->getValType() == FilterType::type;
    });
  }
};

// An iterable view to a given container of Val pointers. Only returns
// Vals of a given Val type.
// NOTE: Add a non-const iterator if needed.
template <typename FilterType, typename InputIt>
class FilteredView {
 public:
  using value_type = FilterType*;
  using const_iterator = FilterIterator<FilterType, InputIt>;

  FilteredView(InputIt first, InputIt last) : input_it_(first), last_(last) {}

  const_iterator cbegin() const {
    return const_iterator(input_it_, last_);
  }

  const_iterator begin() const {
    return cbegin();
  }

  const_iterator cend() const {
    return const_iterator(last_, last_);
  }

  const_iterator end() const {
    return cend();
  }

 private:
  const InputIt input_it_;
  const InputIt last_;
};

template <typename FilterType, typename InputIt>
auto filterVals(InputIt first, InputIt last) {
  return FilteredView<FilterType, InputIt>(first, last);
}

template <typename FilterType, typename ContainerType>
auto filterVals(const ContainerType& inputs) {
  return filterVals<FilterType>(inputs.cbegin(), inputs.cend());
}

} // namespace ir_utils
} // namespace fuser
} // namespace jit
} // namespace torch
