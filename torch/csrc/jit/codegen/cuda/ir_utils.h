#pragma once

#include <torch/csrc/jit/codegen/cuda/type.h>

#include <iterator>
#include <functional>

namespace torch {
namespace jit {
namespace fuser {

namespace ir_utils {

template <typename FilterType, typename Iterator>
class FilterByTypeIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = FilterType*;
  using pointer = value_type*;
  using reference = value_type&;

  FilterByTypeIterator(Iterator begin, Iterator end) : current_(begin), end_(end) {
    advance();
  }

  FilterType* operator*() const {
    return (*current_)->template as<FilterType>();
  }

  FilterType* operator->() const {
    return (*this);
  }

  FilterByTypeIterator& operator++() {
    ++current_;
    advance();
    return *this;
  }

  FilterByTypeIterator operator++(int) {
    const auto before_increment = *this;
    ++current_;
    advance();
    return before_increment;
  }

  bool operator==(const FilterByTypeIterator& other) const {
    TORCH_INTERNAL_ASSERT(
        end_ == other.end_,
        "Comparing two FilteredViews that originate from different containers");
    return current_ == other.current_;
  }

  bool operator!=(const FilterByTypeIterator& other) const {
    return !(*this == other);
  }

 private:
  void advance() {
    current_ = std::find_if(current_, end_, [](const auto& val) {
      return dynamic_cast<const FilterType*>(val) != nullptr;
    });
  }

 private:
  Iterator current_;
  const Iterator end_;
};

// An iterable view to a given container of Val pointers. Only returns
// Vals of a given Val type.
// NOTE: Add a non-const iterator if needed.
template <typename FilterType, typename InputIt>
class FilterByTypeView {
 public:
  using value_type = FilterType*;
  using const_iterator = FilterByTypeIterator<FilterType, InputIt>;

  FilterByTypeView(InputIt first, InputIt last) : input_it_(first), last_(last) {}

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
auto filterByType(InputIt first, InputIt last) {
  return FilterByTypeView<FilterType, InputIt>(first, last);
}

template <typename FilterType, typename ContainerType>
auto filterByType(const ContainerType& inputs) {
  return filterByType<FilterType>(inputs.cbegin(), inputs.cend());
}

template <typename Iterator, typename ValueType>
class FilterIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = ValueType;
  using pointer = value_type*;
  using reference = value_type&;
  using filter_type = std::function<bool(const value_type&)>;

  FilterIterator(Iterator begin, Iterator end, filter_type filter)
      : current_(begin), end_(end), filter_(filter) {
    advance();
  }

  value_type& operator*() {
    return *current_;
  }

  value_type& operator->() {
    //return *this;
    return *current_;
  }

  FilterIterator& operator++() {
    ++current_;
    advance();
    return *this;
  }

  FilterIterator operator++(int) {
    const auto before_increment = *this;
    ++current_;
    advance();
    return before_increment;
  }

  bool operator==(const FilterIterator& other) const {
    TORCH_INTERNAL_ASSERT(
        end_ == other.end_,
        "Comparing two IterDomainIterators that originate from different containers");
    return current_ == other.current_;
  }

  bool operator!=(const FilterIterator& other) const {
    return !(*this == other);
  }

 private:
  void advance() {
    current_ = std::find_if(current_, end_, filter_);
  }

 private:
  Iterator current_;
  const Iterator end_;
  filter_type filter_;
};

template <typename InputIt>
class FilterView {
 public:
  using pointer = typename std::iterator_traits<InputIt>::pointer;
  using value_type = typename std::remove_pointer<pointer>::type;
  using iterator = FilterIterator<InputIt, value_type>;
  using const_iterator = FilterIterator<InputIt, const value_type>;
  using filter_type = std::function<bool(const value_type&)>;

  FilterView(InputIt first, InputIt last, filter_type filter)
      : input_it_(first), last_(last), filter_(filter) {}

  iterator cbegin() const {
    return const_iterator(input_it_, last_, filter_);
  }

  iterator begin() const {
    return cbegin();
  }

  iterator begin() {
    return iterator(input_it_, last_, filter_);
  }

  const_iterator cend() const {
    return const_iterator(last_, last_, filter_);
  }

  const_iterator end() const {
    return cend();
  }

  iterator end() {
    return iterator(last_, last_, filter_);
  }

 private:
  const InputIt input_it_;
  const InputIt last_;
  filter_type filter_;
};

template <typename Iterator>
auto filterView(Iterator first, Iterator last, std::function<bool(const typename std::iterator_traits<Iterator>::value_type&)> filter) {
  return FilterView<Iterator>(first, last, filter);
}

template <typename ContainerType>
auto filterView(const ContainerType& inputs,  std::function<bool(const typename ContainerType::value_type&)> filter) {
  return filterView(inputs.cbegin(), inputs.cend(), filter);
}

template <typename ContainerType>
auto filterView(ContainerType& inputs,  std::function<bool(const typename ContainerType::value_type&)> filter) {
  return filterView(inputs.begin(), inputs.end(), filter);
}

} // namespace ir_utils
} // namespace fuser
} // namespace jit
} // namespace torch
