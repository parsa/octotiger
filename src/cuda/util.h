#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <exception>
#include <string>

namespace octotiger { namespace util 
{

  inline auto& rnd() {
    static std::minstd_rand rnd;
    return rnd;
  }

  template <class T>
  inline std::string string(T el) {
    return std::to_string(el);
  }
  inline std::string string(const char* el) {
    return std::string(el);
  }
  inline std::string string(std::string el) {
    return el;
  }

  template <class T>
  constexpr T ceil_div(T num, T den) {
    return (num + den - 1) / den;
  }

  inline double double_square(double n) {
    return n * n;
  }
  inline double double_cube(double n) {
    return n * n * n;
  }

  inline std::size_t size(std::size_t ld, std::size_t n) {
    return ld * n;
  }

  inline void abort(std::string msg) {
    throw std::runtime_error(msg);
  }

  void check_condition(bool cond, std::string msg) {
    if (cond)
      return;

    util::abort(msg);
  }
}}
#endif  // UTIL_H
