# Copyright (c) 2015-2016 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
  create_tree
  )

foreach(test ${tests})
  add_hpx_executable(
    ${test}
    DEPENDENCIES
      ${dependencies} octotiger_lib
    SOURCES
      ${test}.cpp
    FOLDER "Octotiger/Tests/${test}"
  )

  add_test(NAME ${test} COMMAND $<TARGET_FILE:${test}_exe>)
endforeach()
