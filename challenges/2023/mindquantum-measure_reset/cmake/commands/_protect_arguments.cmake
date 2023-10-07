include_guard()

# ~~~
# Protect each element of list against expansion by using [==[${el}]==]
#
#  _protect_arguments(<variable>)
# ~~~
macro(_protect_arguments name)
  set(_tmp ${${name}})
  set(${name})
  foreach(_el ${_tmp})
    list(APPEND ${name} "[==[${_el}]==]")
  endforeach()
  unset(_tmp)
endmacro()
