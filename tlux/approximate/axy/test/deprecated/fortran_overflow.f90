program overflow_handling
  implicit none
  integer :: result, max_value
  integer(SELECTED_INT_KIND(20)) :: a, b, mod_value ! Holds 

  a = 2**20
  b = 2**11
  max_value = huge(max_value)
  mod_value = max_value

  PRINT *, "A: ", A
  PRINT *, "B: ", B
  PRINT *, "MAX_VALUE: ", MAX_VALUE
  PRINT *, "HUGE(mod_value): ", HUGE(mod_value)
  PRINT *, "A * B: ", A * B

  if (abs(b) > max_value / abs(a)) then
     PRINT *, "fortran_overflow.f90: OVERFLOW detected"
     ! Overflow will occur
     result = mod(a * b, mod_value)
     ! The "+1" ensures that the maximum value itself wraps back to zero
  else
     PRINT *, "fortran_overflow.f90: standard multiplication"
     ! No overflow will occur
     result = a * b
  end if

  PRINT *, "RESULT:  ", RESULT
end program overflow_handling
