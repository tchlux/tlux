program print_binary_integer

implicit none

integer :: i, n, integer_value
character(len=1) :: binary_bit

! Get the integer value from the user.
print *, "Enter an integer value: "
read *, integer_value

! Get the byte size of the integer.
n = sizeof(integer_value)

! Print out the binary representation of the integer.
do i = 1, n
  binary_bit = char(btest(integer_value, i - 1))
  write (*, '(a1)') binary_bit
end do

end program print_binary_integer
