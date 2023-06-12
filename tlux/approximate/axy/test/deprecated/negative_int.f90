program main
    implicit none
    integer :: x
    x = -HUGE(x)
    print*, x
    print*, x-1
    print*, x-2
end program main
