program random_integer_example
    use, intrinsic :: iso_fortran_env, only : INT64
    implicit none

    integer(INT64) :: i, n, random_integer
    real :: random_float

    ! Initialize the random number generator
    call random_seed()

    ! Set the upper limit of the range
    n = 10  ! Example: Generate a random integer between 1 and 10

    ! Print the random integer
    DO I = 1, 20
       ! Generate a random floating point number in [0,1)
       call random_number(random_float)
       ! Scale and shift to [1, N]
       random_integer = 1 + int(random_float * n)
       print *, 'Random integer in range [1, ', n, ']: ', random_integer
    END DO

end program random_integer_example
