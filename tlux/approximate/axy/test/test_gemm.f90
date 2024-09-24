PROGRAM GEMM_TEST
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  USE MATRIX_OPERATIONS, ONLY: GEMM
  
  IMPLICIT NONE

  ! Define matrix dimensions
  CHARACTER, PARAMETER :: &
       OP_A = 'n', & ! No transpose of A
       OP_B = 't' ! Transpose of B
  INTEGER, PARAMETER :: A_ROWS = 3, B_ROWS = 3, DIM = 3
  REAL(KIND=RT), DIMENSION(:,:) :: A(A_ROWS,DIM), B(B_ROWS,DIM), C(A_ROWS,B_ROWS)
  INTEGER :: I, C_ROWS, C_COLS

  C_ROWS = A_ROWS
  C_COLS = B_ROWS

  ! Randomize inputs to see changing outputs.
  ! CALL RANDOM_SEED()
  ! CALL RANDOM_NUMBER(A)
  ! CALL RANDOM_NUMBER(B)

  ! Explicitly set inputs to control behavior.
  A(:,:) = 1.0_RT
  B(:,:) = 1.0_RT

  C = 10.0_RT  ! Create an identity matrix (all ones) for the output

  ! Call GEMM with a multiplier that is zero.
  CALL GEMM(OP_A, OP_B, C_ROWS, C_COLS, DIM, &
       1.0_RT, A, A_ROWS, B, B_ROWS, &
       1.0_RT, C, C_ROWS)

  ! Print the resulting matrix
  WRITE(*,*) ""
  WRITE(*,*) "A"
  DO I = 1, A_ROWS
     WRITE(*, "(' ', 10(F7.3, ','))") A(I,:)
  END DO
  WRITE(*,*) ""
  WRITE(*,*) "B"
  DO I = 1, B_ROWS
     WRITE(*, "(' ', 10(F7.3, ','))") B(I,:)
  END DO
  WRITE(*,*) ""
  WRITE(*,*) "C"
  DO I = 1, C_ROWS
     WRITE(*, "(' ', 10(F7.3, ','))") C(I,:)
  END DO

END PROGRAM
