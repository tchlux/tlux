
! Perform least squares with LAPACK.
! 
!   A is column vectors (of points) if TRANS='T', and row vectors 
!     (of points) if TRANS='N'.
!   B must be COLUMN VECTORS of fit output.
!   X always has a first dimension that is the smaller of A's sizes,
!     and the second dimension is determined by B's columns.
SUBROUTINE LEAST_SQUARES(TRANS, A, B, X)
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE
  REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A, B
  CHARACTER, INTENT(IN) :: TRANS
  REAL(KIND=RT), INTENT(OUT), DIMENSION(MIN(SIZE(A,1),SIZE(A,2)),SIZE(B,2)) :: X
  ! Local variables.
  INTEGER :: M, N, NRHS, LDA, LDB, LWORK, INFO
  REAL(KIND=RT), DIMENSION(:), ALLOCATABLE :: WORK
  EXTERNAL :: SGELS
  ! Set variables for calling least squares routine.
  M = SIZE(A,1)
  N = SIZE(A,2)
  NRHS = SIZE(B,2)
  LDA = SIZE(A,1)
  LDB = SIZE(B,1)
  ! Allocate the work space for the call.
  LWORK = MAX(1, MIN(M,N) + MAX(MIN(M,N), NRHS))
  ALLOCATE(WORK(LWORK))
  ! Make the call to the least squares routine.
  CALL SGELS( TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK, INFO )
  X(:,:) = B(:SIZE(X,1),:SIZE(X,2))
END SUBROUTINE LEAST_SQUARES
