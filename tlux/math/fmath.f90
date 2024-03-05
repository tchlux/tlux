
MODULE FMATH
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE

CONTAINS

  ! Orthogonalize and normalize column vectors of A with pivoting.
  SUBROUTINE ORTHONORMALIZE(A, LENGTHS)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: LENGTHS ! SIZE(A,2)
    REAL(KIND=RT) :: L, V
    INTEGER :: I, J, K
    column_orthogonolization : DO I = 1, SIZE(A,2)
       LENGTHS(I:) = SUM(A(:,I:)**2, 1)
       ! Pivot the largest magnitude vector to the front.
       J = I-1+MAXLOC(LENGTHS(I:),1)
       IF (J .NE. I) THEN
          ! Swap lengths.
          L = LENGTHS(I)
          LENGTHS(I) = LENGTHS(J)
          LENGTHS(J) = L
          ! Swap columns.
          DO K = 1, SIZE(A,1)
             V = A(K,I)
             A(K,I) = A(K,J)
             A(K,J) = V
          END DO
       END IF
       ! Subtract the current vector from all others (if length is substantially positive).
       IF (LENGTHS(I) .GT. EPSILON(1.0_RT)) THEN
          LENGTHS(I) = SQRT(LENGTHS(I)) ! Finish the 2-norm calculation.
          A(:,I) = A(:,I) / LENGTHS(I) ! Scale the vector to be unit length.
          ! Remove previous vector components from this one that might appear when
          !  scaling the length UP to 1 (i.e., minimize emergent colinearities).
          IF (LENGTHS(I) .LT. 1.0_RT) THEN
             A(:,I) = A(:,I) - MATMUL(A(:,1:I-1), MATMUL(TRANSPOSE(A(:,1:I-1)), A(:,I)))
             A(:,I) = A(:,I) / NORM2(A(:,I))
          END IF
          ! 
          ! Proceed to remove this direction from remaining vectors, if there are others.
          IF (I .LT. SIZE(A,2)) THEN
             LENGTHS(I+1:) = MATMUL(A(:,I), A(:,I+1:))
             DO J = I+1, SIZE(A,2)
                A(:,J) = A(:,J) - LENGTHS(J) * A(:,I)
             END DO
          END IF
       ! Otherwise, the length is nearly zero and this was the largest vector. Exit.
       ELSE ! (LENGTHS(I) .LE. EPSILON(1.0_RT))
          LENGTHS(I:) = 0.0_RT
          ! A(:,I:) = 0.0_RT ! <- Expected or not? Unclear. They're already (practically) zero.
          EXIT column_orthogonolization
       END IF
    END DO column_orthogonolization
  END SUBROUTINE ORTHONORMALIZE

END MODULE FMATH
