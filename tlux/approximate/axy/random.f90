MODULE RANDOM
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, IT => INT64

  IMPLICIT NONE

CONTAINS

  ! Generate randomly distributed vectors on the N-sphere.
  SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
    ! Local variables. LOCAL ALLOCATION
    REAL(KIND=RT), DIMENSION(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2)) :: TEMP_VECS
    REAL(KIND=RT), PARAMETER :: PI = 3.141592653589793
    REAL(KIND=RT) :: LEN
    INTEGER :: I, J, K
    ! Skip empty vector sets.
    IF (SIZE(COLUMN_VECTORS) .LE. 0) RETURN
    ! Generate random numbers in the range [0,1].
    CALL RANDOM_NUMBER(COLUMN_VECTORS(:,:))
    CALL RANDOM_NUMBER(TEMP_VECS(:,:))
    ! Map the random uniform numbers to a radial distribution.
    COLUMN_VECTORS(:,:) = SQRT(-LOG(COLUMN_VECTORS(:,:))) * COS(PI * TEMP_VECS(:,:))
    ! Orthogonalize the vectors in (random) order.
    IF (SIZE(COLUMN_VECTORS,1) .GT. 1) THEN
       ! Compute the last vector that is part of the orthogonalization.
       K = MIN(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2))
       ! Orthogonalize the "lazy way" without column pivoting.
       ! Could result in imperfectly orthogonal vectors (because of
       ! rounding errors being enlarged by upscaling), that is acceptable.
       DO I = 1, K-1
          LEN = NORM2(COLUMN_VECTORS(:,I))
          IF (LEN .GT. 0.0_RT) THEN
             ! Make this column unit length.
             COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
             ! Compute multipliers (store in row of TEMP_VECS) and subtract
             ! from all remaining columns (doing the orthogonalization).
             TEMP_VECS(1,I+1:K) = MATMUL(COLUMN_VECTORS(:,I), COLUMN_VECTORS(:,I+1:K))
             DO J = I+1, K
                COLUMN_VECTORS(:,J) = COLUMN_VECTORS(:,J) - TEMP_VECS(1,J) * COLUMN_VECTORS(:,I)
             END DO
          ELSE
             ! This should not happen (unless the vectors are at least in the
             !   tens of thousands, in which case a different method should be used).
             PRINT *, 'ERROR: Random unit vector failed to initialize correctly, rank deficient.'
          END IF
       END DO
       ! Make the rest of the column vectors unit length.
       DO I = K, SIZE(COLUMN_VECTORS,2)
          LEN = NORM2(COLUMN_VECTORS(:,I))
          IF (LEN .GT. 0.0_RT)  COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
       END DO
    END IF
  END SUBROUTINE RANDOM_UNIT_VECTORS

  ! Generate a new RANGE_STATE for a random number generator that creates random
  ! numbers in a range in a cyclic, covering, and nonrepeating fashion using
  ! a linear random number generator.
  SUBROUTINE RANDOM_RANGE(FIRST, LAST, STEP, COUNT, STATE)
    INTEGER(KIND=IT), INTENT(IN) :: FIRST
    INTEGER(KIND=IT), INTENT(IN), OPTIONAL :: LAST, STEP, COUNT
    INTEGER(KIND=IT), INTENT(OUT) :: STATE ! TODO: This needs to be custom type.
  END SUBROUTINE RANDOM_RANGE

END MODULE RANDOM
