MODULE RANDOM
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, IT => INT64

  IMPLICIT NONE

CONTAINS

  ! Define a function for generating random integers.
  ! Optional MAX_VALUE is a noninclusive upper bound for the value generated.
  FUNCTION RANDOM_INTEGER(MAX_VALUE) RESULT(RANDOM_INT)
    REAL(KIND=RT) :: R
    INTEGER(KIND=IT), INTENT(IN), OPTIONAL :: MAX_VALUE
    INTEGER(KIND=IT) :: RANDOM_INT
    CALL RANDOM_NUMBER(R)
    IF (PRESENT(MAX_VALUE)) THEN
       RANDOM_INT = INT(R * MAX_VALUE, KIND=IT)
    ELSE
       RANDOM_INT = INT(R * HUGE(RANDOM_INT), KIND=IT)
    END IF
  END FUNCTION RANDOM_INTEGER

  ! Generate randomly distributed vectors on the N-sphere.
  SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
    ! Local variables.
    REAL(KIND=RT), DIMENSION(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2)) :: TEMP_VECS ! LOCAL ALLOCATION
    REAL(KIND=RT), PARAMETER :: PI = 3.141592653589793
    REAL(KIND=RT) :: LEN
    INTEGER :: I, J, K
    LOGICAL :: GENERATED_WARNING
    ! Skip empty vector sets.
    IF (SIZE(COLUMN_VECTORS) .LE. 0) RETURN
    ! Prepare this state variable to prevent redundant messages.
    GENERATED_WARNING = .FALSE.
    ! Generate random numbers in the range [0,1].
    CALL RANDOM_NUMBER(COLUMN_VECTORS(:,:))
    CALL RANDOM_NUMBER(TEMP_VECS(:,:))
    ! Map the random uniform numbers to a radial distribution.
    !   WARNING: `LOG(0.0) = -Infinity` and similarly for any values less than EPSILON.
    WHERE (COLUMN_VECTORS(:,:) .GT. EPSILON(COLUMN_VECTORS(1,1)))
       COLUMN_VECTORS(:,:) = SQRT(-LOG(COLUMN_VECTORS(:,:))) * COS(PI * TEMP_VECS(:,:))
    END WHERE
    ! Orthogonalize the first K vectors in (random) order.
    IF (SIZE(COLUMN_VECTORS,1) .GT. 1) THEN
       ! Compute the last vector that is part of the orthogonalization.
       K = MIN(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2))
       ! Orthogonalize the "lazy way" without column pivoting. Could
       ! result in imperfectly orthogonal vectors (because of rounding
       ! errors being enlarged by upscaling), that is acceptable here.
       DO I = 1, K
          LEN = NORM2(COLUMN_VECTORS(:,I))
          ! Generate a new random vector (that might be linearly dependent on previous)
          !   when there is nothing remaining of the current vector after orthogonalization.
          DO WHILE (LEN .LT. EPSILON(LEN))
             IF (.NOT. GENERATED_WARNING) THEN
                PRINT *, ' WARNING (random.f90): Encountered length-zero vector during orthogonalization. Forced'
                PRINT *, '   to generate new random vector. Some vectors are likely to be linearly dependent.'
                GENERATED_WARNING = .TRUE.
             END IF
             CALL RANDOM_NUMBER(COLUMN_VECTORS(:,I))
             CALL RANDOM_NUMBER(TEMP_VECS(:,I))
             COLUMN_VECTORS(:,I) = SQRT(-LOG(COLUMN_VECTORS(:,I))) * COS(PI * TEMP_VECS(:,I))
             LEN = NORM2(COLUMN_VECTORS(:,I))
          END DO
          ! Make this column unit length.
          COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
          ! Compute multipliers (store in row of TEMP_VECS) and subtract
          ! from all remaining columns (doing the orthogonalization).
          TEMP_VECS(1,I+1:K) = MATMUL(COLUMN_VECTORS(:,I), COLUMN_VECTORS(:,I+1:K))
          DO J = I+1, K
             COLUMN_VECTORS(:,J) = COLUMN_VECTORS(:,J) - TEMP_VECS(1,J) * COLUMN_VECTORS(:,I)
          END DO
       END DO
       ! Make the rest of the column vectors unit length.
       DO I = K, SIZE(COLUMN_VECTORS,2)
          LEN = NORM2(COLUMN_VECTORS(:,I))
          IF (LEN .GT. 0.0_RT) COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
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
