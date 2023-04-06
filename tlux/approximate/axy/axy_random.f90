MODULE RANDOM
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, IT => INT64, INT32
  USE ISO_C_BINDING, ONLY: LT => C_BOOL

  IMPLICIT NONE

  ! Constants used throughout the module.
  INTEGER(KIND=IT), PARAMETER :: ZERO = 0_IT
  INTEGER(KIND=IT), PARAMETER :: ONE = 1_IT
  INTEGER(KIND=IT), PARAMETER :: TWO = 2_IT
  INTEGER(KIND=IT), PARAMETER :: FOUR = 4_IT
  REAL(KIND=RT), PARAMETER :: PI = 3.141592653589793


CONTAINS


  ! Generate randomly distributed vectors on the N-sphere.
  SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
    ! Local variables.
    REAL(KIND=RT), DIMENSION(:,:), ALLOCATABLE :: TEMP_VECS
    REAL(KIND=RT) :: LEN
    INTEGER(KIND=IT) :: I, J, K
    LOGICAL(KIND=LT) :: GENERATED_WARNING
    ! Skip empty vector sets.
    IF (SIZE(COLUMN_VECTORS, KIND=IT) .LE. ZERO) THEN
       RETURN
    ELSE IF (SIZE(COLUMN_VECTORS, ONE, KIND=IT) .EQ. ONE) THEN
       COLUMN_VECTORS(:,:) = 1.0_RT
       RETURN
    END IF
    ! Allocate space to use for generating the random numbers
    ALLOCATE(TEMP_VECS( &
         SIZE(COLUMN_VECTORS, ONE, KIND=IT), &
         SIZE(COLUMN_VECTORS, TWO, KIND=IT) &
    ))
    ! Prepare this state variable to prevent redundant messages.
    GENERATED_WARNING = .FALSE._LT
    ! Generate random numbers in the range [0,1].
    CALL RANDOM_NUMBER(COLUMN_VECTORS(:,:))
    CALL RANDOM_NUMBER(TEMP_VECS(:,:))
    ! Map the random uniform numbers to a radial distribution.
    !   WARNING: `LOG(0.0) = -Infinity` and similarly for any values less than EPSILON.
    WHERE (COLUMN_VECTORS(:,:) .GT. EPSILON(COLUMN_VECTORS(ONE,ONE)))
       COLUMN_VECTORS(:,:) = SQRT(-LOG(COLUMN_VECTORS(:,:))) * COS(PI * TEMP_VECS(:,:))
    END WHERE
    ! Orthogonalize the first K vectors in (random) order.
    IF (SIZE(COLUMN_VECTORS, ONE, KIND=IT) .GT. ONE) THEN
       ! Compute the last vector that is part of the orthogonalization.
       K = MIN(SIZE(COLUMN_VECTORS, ONE, KIND=IT), SIZE(COLUMN_VECTORS, TWO, KIND=IT))
       ! Orthogonalize the "lazy way" without column pivoting. Could
       ! result in imperfectly orthogonal vectors (because of rounding
       ! errors being enlarged by upscaling), but that is accepted
       ! here as "randomness".
       DO I = ONE, K
          LEN = NORM2(COLUMN_VECTORS(:,I))
          ! Generate a new random vector (that might be linearly dependent on previous)
          !   when there is nothing remaining of the current vector after orthogonalization.
          DO WHILE (LEN .LT. EPSILON(LEN))
             IF (.NOT. GENERATED_WARNING) THEN
                PRINT *, ' WARNING (axy_random.f90): Encountered length-zero vector during orthogonalization.'
                PRINT *, '   Forced to generate new random vector. Some vectors are likely to be linearly dependent.'
                GENERATED_WARNING = .TRUE._LT
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
          TEMP_VECS(ONE,I+ONE:K) = MATMUL(COLUMN_VECTORS(:,I), COLUMN_VECTORS(:,I+ONE:K))
          DO J = I+ONE, K
             COLUMN_VECTORS(:,J) = COLUMN_VECTORS(:,J) - TEMP_VECS(ONE,J) * COLUMN_VECTORS(:,I)
          END DO
       END DO
       ! Make the rest of the column vectors unit length.
       DO I = K, SIZE(COLUMN_VECTORS, TWO, KIND=IT)
          LEN = NORM2(COLUMN_VECTORS(:,I))
          IF (LEN .GT. 0.0_RT) COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
       END DO
    END IF
  END SUBROUTINE RANDOM_UNIT_VECTORS


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


  ! Given the variables for a linear iterator, initialize it.
  SUBROUTINE INITIALIZE_ITERATOR(I_LIMIT, I_NEXT, I_MULT, I_STEP, I_MOD, SEED)
    INTEGER(KIND=IT), INTENT(IN) :: I_LIMIT
    INTEGER(KIND=IT), INTENT(OUT) :: I_NEXT, I_MULT, I_STEP, I_MOD
    INTEGER(KIND=IT), INTENT(IN), OPTIONAL :: SEED
    !  Storage for seeding the random number generator (for repeatability). LOCAL ALLOCATION
    INTEGER, DIMENSION(:), ALLOCATABLE :: SEED_ARRAY
    INTEGER :: I
    ! Set a random seed, if one was provided (otherwise leave default).
    IF (PRESENT(SEED)) THEN
       CALL RANDOM_SEED(SIZE=I)
       ALLOCATE(SEED_ARRAY(I))
       SEED_ARRAY(:) = INT(SEED)
       CALL RANDOM_SEED(PUT=SEED_ARRAY(:))
       DEALLOCATE(SEED_ARRAY)
    END IF
    ! 
    ! Construct an additive term, multiplier, and modulus for a linear
    ! congruential generator. These generators are cyclic and do not
    ! repeat when they maintain the properties:
    ! 
    !   1) "modulus" and "additive term" are relatively prime.
    !   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    !   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    ! 
    I_NEXT = RANDOM_INTEGER(MAX_VALUE=I_LIMIT) ! Pick a random initial value.
    I_MULT = ONE + FOUR * (I_LIMIT + RANDOM_INTEGER(MAX_VALUE=I_LIMIT)) ! Pick a multiplier 1 greater than a multiple of 4.
    I_STEP = ONE + TWO * RANDOM_INTEGER(MAX_VALUE=I_LIMIT) ! Pick a random odd-valued additive term.
    I_MOD = TWO ** CEILING(LOG(REAL(I_LIMIT)) / LOG(2.0_RT)) ! Pick a power-of-2 modulus just big enough to generate all numbers.
    ! Cap the multiplier and step by the "I_MOD" (since it doesn't matter if they are larger).
    I_MULT = MOD(I_MULT, I_MOD)
    I_STEP = MOD(I_STEP, I_MOD)
    ! Unseed the random number generator if it was seeded.
    IF (PRESENT(SEED)) THEN
       CALL RANDOM_SEED()
    END IF
  END SUBROUTINE INITIALIZE_ITERATOR

  
  ! Get the next index in the model point iterator.
  FUNCTION GET_NEXT_INDEX(I_LIMIT, I_NEXT, I_MULT, I_STEP, I_MOD, RESHUFFLE) RESULT(NEXT_I)
    INTEGER(KIND=IT), INTENT(IN) :: I_LIMIT
    INTEGER(KIND=IT), INTENT(INOUT) :: I_NEXT, I_MULT, I_STEP, I_MOD
    LOGICAL(KIND=LT), INTENT(IN), OPTIONAL :: RESHUFFLE
    INTEGER(KIND=IT) :: NEXT_I, I
    ! If the I_NEXT is not within the limit, cycle it until it is.
    next_candidate : DO I = ONE, I_LIMIT
       IF (I_NEXT .LT. I_LIMIT) THEN
          EXIT next_candidate
       END IF
       I_NEXT = MOD(I_NEXT * I_MULT + I_STEP, I_MOD)
    END DO next_candidate
    IF (I_NEXT .GE. I_LIMIT) THEN
       PRINT *, 'ERROR: Iterator failed to arrive at valid value after cycling the full limit.', &
            I_LIMIT, I_NEXT, I_MULT, I_STEP, I_MOD
       NEXT_I = 0
       RETURN
    END IF
    ! Store the "NEXT_I" that is currently set (and add 1 to make the range [1,limit]).
    NEXT_I = I_NEXT + ONE
    ! Reshuffle this data iterator if that behavior is desired.
    IF (PRESENT(RESHUFFLE)) THEN
       IF (RESHUFFLE) THEN
          IF (NEXT_I .EQ. I_LIMIT) THEN
             CALL INITIALIZE_ITERATOR(I_LIMIT, I_NEXT, I_MULT, I_STEP, I_MOD)
             I_NEXT = NEXT_I - ONE
          END IF
       END IF
    END IF
    ! Cycle I_NEXT to the next value in the sequence.
    I_NEXT = MOD(I_NEXT * I_MULT + I_STEP, I_MOD)
  END FUNCTION GET_NEXT_INDEX


  ! Map an integer I in the range [1, MAX_VALUE**2] to a unique pair
  !  of integers PAIR1 and PAIR2 with both in the range [1, MAX_VALUE].
  SUBROUTINE INDEX_TO_PAIR(MAX_VALUE, I, PAIR1, PAIR2)
    INTEGER(KIND=IT), INTENT(IN) :: MAX_VALUE, I
    INTEGER(KIND=IT), INTENT(OUT) :: PAIR1, PAIR2
    ! I = (PAIR1-ONE) * MAX_VALUE + PAIR2 ! 123456789
    PAIR1 = ONE + (I-ONE) / MAX_VALUE     ! 111222333
    PAIR2 = ONE + MOD((I-ONE), MAX_VALUE) ! 123123123
  END SUBROUTINE INDEX_TO_PAIR


  ! Map a pair of integers PAIR1 and PAIR2 in the range [1, MAX_VALUE]
  !  to an integer I in the range [1, MAX_VALUE**2].
  SUBROUTINE PAIR_TO_INDEX(MAX_VALUE, PAIR1, PAIR2, I)
    INTEGER(KIND=IT), INTENT(IN) :: MAX_VALUE, PAIR1, PAIR2
    INTEGER(KIND=IT), INTENT(OUT) :: I
    ! PAIR1 = ONE + (I-ONE) / MAX_VALUE     ! 111222333
    ! PAIR2 = ONE + MOD((I-ONE), MAX_VALUE) ! 123123123
    I = (PAIR1-ONE) * MAX_VALUE + PAIR2     ! 123456789
  END SUBROUTINE PAIR_TO_INDEX


END MODULE RANDOM
