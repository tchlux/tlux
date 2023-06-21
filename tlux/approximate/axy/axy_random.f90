! Module for all random number generation used in the AXY package.
MODULE RANDOM
  USE ISO_FORTRAN_ENV, ONLY: REAL32, INT64, INT32
  USE ISO_C_BINDING, ONLY: LT => C_BOOL
  USE PCG32_MODULE, ONLY: PCG_INT => INT64, &
       PCG32_SEED_RANDOM, PCG32_RANDOM, PCG32_RANDOM_REAL, PCG32_BOUNDED_RANDOM

  IMPLICIT NONE

  ! Constants used throughout the module.
  INTEGER(KIND=INT64), PARAMETER :: ZERO = 0_INT64
  INTEGER(KIND=INT64), PARAMETER :: ONE = 1_INT64
  INTEGER(KIND=INT64), PARAMETER :: TWO = 2_INT64
  INTEGER(KIND=INT64), PARAMETER :: FOUR = 4_INT64
  REAL(KIND=REAL32), PARAMETER :: PI = 3.141592653589793
  INTEGER(KIND=INT64), PARAMETER :: RIGHT_32 = SHIFTL(1_INT64, 32) - 1  ! 111..

CONTAINS

  ! Set the seed for the random number generator.
  SUBROUTINE SEED_RANDOM(SEED)
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: SEED
    INTEGER(KIND=INT64) :: S
    IF (PRESENT(SEED)) THEN ; S = SEED
    ELSE ; CALL SYSTEM_CLOCK(COUNT=S) ; END IF
    CALL PCG32_SEED_RANDOM(STATE=INT(S,KIND=PCG_INT), SEQ=NOT(INT(S,KIND=PCG_INT)))
  END SUBROUTINE SEED_RANDOM


  ! Define a function for generating random integers.
  ! Optional MAX_VALUE is a noninclusive upper bound for the value generated.
  !   WARNING: Only generates integers in the range [0, 2**32).
  FUNCTION RANDOM_INTEGER(MAX_VALUE) RESULT(RANDOM_INT)
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: MAX_VALUE
    INTEGER(KIND=INT64) :: RANDOM_INT
    IF (PRESENT(MAX_VALUE)) THEN
       RANDOM_INT = IAND(RIGHT_32, TRANSFER(PCG32_BOUNDED_RANDOM(INT(MAX_VALUE, KIND=INT32)), RANDOM_INT))
    ELSE
       RANDOM_INT = IAND(RIGHT_32, TRANSFER(PCG32_RANDOM(), RANDOM_INT))
    END IF
  END FUNCTION RANDOM_INTEGER


  ! Fortran implementation of the algorithm for generating uniform likelihood
  !  real numbers described at http://mumble.net/~campbell/tmp/random_real.c
  SUBROUTINE RANDOM_REAL(R, S, V)
    REAL(KIND=REAL32), INTENT(OUT), OPTIONAL :: R(*), V
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: S
    INTEGER(KIND=INT64) :: I
    ! Populate V with a random real value.
    IF (PRESENT(V)) THEN
       V = PCG32_RANDOM_REAL()
    ! Populate R with random real values.
    ELSE IF (PRESENT(R) .AND. PRESENT(S)) THEN
       generation_loop : DO I = 1, S
          R(I) = PCG32_RANDOM_REAL()
       END DO generation_loop
    END IF
  END SUBROUTINE RANDOM_REAL


  ! Generate randomly distributed vectors on the N-sphere.
  SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
    REAL(KIND=REAL32), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
    ! Local variables.
    REAL(KIND=REAL32), DIMENSION(:,:), ALLOCATABLE :: TEMP_VECS
    REAL(KIND=REAL32) :: LEN
    INTEGER(KIND=INT64) :: I, J, K
    LOGICAL(KIND=LT) :: GENERATED_WARNING
    ! Skip empty vector sets.
    IF (SIZE(COLUMN_VECTORS, KIND=INT64) .LE. ZERO) THEN
       RETURN
    ELSE IF (SIZE(COLUMN_VECTORS, ONE, KIND=INT64) .EQ. ONE) THEN
       COLUMN_VECTORS(:,:) = 1.0_REAL32
       RETURN
    END IF
    ! Allocate space to use for generating the random numbers
    ALLOCATE(TEMP_VECS( &
         SIZE(COLUMN_VECTORS, ONE, KIND=INT64), &
         SIZE(COLUMN_VECTORS, TWO, KIND=INT64) &
    ))
    ! Prepare this state variable to prevent redundant messages.
    GENERATED_WARNING = .FALSE._LT
    ! Generate random numbers in the range [0,1].
    CALL RANDOM_REAL(R=COLUMN_VECTORS(:,:), S=SIZE(COLUMN_VECTORS,KIND=INT64))
    CALL RANDOM_REAL(R=TEMP_VECS(:,:), S=SIZE(TEMP_VECS,KIND=INT64))
    ! Map the random uniform numbers to a radial distribution.
    !   WARNING: `LOG(0.0) = -Infinity` and similarly for any values less than EPSILON.
    WHERE (COLUMN_VECTORS(:,:) .GT. EPSILON(COLUMN_VECTORS(ONE,ONE)))
       COLUMN_VECTORS(:,:) = SQRT(-LOG(COLUMN_VECTORS(:,:))) * COS(PI * TEMP_VECS(:,:))
    END WHERE
    ! Orthogonalize the first K vectors in (random) order.
    IF (SIZE(COLUMN_VECTORS, ONE, KIND=INT64) .GT. ONE) THEN
       ! Compute the last vector that is part of the orthogonalization.
       K = MIN(SIZE(COLUMN_VECTORS, ONE, KIND=INT64), SIZE(COLUMN_VECTORS, TWO, KIND=INT64))
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
             CALL RANDOM_REAL(R=COLUMN_VECTORS(:,I), S=SIZE(COLUMN_VECTORS,1,KIND=INT64))
             CALL RANDOM_REAL(R=TEMP_VECS(:,I), S=SIZE(TEMP_VECS,1,KIND=INT64))
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
       DO I = K, SIZE(COLUMN_VECTORS, TWO, KIND=INT64)
          LEN = NORM2(COLUMN_VECTORS(:,I))
          IF (LEN .GT. 0.0_REAL32) COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
       END DO
    END IF
  END SUBROUTINE RANDOM_UNIT_VECTORS


  ! Given the variables for a linear iterator, initialize it.
  SUBROUTINE INITIALIZE_ITERATOR(I_LIMIT, I_NEXT, I_MULT, I_STEP, I_MOD, I_ITER, SEED)
    INTEGER(KIND=INT64), INTENT(IN) :: I_LIMIT
    INTEGER(KIND=INT64), INTENT(OUT) :: I_NEXT, I_MULT, I_STEP, I_MOD, I_ITER
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: SEED
    !  Storage for seeding the random number generator (for repeatability). LOCAL ALLOCATION
    INTEGER, DIMENSION(:), ALLOCATABLE :: SEED_ARRAY
    INTEGER :: I
    ! Set a random seed, if one was provided (otherwise leave default).
    IF (PRESENT(SEED)) CALL SEED_RANDOM(SEED)
    ! 
    ! Construct an additive term, multiplier, and modulus for a linear
    ! congruential generator. These generators are cyclic and do not
    ! repeat when they maintain the properties:
    ! 
    !   1) "modulus" and "additive term" are relatively prime.
    !   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    !   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    ! 
    I_NEXT = ONE + RANDOM_INTEGER(MAX_VALUE=I_LIMIT) ! Pick a random initial value.
    I_MULT = ONE + FOUR * (I_LIMIT + ONE + RANDOM_INTEGER(MAX_VALUE=I_LIMIT)) ! Pick a multiplier 1 greater than a multiple of 4.
    I_STEP = ONE + TWO * (ONE + RANDOM_INTEGER(MAX_VALUE=I_LIMIT)) ! Pick a random odd-valued additive term.
    I_MOD = TWO ** MIN(62,CEILING(LOG(REAL(I_LIMIT)) / LOG(2.0_REAL32))) ! Pick a power-of-2 modulus just big enough to generate all numbers.
    ! Cap the multiplier and step by the "I_MOD" (since it doesn't matter if they are larger).
    I_MULT = MOD(I_MULT, I_MOD)
    I_STEP = MOD(I_STEP, I_MOD)
    ! Set the iteration to zero.
    I_ITER = ZERO
    ! Unseed the random number generator if it was seeded.
    IF (PRESENT(SEED)) THEN
       CALL RANDOM_SEED()
    END IF
  END SUBROUTINE INITIALIZE_ITERATOR

  
  ! Get the next index in the model point iterator.
  FUNCTION GET_NEXT_INDEX(I_LIMIT, I_NEXT, I_MULT, I_STEP, I_MOD, I_ITER, RESHUFFLE) RESULT(NEXT_I)
    INTEGER(KIND=INT64), INTENT(IN) :: I_LIMIT
    INTEGER(KIND=INT64), INTENT(INOUT) :: I_NEXT, I_MULT, I_STEP, I_MOD, I_ITER
    LOGICAL(KIND=LT), INTENT(IN), OPTIONAL :: RESHUFFLE
    INTEGER(KIND=INT64) :: NEXT_I, I
    ! If the I_NEXT is not within the limit, cycle it until it is.
    next_candidate : DO I = ONE, I_LIMIT
       IF (I_NEXT .LT. I_LIMIT) THEN
          EXIT next_candidate
       END IF
       I_NEXT = MOD(I_NEXT * I_MULT + I_STEP, I_MOD)
       I_ITER = I_ITER + ONE
    END DO next_candidate
    IF ((I_LIMIT .GT. ZERO) .AND. (I_NEXT .GE. I_LIMIT)) THEN
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
          IF (I_ITER .GE. I_MOD) THEN
             CALL INITIALIZE_ITERATOR(I_LIMIT, I_NEXT, I_MULT, I_STEP, I_MOD, I_ITER)
             I_NEXT = NEXT_I - ONE
          END IF
       END IF
    END IF
    ! Cycle I_NEXT to the next value in the sequence.
    I_NEXT = MOD(I_NEXT * I_MULT + I_STEP, I_MOD)
    I_ITER = I_ITER + ONE
  END FUNCTION GET_NEXT_INDEX


  ! Map an integer I in the range [1, MAX_VALUE**2] to a unique pair
  !  of integers PAIR1 and PAIR2 with both in the range [1, MAX_VALUE].
  SUBROUTINE INDEX_TO_PAIR(NUM_ELEMENTS, I, PAIR1, PAIR2)
    INTEGER(KIND=INT64), INTENT(IN) :: NUM_ELEMENTS, I
    INTEGER(KIND=INT64), INTENT(OUT) :: PAIR1, PAIR2
    INTEGER(KIND=INT64) :: N2, C, J
    N2 = NUM_ELEMENTS**TWO
    C = INT(SQRT(REAL(N2 - I)), KIND=INT64)
    J = (C + ONE)**TWO - N2 + I
    IF (BTEST(J,0)) THEN ! J is odd.
       PAIR2 = NUM_ELEMENTS - C
       PAIR1 = PAIR2 + J / TWO
    ELSE ! J is even.
       PAIR1 = NUM_ELEMENTS - C
       PAIR2 = PAIR1 + J / TWO
    END IF
  END SUBROUTINE INDEX_TO_PAIR


  ! Map a pair of integers PAIR1 and PAIR2 in the range [1, MAX_VALUE]
  !  to an integer I in the range [1, MAX_VALUE**2].
  SUBROUTINE PAIR_TO_INDEX(NUM_ELEMENTS, PAIR1, PAIR2, I)
    INTEGER(KIND=INT64), INTENT(IN) :: NUM_ELEMENTS, PAIR1, PAIR2
    INTEGER(KIND=INT64), INTENT(OUT) :: I
    IF (PAIR1 < PAIR2) THEN
       I = TWO * (NUM_ELEMENTS * (PAIR1 - ONE) + PAIR2) - PAIR1**TWO - ONE
    ELSE
       I = TWO * (NUM_ELEMENTS * (PAIR2 - ONE) + PAIR1) - PAIR2**TWO
    END IF
  END SUBROUTINE PAIR_TO_INDEX


END MODULE RANDOM
