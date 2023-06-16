! WARNING: This code is inefficient because there is no unsigned
!  integer in Fortran, which means that the desired implicit
!  modulo behavior does not work correctly for 64-bit arithmetic.
!  To compensate, the next larger integer size is used (128-bit)
!  and the mod is done explicitly. In turn, the 32-bit integer
!  output is SIGNED, and can be negative.
! 
! A Fortran implmementation of the basic PCG 32-bit random
!  number generator with a 64 bit state. Specifically the
!  XOR shift high bits with random rotation (PCG-XSH-RR-64_32).
! 
!   https://www.pcg-random.org/using-pcg-c-basic.html
!   https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf
! 
MODULE PCG32_MODULE
  USE ISO_FORTRAN_ENV, ONLY: INT32, REAL32
  USE ISO_C_BINDING, ONLY: C_SIZEOF
  IMPLICIT NONE

  ! Define an integer that holds at least 2**64 = 1.845e19
  ! Since Fortran doesn't have unsigned integers, this will be an INT128.
  INTEGER, PARAMETER :: INT64 = SELECTED_INT_KIND(19) 

  TYPE, BIND(C) :: PCG32_STATE
     INTEGER(KIND=INT64) :: STATE  
     INTEGER(KIND=INT64) :: INC
  END TYPE PCG32_STATE

  TYPE(PCG32_STATE), TARGET, SAVE :: PCG32_GLOBAL = PCG32_STATE(&
       INT(Z'853C49E6748FEA9B', KIND=INT64), & ! Recommended default.
       INT(Z'DA3E39CB94B95BDB', KIND=INT64) & ! Recommended default.
  )

  ! The standard multiplier for the PCG random number generation scheme.
  INTEGER(KIND=INT64), PARAMETER :: PCG32_MULT = 6364136223846793005_INT64
  INTEGER(KIND=INT64), PARAMETER :: PCG32_MOD = 2_INT64 ** 64_INT64
  INTEGER(KIND=INT64), PARAMETER :: RIGHT_32 = SHIFTL(1_INT64, 32) - 1  ! 32 rightmost bits.
  INTEGER(KIND=INT64), PARAMETER :: RIGHT_64 = SHIFTL(1_INT64, 64) - 1  ! 64 rightmost bits.

CONTAINS

  ! Seed the random state.
  SUBROUTINE PCG32_SEED_RANDOM(STATE, SEQ, RNG)
    INTEGER(KIND=INT64), INTENT(IN) :: STATE, SEQ
    TYPE(PCG32_STATE), INTENT(INOUT), OPTIONAL, TARGET :: RNG
    TYPE(PCG32_STATE), POINTER :: PCG32
    INTEGER(KIND=INT32) :: SAMPLE
    ! Declare the RNG (use the global if none is provided).
    IF (PRESENT(RNG)) THEN ; PCG32 => RNG
    ELSE ; PCG32 => PCG32_GLOBAL ; END IF
    ! Ensure the increment is positive and odd, is rprime with modulus (2**32).
    PCG32%INC = MOD(2_INT64 * IAND(ABS(SEQ),RIGHT_64) + 1_INT64, PCG32_MOD)
    ! State can be any positive value.
    PCG32%STATE = MOD(PCG32%INC + IAND(ABS(STATE),RIGHT_64), PCG32_MOD)
    ! Draw one sample to initialize (and step past the provided STATE).
    SAMPLE = PCG32_RANDOM(RNG=PCG32)
  END SUBROUTINE PCG32_SEED_RANDOM

  ! Generate a random 32-bit integer.
  FUNCTION PCG32_RANDOM(RNG) RESULT(RANDOM_NUMBER)
    TYPE(PCG32_STATE), INTENT(INOUT), OPTIONAL, TARGET :: RNG
    TYPE(PCG32_STATE), POINTER :: PCG32
    INTEGER(KIND=INT32) :: RANDOM_NUMBER
    INTEGER(KIND=INT64) :: OLDSTATE
    INTEGER(KIND=INT64) :: XORSHIFTED, ROT
    IF (PRESENT(RNG)) THEN ; PCG32 => RNG
    ELSE ; PCG32 => PCG32_GLOBAL ; END IF
    ! Store the previous state, it will be used to generate the output.
    OLDSTATE = PCG32%STATE
    ! Compute the new state with a regular linear generator.
    PCG32%STATE = MOD(OLDSTATE*PCG32_MULT + PCG32%INC, PCG32_MOD)
    ! Generate the output 32 bit random number by computing the operation:
    !    rotate32((state ^ (state >> 18)) >> 27, state >> 59);
    ! 
    ! Increase the randomness of the bottom 18 bits by exclusive-or'ing
    !  them with higher bits. Move entire sequence to the right by 27, 
    !  leaving the top 5 for rotation, and now the bottom 32 are "ready".
    ! 
    ! 18 is half of 32+5=37, meaning it's the "most" we could usefully mix
    !  the higher bits with the lower bits to increase randomness? Not sure.
    XORSHIFTED = IAND(RIGHT_32, SHIFTR(IEOR(OLDSTATE, SHIFTR(OLDSTATE, 18)), 27))
    ! Extract rotation from top 5 bits with a value of [0, 31].
    ROT = SHIFTR(OLDSTATE, 59) ! Extract bits indexed 59, 60, 61, 62, 63.
    ! Compute the final random number by doing a random rotation of the right 32 bits.
    OLDSTATE = IOR( &
         SHIFTR(XORSHIFTED, ROT), &
         SHIFTL(XORSHIFTED, 32 - ROT) &
    )
    ! Using the 'TRANSFER' function copies the lowest 32 bits out of the OLDSTATE and into RANDOM_NUMBER.
    RANDOM_NUMBER = TRANSFER(OLDSTATE, RANDOM_NUMBER)
  END FUNCTION PCG32_RANDOM

  ! Generate a random 32-bit integer in a bounded range.
  FUNCTION PCG32_BOUNDED_RANDOM(BOUND, RNG) RESULT(RANDOM_NUMBER)
    INTEGER(KIND=INT32), INTENT(IN) :: BOUND
    TYPE(PCG32_STATE), INTENT(INOUT), OPTIONAL, TARGET :: RNG
    INTEGER(KIND=INT64) :: THRESHOLD, RAND
    INTEGER(KIND=INT32) :: RANDOM_NUMBER
    ! Make sure that we only use random numbers that come from a range
    !  that is a multiple of "BOUND" by excluding the lower numbers from
    !  the range that can be generated.
    ! 
    THRESHOLD = MOD(PCG32_MOD, INT(BOUND,KIND=INT64))
    DO
       ! For some reason a "1" lands in bit position 32 (0-indexed). Remove it.
       RAND = IAND(RIGHT_32, TRANSFER(PCG32_RANDOM(RNG=RNG), RAND))
       IF (RAND >= THRESHOLD) THEN
          RANDOM_NUMBER = TRANSFER(MOD(RAND, INT(BOUND,KIND=INT64)), RANDOM_NUMBER)
          EXIT
       END IF
    END DO
  END FUNCTION PCG32_BOUNDED_RANDOM

  ! Define a function for generating random 32-bit REAL values.
  FUNCTION PCG32_RANDOM_REAL(RNG) RESULT(RANDOM_NUMBER)
    TYPE(PCG32_STATE), INTENT(INOUT), OPTIONAL, TARGET :: RNG
    REAL(KIND=REAL32) :: RANDOM_NUMBER
    INTEGER(KIND=INT64) :: EXPONENT
    INTEGER(KIND=INT64) :: SIGNIFICAND, TEMP
    INTEGER(KIND=INT64) :: SHIFT
    ! Generate an initial random value.
    SIGNIFICAND = IAND(RIGHT_32, TRANSFER(PCG32_RANDOM(RNG), SIGNIFICAND))
    ! When generating the random real, we must ensure uniform spacing for numbers
    !  smaller than (1/2**32), do this by generating more random bits whenever we
    !  find ourselves in the "smallest bucket".
    EXPONENT = -32
    DO WHILE (SIGNIFICAND .EQ. 0)
       ! Generate a new significand.
       SIGNIFICAND = IAND(RIGHT_32, TRANSFER(PCG32_RANDOM(RNG), SIGNIFICAND))
                                     !  -148 =  -126 + 1 - 23  (32 bits)
       EXPONENT = EXPONENT - 32      ! -1074 = -1022 + 1 - 53  (64 bits)
       ! If the exponent falls below the value (emin + 1 - p),
       ! the exponent of the smallest subnormal, we are
       ! guaranteed the result will be rounded to zero.
       IF (EXPONENT .LT. -148) THEN
          RANDOM_NUMBER = 0.0_REAL32
          RETURN
       ENDIF
    END DO
    ! There is a 1 somewhere in significand, not necessarily in
    ! the most significant position.  If there are leading zeros,
    ! shift them into the exponent and refill the less-significant
    ! bits of the significand.  Can't predict one way or another
    ! whether there are leading zeros: there's a fifty-fifty
    ! chance, if random32 is uniformly distributed.
    SHIFT = LEADZ(SIGNIFICAND) - (8*C_SIZEOF(SIGNIFICAND) - 32) ! Leading zeros in last 32 bits.
    IF (SHIFT .NE. 0) THEN
       EXPONENT = EXPONENT - SHIFT
       SIGNIFICAND = IAND(RIGHT_32, SHIFTL(SIGNIFICAND, SHIFT))
       TEMP = IAND(RIGHT_32, TRANSFER(PCG32_RANDOM(RNG), TEMP))
       SIGNIFICAND = IOR(SIGNIFICAND, SHIFTR(TEMP, 32_INT64 - SHIFT))
    ENDIF
    ! Set the sticky bit, since there is almost surely another 1
    ! in the bit stream.  Otherwise, we might round what looks
    ! like a tie to even when, almost surely, were we to look
    ! further in the bit stream, there would be a 1 breaking the
    ! tie.
    SIGNIFICAND = IOR(SIGNIFICAND, 1_INT64)
    ! Finally, convert to double (rounding) and scale by 2^exponent.
    RANDOM_NUMBER = REAL(SIGNIFICAND, KIND=REAL32) * 2.0_REAL32 ** EXPONENT
    IF ((RANDOM_NUMBER .LT. 0.0) .OR. (RANDOM_NUMBER .GT. 1.0)) THEN
       PRINT *, ''
       PRINT *, 'ERROR: PCG generated an invalid REAL32, bug detected.'
       ! TODO: Print the state (assign default RNG if not provided).
       PRINT *, ''
       STOP 11
    END IF
  END FUNCTION PCG32_RANDOM_REAL

END MODULE PCG32_MODULE


!2023-05-16 08:16:55
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !
! ! Print out a long integer.                                     !
! SUBROUTINE PRINT_LONG(NUM, BITS)                                !
!   INTEGER(KIND=INT64), INTENT(IN) :: NUM                        !
!   INTEGER, INTENT(IN), OPTIONAL :: BITS                         !
!   INTEGER :: I, B, U                                            !
!   ! Set the default for the number of bits to print.            !
!   IF (PRESENT(BITS)) THEN                                       !
!      U = BITS                                                   !
!   ELSE                                                          !
!      U = C_SIZEOF(NUM) * 8                                      !
!   END IF                                                        !
!   ! Print the binary number on one line.                        !
!   WRITE (*,'("  b")',ADVANCE='NO')                              !
!   DO I = U-1, 0, -1                                             !
!      IF (BTEST(NUM, I)) THEN                                    !
!         B = 1                                                   !
!      ELSE                                                       !
!         B = 0                                                   !
!      END IF                                                     !
!      WRITE (*,'(i1)',ADVANCE='NO') B                            !
!      IF (MOD(I,8) .EQ. 0) WRITE (*,'(" ")',ADVANCE='NO')        !
!   END DO                                                        !
!   PRINT *, ''                                                   !
! END SUBROUTINE PRINT_LONG                                       !
!                                                                 !
! ! Print out an integer.                                         !
! SUBROUTINE PRINT_INT(NUM, BITS)                                 !
!   INTEGER(KIND=INT32), INTENT(IN) :: NUM                        !
!   INTEGER, INTENT(IN), OPTIONAL :: BITS                         !
!   INTEGER :: I, B, U                                            !
!   ! Set the default for the number of bits to print.            !
!   IF (PRESENT(BITS)) THEN                                       !
!      U = BITS                                                   !
!   ELSE                                                          !
!      U = C_SIZEOF(NUM) * 8                                      !
!   END IF                                                        !
!   ! Print the binary number on one line.                        !
!   WRITE (*,'("  b")',ADVANCE='NO')                              !
!   DO I = U-1, 0, -1                                             !
!      IF (BTEST(NUM, I)) THEN                                    !
!         B = 1                                                   !
!      ELSE                                                       !
!         B = 0                                                   !
!      END IF                                                     !
!      WRITE (*,'(i1)',ADVANCE='NO') B                            !
!      IF (MOD(I,8) .EQ. 0) WRITE (*,'(" ")',ADVANCE='NO')        !
!   END DO                                                        !
!   PRINT *, ''                                                   !
! END SUBROUTINE PRINT_INT                                        !
!                                                                 !
! PRINT *, ""                                                     !
! PRINT *, "OLDSTATE: ", OLDSTATE                                 !
! CALL PRINT_LONG(OLDSTATE, 65)                                   !
! PRINT *, "ROT: ", ROT                                           !
! CALL PRINT_LONG(ROT, 65)                                        !
! PRINT *, "XORSHIFTED: ", XORSHIFTED                             !
! CALL PRINT_LONG(RIGHT_32, 65)                                   !
! CALL PRINT_LONG(XORSHIFTED, 65)                                 !
! PRINT *, "RIGHT SIDE (of circle)", SHIFTR(XORSHIFTED, ROT)      !
! CALL PRINT_LONG(SHIFTR(XORSHIFTED, ROT), 65)                    !
! PRINT *, "LEFT SIDE (of circle)", SHIFTL(XORSHIFTED, 32 - ROT)  !
! CALL PRINT_LONG(SHIFTL(XORSHIFTED, 32 - ROT), 65)               !
! PRINT *, "final value: ", OLDSTATE                              !
! CALL PRINT_LONG(OLDSTATE, 65)                                   !
! PRINT *, "RANDOM_NUMBER: ", RANDOM_NUMBER                       !
! CALL PRINT_INT(RANDOM_NUMBER)                                   !
! PRINT *, ""                                                     !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !
