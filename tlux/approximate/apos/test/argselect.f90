    ! ------------------------------------------------------------------
    !                       FastSelect method
    ! 
    ! Given VALUES list of numbers, rearrange the elements of VALUES
    ! such that the element at index K has rank K (holds its same
    ! location as if all of VALUES were sorted). Symmetrically rearrange
    ! array INDICES to keep track of prior indices.
    ! 
    ! This algorithm uses the same conceptual approach as Floyd-Rivest,
    ! but instead of standard-deviation based selection of bounds for
    ! recursion, a rank-based method is used to pick the subset of
    ! values that is searched. This simplifies the code and improves
    ! interpretability, while achieving the same tunable performance.
    ! 
    ! Arguments:
    ! 
    !   VALUES   --  A 1D array of real numbers.
    !   INDICES  --  A 1D array of original indices for elements of VALUES.
    !   K        --  A positive integer for the rank index about which
    !                VALUES should be rearranged.
    ! Optional:
    ! 
    !   DIVISOR  --  A positive integer >= 2 that represents the
    !                division factor used for large VALUES arrays.
    !   MAX_SIZE --  An integer >= DIVISOR that represents the largest
    !                sized VALUES for which the worst-case pivot value
    !                selection is tolerable. A worst-case pivot causes
    !                O( SIZE(VALUES)^2 ) runtime. This value should be
    !                determined heuristically based on compute hardware.
    ! 
    ! Output:
    ! 
    !   The elements of the array VALUES are rearranged such that the
    !   element at position VALUES(K) is in the same location it would
    !   be if all of VALUES were in sorted order. Also known as,
    !   VALUES(K) has rank K.
    ! 
    RECURSIVE SUBROUTINE ARGSELECT(VALUES, K, INDICES, DIVISOR, MAX_SIZE, RECURSING)
      USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
      ! Arguments
      REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: VALUES
      INTEGER, INTENT(IN) :: K
      INTEGER, INTENT(OUT), DIMENSION(SIZE(VALUES)) :: INDICES
      INTEGER, INTENT(IN), OPTIONAL :: DIVISOR, MAX_SIZE
      LOGICAL, INTENT(IN), OPTIONAL :: RECURSING
      ! Locals
      INTEGER :: LEFT, RIGHT, L, R, MS, D
      REAL(KIND=RT) :: P
      ! Initialize the divisor (for making subsets).
      IF (PRESENT(DIVISOR)) THEN ; D = DIVISOR
      ELSE IF (SIZE(INDICES) .GE. 8388608) THEN ; D = 32 ! 2**5 ! 2**23
      ELSE IF (SIZE(INDICES) .GE. 1048576) THEN ; D = 8  ! 2**3 ! 2**20
      ELSE                                      ; D = 4  ! 2**2
      END IF
      ! Initialize the max size (before subsets are created).
      IF (PRESENT(MAX_SIZE)) THEN ; MS = MAX_SIZE
      ELSE                        ; MS = 1024 ! 2**10
      END IF
      ! When not recursing, set the INDICES to default values.
      IF (.NOT. PRESENT(RECURSING)) THEN
         FORALL(D=1:SIZE(VALUES)) INDICES(D) = D
      END IF
      ! Initialize LEFT and RIGHT to be the entire array.
      LEFT = 1
      RIGHT = SIZE(INDICES)
      ! Loop until done finding the K-th element.
      DO WHILE (LEFT .LT. RIGHT)
         ! Use SELECT recursively to improve the quality of the
         ! selected pivot value for large arrays.
         IF (RIGHT - LEFT .GT. MS) THEN
            ! Compute how many elements should be left and right of K
            ! to maintain the same percentile in a subset.
            L = K - K / D
            R = L + (SIZE(INDICES) / D)
            ! Perform fast select on an array a fraction of the size about K.
            CALL ARGSELECT(VALUES(:), K - L + 1, INDICES(L:R), DIVISOR, MAX_SIZE, RECURSING=.TRUE.)
         END IF
         ! Pick a partition element at position K.
         P = VALUES(INDICES(K))
         L = LEFT
         R = RIGHT
         ! Move the partition element to the front of the list.
         CALL SWAP_INT(INDICES(LEFT), INDICES(K))
         ! Pre-swap the left and right elements (temporarily putting a
         ! larger element on the left) before starting the partition loop.
         IF (VALUES(INDICES(RIGHT)) .GT. P) THEN
            CALL SWAP_INT(INDICES(LEFT), INDICES(RIGHT))
         END IF
         ! Now partition the elements about the pivot value "P".
         DO WHILE (L .LT. R)
            CALL SWAP_INT(INDICES(L), INDICES(R))
            L = L + 1
            R = R - 1
            DO WHILE (VALUES(INDICES(L)) .LT. P) ; L = L + 1 ; END DO
            DO WHILE (VALUES(INDICES(R)) .GT. P) ; R = R - 1 ; END DO
         END DO
         ! Place the pivot element back into its appropriate place.
         IF (VALUES(INDICES(LEFT)) .EQ. P) THEN
            CALL SWAP_INT(INDICES(LEFT), INDICES(R))
         ELSE
            R = R + 1
            CALL SWAP_INT(INDICES(R), INDICES(RIGHT))
         END IF
         ! adjust left and right towards the boundaries of the subset
         ! containing the (k - left + 1)th smallest element
         IF (R .LE. K) LEFT = R + 1
         IF (K .LE. R) RIGHT = R - 1
      END DO

    CONTAINS

      SUBROUTINE SWAP_INT(V1, V2)
        INTEGER, INTENT(INOUT) :: V1, V2
        INTEGER :: TEMP
        TEMP = V1
        V1 = V2
        V2 = TEMP
      END SUBROUTINE SWAP_INT

    END SUBROUTINE ARGSELECT
