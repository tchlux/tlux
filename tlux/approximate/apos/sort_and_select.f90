! A module for fast sorting and selecting of data.
MODULE SORT_AND_SELECT
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE

CONTAINS

  ! Swap the values of two integers.
  SUBROUTINE SWAP_INT(V1, V2)
    INTEGER, INTENT(INOUT) :: V1, V2
    INTEGER :: TEMP
    TEMP = V1
    V1 = V2
    V2 = TEMP
  END SUBROUTINE SWAP_INT

  !                       FastSelect method
  ! 
  ! Given VALUES list of numbers, rearrange the elements of INDICES
  ! such that the element of VALUES at INDICES(K) has rank K (holds
  ! its same location as if all of VALUES were sorted in INDICES).
  ! All elements of VALUES at INDICES(:K-1) are less than or equal,
  ! while all elements of VALUES at INDICES(K+1:) are greater or equal.
  ! 
  ! This algorithm uses a recursive approach to exponentially shrink
  ! the number of indices that have to be considered to find the
  ! element of desired rank, while simultaneously pivoting values
  ! that are less than the target rank left and larger right.
  ! 
  ! Arguments:
  ! 
  !   VALUES   --  A 1D array of real numbers. Will not be modified.
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
  ! Output:
  ! 
  !   INDICES  --  A 1D array of original indices for elements of VALUES.
  ! 
  !   The elements of the array INDICES are rearranged such that the
  !   element at position VALUES(INDICES(K)) is in the same location 
  !   it would be if all of VALUES were referenced in sorted order in
  !   INDICES. Also known as, VALUES(INDICES(K)) has rank K.
  ! 
  RECURSIVE SUBROUTINE ARGSELECT(VALUES, K, INDICES, DIVISOR, MAX_SIZE, RECURSING)
    ! Arguments
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: VALUES
    INTEGER, INTENT(IN) :: K
    INTEGER, INTENT(INOUT), DIMENSION(:) :: INDICES
    INTEGER, INTENT(IN), OPTIONAL :: DIVISOR, MAX_SIZE
    LOGICAL, INTENT(IN), OPTIONAL :: RECURSING
    ! Locals
    INTEGER :: LEFT, RIGHT, L, R, MS, D, I
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
       FORALL(I=1:SIZE(INDICES)) INDICES(I) = I
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
          CALL ARGSELECT(VALUES(:), K - L + 1, INDICES(L:R), &
               DIVISOR=D, MAX_SIZE=MS, RECURSING=.TRUE.)
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
  END SUBROUTINE ARGSELECT
  
  !                         FastSort
  ! 
  ! This routine uses a combination of QuickSort (with modestly
  ! intelligent pivot selection) and Insertion Sort (for small arrays)
  ! to achieve very fast average case sort times for both random and
  ! partially sorted data. The pivot is selected for QuickSort as the
  ! median of the first, middle, and last values in the array.
  ! 
  ! Arguments:
  ! 
  !   VALUES   --  A 1D array of real numbers.
  !   INDICES  --  A 1D array of original indices for elements of VALUES.
  ! 
  ! Optional:
  ! 
  !   MIN_SIZE --  An positive integer that represents the largest
  !                sized VALUES for which a partition about a pivot
  !                is used to reduce the size of a an unsorted array.
  !                Any size less than this will result in the use of
  !                INSERTION_ARGSORT instead of ARGPARTITION.
  ! 
  ! Output:
  ! 
  !   The elements of the array INDICES contain ths sorted order of VALUES.
  ! 
  RECURSIVE SUBROUTINE ARGSORT(VALUES, INDICES, MIN_SIZE, INIT_INDS)
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:) :: VALUES
    INTEGER,       INTENT(INOUT), DIMENSION(:) :: INDICES
    INTEGER,       INTENT(IN), OPTIONAL        :: MIN_SIZE
    LOGICAL,       INTENT(IN), OPTIONAL        :: INIT_INDS
    ! Local variables
    LOGICAL :: INIT
    INTEGER :: I, MS
    IF (PRESENT(MIN_SIZE)) THEN ; MS = MIN_SIZE
    ELSE                        ; MS = 2**6
    END IF
    IF (PRESENT(INIT_INDS)) THEN ; INIT = INIT_INDS
    ELSE                         ; INIT = .TRUE.
    END IF
    ! Initialize all indices (for the first call).
    IF (INIT) THEN
       FORALL (I=1:SIZE(INDICES)) INDICES(I) = I
    END IF
    ! Base case, return.
    IF (SIZE(INDICES) .LT. MS) THEN
       CALL INSERTION_ARGSORT(VALUES, INDICES)
    ! Call this function recursively after pivoting about the median.
    ELSE
       ! ---------------------------------------------------------------
       ! If you are having slow runtime with the selection of pivot values 
       ! provided by ARGPARTITION, then consider using ARGSELECT instead.
       I = ARGPARTITION(VALUES, INDICES)
       ! ---------------------------------------------------------------
       ! I = SIZE(INDICES) / 2
       ! CALL ARGSELECT(VALUES, INDICES, I)
       ! ---------------------------------------------------------------
       CALL ARGSORT(VALUES(:), INDICES(:I-1), MS, INIT_INDS=.FALSE.)
       CALL ARGSORT(VALUES(:), INDICES(I+1:), MS, INIT_INDS=.FALSE.)
    END IF
  END SUBROUTINE ARGSORT

  ! This function efficiently partitions values based on the median
  ! of the first, middle, and last elements of the VALUES array. This
  ! function returns the index of the pivot.
  FUNCTION ARGPARTITION(VALUES, INDICES) RESULT(LEFT)
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:) :: VALUES
    INTEGER,       INTENT(INOUT), DIMENSION(:) :: INDICES
    INTEGER :: LEFT, MID, RIGHT
    REAL(KIND=RT) :: PIVOT
    ! Use the median of the first, middle, and last element as the
    ! pivot. Place the pivot at the end of the array.
    MID = (1 + SIZE(INDICES)) / 2
    ! Swap the first and last elements (if the last is smaller).
    IF (VALUES(INDICES(SIZE(INDICES))) < VALUES(INDICES(1))) THEN
       CALL SWAP_INT(INDICES(1), INDICES(SIZE(INDICES)))
    END IF
    ! Swap the middle and first elements (if the middle is smaller).
    IF (VALUES(INDICES(MID)) < VALUES(INDICES(SIZE(INDICES)))) THEN
       CALL SWAP_INT(INDICES(MID), INDICES(SIZE(INDICES)))       
       ! Swap the last and first elements (if the last is smaller).
       IF (VALUES(INDICES(SIZE(INDICES))) < VALUES(INDICES(1))) THEN
          CALL SWAP_INT(INDICES(1), INDICES(SIZE(INDICES)))
       END IF
    END IF
    ! Set the pivot, LEFT index and RIGHT index (skip the smallest,
    ! which is in location 1, and the pivot at the end).
    PIVOT = VALUES(INDICES(SIZE(INDICES)))
    LEFT  = 2
    RIGHT = SIZE(INDICES) - 1
    ! Partition all elements to the left and right side of the pivot
    ! (left if they are smaller, right if they are bigger).
    DO WHILE (LEFT < RIGHT)
       ! Loop left until we find a value that is greater or equal to pivot.
       DO WHILE (VALUES(INDICES(LEFT)) < PIVOT)
          LEFT = LEFT + 1
       END DO
       ! Loop right until we find a value that is less or equal to pivot (or LEFT).
       DO WHILE (RIGHT .NE. LEFT)
          IF (VALUES(INDICES(RIGHT)) .LT. PIVOT) EXIT
          RIGHT = RIGHT - 1
       END DO
       ! Now we know that [VALUES(RIGHT) < PIVOT < VALUES(LEFT)], so swap them.
       CALL SWAP_INT(INDICES(LEFT), INDICES(RIGHT))
    END DO
    ! The last swap was done even though LEFT == RIGHT, we need to undo.
    CALL SWAP_INT(INDICES(LEFT), INDICES(RIGHT))
    ! Finally, we put the pivot back into its proper location.
    CALL SWAP_INT(INDICES(LEFT), INDICES(SIZE(INDICES)))
  END FUNCTION ARGPARTITION

  ! Insertion sort (best for small lists).
  SUBROUTINE INSERTION_ARGSORT(VALUES, INDICES)
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:) :: VALUES
    INTEGER,       INTENT(INOUT), DIMENSION(:) :: INDICES
    ! Local variables.
    REAL(KIND=RT)   :: TEMP_VAL
    INTEGER :: I, BEFORE, AFTER, TEMP_IND
    ! Return for the base case.
    IF (SIZE(INDICES) .LE. 1) RETURN
    ! Put the smallest value at the front of the list.
    I = MINLOC(VALUES(INDICES(:)),1)
    CALL SWAP_INT(INDICES(1), INDICES(I))
    ! Insertion sort the rest of the array.
    DO I = 3, SIZE(INDICES)
       TEMP_IND = INDICES(I)
       TEMP_VAL = VALUES(TEMP_IND)
       ! Search backwards in the list, 
       BEFORE = I - 1
       AFTER  = I
       DO WHILE (TEMP_VAL .LT. VALUES(INDICES(BEFORE)))
          INDICES(AFTER) = INDICES(BEFORE)
          BEFORE = BEFORE - 1
          AFTER  = AFTER - 1
       END DO
       ! Put the value into its place (where it is greater than the
       ! element before it, but less than all values after it).
       INDICES(AFTER) = TEMP_IND
    END DO
  END SUBROUTINE INSERTION_ARGSORT

END MODULE SORT_AND_SELECT
