
PROGRAM TEST
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE
  REAL :: A(3,3)
  REAL :: B(3)
  REAL :: C(SIZE(B))
  INTEGER :: I, J
  REAL :: D(10000)
  INTEGER :: DI(SIZE(D))
  ! Compute A.
  PRINT *, 'A:'
  DO I = 1, SIZE(A,1)
     A(I,:I) = 1.0
     A(:,I+1:) = 0.0
     PRINT *, '  ', A(I,:)
  END DO
  ! Compute B.
  B(:) = 1.0
  PRINT *, 'B:'
  PRINT *, '  ', B(:)
  ! Compute C.
  C = MATMUL(B(:), A(:,:))
  PRINT *, 'C:'
  PRINT *, '  ', C(:)

  C = MAX(2.0, MATMUL(B(:), A(:,:)))
  PRINT *, 'MAX(2,C):'
  PRINT *, '  ', C(:)
  PRINT *, ''
  test_loop : DO I = 1, 1000
     CALL RANDOM_NUMBER(D(:))
     CALL ARGSORT(D(:), DI(:))
     DO J = 1, SIZE(D)-1
        IF (D(DI(J)) .GT. D(DI(J+1))) THEN
           PRINT *, 'ARGSORT ERROR between indices', J, 'and', J+1
           PRINT *, 'D: ', D(:)
           PRINT *, 'DI:', DI(:)
           EXIT test_loop
        END IF
     END DO
  END DO test_loop

CONTAINS

    SUBROUTINE SWAP_INT(V1, V2)
      INTEGER, INTENT(INOUT) :: V1, V2
      INTEGER :: TEMP
      TEMP = V1
      V1 = V2
      V2 = TEMP
    END SUBROUTINE SWAP_INT


    ! ------------------------------------------------------------------
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
    !   The elements of the array VALUES are sorted and all elements of
    !   INDICES are sorted symmetrically (given INDICES = 1, ...,
    !   SIZE(VALUES) beforehand, final INDICES will show original index
    !   of each element of VALUES before the sort operation).
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
         ! ! Requires 'USE FAST_SELECT' at top of subroutine or module.
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
      REAL(KIND=RT)   :: PIVOT
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
    ! ------------------------------------------------------------------

END PROGRAM TEST
