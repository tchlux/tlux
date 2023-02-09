! A nearest neighbor tree structure that picks points on the convex hull and
!  splits regions in half by the 2-norm distance to the median child.
!  Construction is parallelized for shared memory architectures with OpenMP,
!  and querying is parallelized over batched query points (but serial for
!  a single query). In addition a nearest neighbor query budget can be
!  provided to generate approximate results, with the guarantee that exact
!  nearest neighbors will be found given a budget greater than the logarithm
!  base two of the number of points in the tree.

MODULE BALL_TREE
  USE ISO_FORTRAN_ENV, ONLY: R32 => REAL32, I64 => INT64, I32 => INT32
  USE PRUNE,       ONLY: LEVEL
  USE SWAP,        ONLY: SWAP_I64
  USE FAST_SELECT, ONLY: ARGSELECT
  USE FAST_SORT,   ONLY: ARGSORT
  IMPLICIT NONE

  ! Max bytes for which a doubling of memory footprint (during copy)
  !  is allowed to happen (switches to using scratch file instead).
  INTEGER(KIND=I64) :: MAX_COPY_BYTES = 2_I64 ** 33_I64 ! 8GB
  INTEGER(KIND=I32) :: NUMBER_OF_THREADS

  ! Function that is defined by OpenMP.
  INTERFACE
     ! Enabling nested parallelization.
     SUBROUTINE OMP_SET_NESTED(NESTED)
       LOGICAL :: NESTED
     END SUBROUTINE OMP_SET_NESTED
     ! Nested parallelism levels.
     FUNCTION OMP_GET_MAX_ACTIVE_LEVELS()
       INTEGER :: OMP_GET_MAX_ACTIVE_LEVELS
     END FUNCTION OMP_GET_MAX_ACTIVE_LEVELS
     SUBROUTINE OMP_SET_MAX_ACTIVE_LEVELS(MAX_LEVELS)
       INTEGER :: MAX_LEVELS
     END SUBROUTINE OMP_SET_MAX_ACTIVE_LEVELS
     ! Number of threads.
     FUNCTION OMP_GET_MAX_THREADS()
       INTEGER :: OMP_GET_MAX_THREADS
     END FUNCTION OMP_GET_MAX_THREADS
     SUBROUTINE OMP_SET_NUM_THREADS(NUM_THREADS)
       INTEGER :: NUM_THREADS
     END SUBROUTINE OMP_SET_NUM_THREADS
     ! Thread number.
     FUNCTION OMP_GET_THREAD_NUM()
       INTEGER :: OMP_GET_THREAD_NUM
     END FUNCTION OMP_GET_THREAD_NUM
  END INTERFACE

CONTAINS

  ! Configure OpenMP parallelism for this ball tree code.
  SUBROUTINE CONFIGURE(NUM_THREADS, MAX_LEVELS, NESTED)
    INTEGER, OPTIONAL :: NUM_THREADS, MAX_LEVELS
    LOGICAL, OPTIONAL :: NESTED
    ! Nested parallelism.
    IF (PRESENT(NESTED)) THEN
       CALL OMP_SET_NESTED(NESTED)
    ELSE
       CALL OMP_SET_NESTED(.TRUE.)
    END IF
    ! Max nested levels of parallelism.
    IF (PRESENT(MAX_LEVELS)) THEN
       CALL OMP_SET_MAX_ACTIVE_LEVELS(MAX_LEVELS)
    ELSE
       CALL OMP_SET_MAX_ACTIVE_LEVELS( &
            1 + INT(CEILING(LOG(REAL(OMP_GET_MAX_THREADS())) / LOG(2.0))) &
       )
    END IF
    ! Number of threads used by default in loops.
    IF (PRESENT(NUM_THREADS)) THEN
       NUMBER_OF_THREADS = NUM_THREADS
       CALL OMP_SET_NUM_THREADS(NUM_THREADS)
    ELSE
       NUMBER_OF_THREADS = OMP_GET_MAX_THREADS()
       CALL OMP_SET_NUM_THREADS(NUMBER_OF_THREADS)
    END IF
  END SUBROUTINE CONFIGURE

  ! Compute the square sums of a bunch of points (with parallelism).
  SUBROUTINE COMPUTE_SQUARE_SUMS(POINTS, SQ_SUMS)
    REAL(KIND=R32), INTENT(IN),  DIMENSION(:,:) :: POINTS
    REAL(KIND=R32), INTENT(OUT), DIMENSION(:) :: SQ_SUMS
    INTEGER :: I
    !$OMP PARALLEL DO
    DO I = 1, SIZE(POINTS,2)
       SQ_SUMS(I) = SUM(POINTS(:,I)**2)
    END DO
  END SUBROUTINE COMPUTE_SQUARE_SUMS

  ! Re-arrange elements of POINTS into a binary ball tree about medians.
  RECURSIVE SUBROUTINE BUILD_TREE(POINTS, SQ_SUMS, RADII, MEDIANS, SQ_DISTS, ORDER, ROOT, LEAF_SIZE)
    REAL(KIND=R32), INTENT(INOUT), DIMENSION(:,:) :: POINTS
    REAL(KIND=R32), INTENT(OUT), DIMENSION(:) :: SQ_SUMS
    REAL(KIND=R32), INTENT(OUT), DIMENSION(:) :: RADII
    REAL(KIND=R32), INTENT(OUT), DIMENSION(:) :: MEDIANS
    REAL(KIND=R32), INTENT(INOUT), DIMENSION(:) :: SQ_DISTS
    INTEGER(KIND=I64), INTENT(INOUT), DIMENSION(:) :: ORDER
    INTEGER(KIND=I64), INTENT(IN), OPTIONAL :: ROOT, LEAF_SIZE
    ! Local variables
    INTEGER(KIND=I64) :: CENTER_IDX, MID, I, J, LS
    REAL(KIND=R32) :: MAX_SQ_DIST, SQ_DIST, SHIFT
    REAL(KIND=R32), ALLOCATABLE, DIMENSION(:) :: PT
    ALLOCATE(PT(1:SIZE(POINTS,1,KIND=I64)))
    ! Set the leaf size to 1 by default (most possible work required,
    ! but guarantees successful use with any leaf size).
    IF (PRESENT(LEAF_SIZE)) THEN ; LS = LEAF_SIZE
    ELSE                         ; LS = 1_I64
    END IF
    ! Set the index of the 'root' of the tree.
    IF (PRESENT(ROOT)) THEN ; CENTER_IDX = ROOT
    ELSE
       ! 1) Compute distances between first point (random) and all others.
       ! 2) Pick the furthest point (on convex hull) from first as the center node.
       J = ORDER(1)
       PT(:) = POINTS(:,J)
       SQ_DISTS(1) = 0.0_R32
       !$OMP PARALLEL DO
       ROOT_TO_ALL : DO I = 2_I64, SIZE(ORDER,KIND=I64)
          SQ_DISTS(I) = SQ_SUMS(J) + SQ_SUMS(ORDER(I)) - &
               2.0_R32 * DOT_PRODUCT(POINTS(:,ORDER(I)), PT(:))
       END DO ROOT_TO_ALL
       CENTER_IDX = MAXLOC(SQ_DISTS(:),1)
       ! Now CENTER_IDX is the selected center for this node in tree.
    END IF

    ! Move the "center" to the first position.
    CALL SWAP_I64(ORDER(1), ORDER(CENTER_IDX))
    ! Measure squared distance beween "center" node and all other points.
    J = ORDER(1)
    PT(:) = POINTS(:,J)
    SQ_DISTS(1) = 0.0_R32

    !$OMP PARALLEL DO
    CENTER_TO_ALL : DO I = 2_I64, SIZE(ORDER,KIND=I64)
       SQ_DISTS(I) = SQ_SUMS(J) + SQ_SUMS(ORDER(I)) - &
            2.0_R32 * DOT_PRODUCT(POINTS(:,ORDER(I)), PT(:))
    END DO CENTER_TO_ALL

    ! Base case for recursion, once we have few enough points, exit.
    IF (SIZE(ORDER,KIND=I64) .LE. LS) THEN
       SQ_DISTS(1) = MAXVAL(SQ_DISTS(:))
       RADII(ORDER(1)) = SQRT(SQ_DISTS(1))
       MEDIANS(ORDER(1)) = RADII(ORDER(1))
       IF (SIZE(ORDER,KIND=I64) .GT. 1_I64) THEN
          RADII(ORDER(2:)) = 0.0_R32
          MEDIANS(ORDER(2:)) = 0.0_R32
       END IF
       RETURN
    ELSE IF (SIZE(ORDER,KIND=I64) .EQ. 2_I64) THEN
       ! If the leaf size is 1 and there are only 2 elements, store
       ! the radius and exit (since there are no further steps.
       RADII(ORDER(1)) = SQRT(SQ_DISTS(2))
       MEDIANS(ORDER(1)) = RADII(ORDER(1))
       RADII(ORDER(2)) = 0.0_R32
       MEDIANS(ORDER(2)) = 0.0_R32
       RETURN
    END IF

    ! Rearrange "SQ_DISTS" about the median value.
    ! Compute the last index that will belong "inside" this node.
    MID = (SIZE(ORDER,KIND=I64) + 2_I64) / 2_I64
    CALL ARGSELECT(SQ_DISTS(2:), ORDER(2:), MID - 1_I64)
    MEDIANS(ORDER(1)) = SQRT(SQ_DISTS(MID))
    ! Now ORDER has been rearranged such that the median distance
    ! element of POINTS is at the median location.
    ! Identify the furthest point (must be in second half of list).
    I = MID + MAXLOC(SQ_DISTS(MID+1_I64:),1)
    ! Store the "radius" of this ball, the furthest point.
    RADII(ORDER(1)) = SQRT(SQ_DISTS(I))
    ! Move the median point (furthest "interior") to the front (inner root).
    CALL SWAP_I64(ORDER(2), ORDER(MID))
    ! Move the furthest point into the spot after the median (outer root).
    CALL SWAP_I64(ORDER(MID+1_I64), ORDER(I))

    !$OMP PARALLEL NUM_THREADS(2)
    !$OMP SECTIONS
    !$OMP SECTION
    ! Recurisively create this tree.
    !   build a tree with the root being the furthest from this center
    !   for the remaining "interior" points of this center node.
    CALL BUILD_TREE(POINTS, SQ_SUMS, RADII, MEDIANS, SQ_DISTS(2_I64:MID), ORDER(2_I64:MID), 1_I64, LS)
    !$OMP SECTION
    !   build a tree with the root being the furthest from this center
    !   for the remaining "exterior" points of this center node.
    !   Only perform this operation if there are >0 points available.
    IF (MID < SIZE(ORDER,KIND=I64)) THEN
       CALL BUILD_TREE(POINTS, SQ_SUMS, RADII, MEDIANS, SQ_DISTS(MID+1_I64:), ORDER(MID+1_I64:), 1_I64, LS)
    END IF
    !$OMP END SECTIONS
    !$OMP END PARALLEL
    DEALLOCATE(PT)
  END SUBROUTINE BUILD_TREE


  ! Compute the K nearest elements of TREE to each point in POINTS.
  SUBROUTINE NEAREST(POINTS, K, TREE, SQ_SUMS, RADII, MEDIANS, ORDER, &
       LEAF_SIZE, INDICES, DISTS, IWORK, RWORK, TO_SEARCH, RANDOMNESS)
    REAL(KIND=R32), INTENT(IN), DIMENSION(:,:) :: POINTS, TREE
    REAL(KIND=R32), INTENT(IN), DIMENSION(:) :: SQ_SUMS
    REAL(KIND=R32), INTENT(IN), DIMENSION(:) :: RADII
    REAL(KIND=R32), INTENT(IN), DIMENSION(:) :: MEDIANS
    INTEGER(KIND=I64), INTENT(IN), DIMENSION(:) :: ORDER
    INTEGER(KIND=I64), INTENT(IN) :: K, LEAF_SIZE
    INTEGER(KIND=I64), INTENT(OUT), DIMENSION(:,:) :: INDICES ! (K, SIZE(POINTS,2))
    REAL(KIND=R32), INTENT(OUT), DIMENSION(:,:) :: DISTS ! (K, SIZE(POINTS,2))
    INTEGER(KIND=I64), INTENT(INOUT), DIMENSION(:,:) :: IWORK ! (K+LEAF_SIZE+2, NUMBER_OF_THEADS)
    REAL(KIND=R32), INTENT(INOUT), DIMENSION(:,:) :: RWORK ! (K+LEAF_SIZE+2, NUMBER_OF_THEADS)
    INTEGER(KIND=I64), INTENT(IN),  OPTIONAL :: TO_SEARCH
    REAL(KIND=R32), INTENT(IN),  OPTIONAL :: RANDOMNESS
    ! Local variables.
    INTEGER(KIND=I64) :: I, T, B, BUDGET, NT
    REAL(KIND=R32) :: RAND_PROB
    IF (SIZE(ORDER) .LE. 0) THEN
       INDICES(:,:) = 0
       DISTS(:,:) = HUGE(DISTS(1,1))
       RETURN
    END IF
    ! Set the budget.
    IF (PRESENT(TO_SEARCH)) THEN
       BUDGET = MAX(K, TO_SEARCH)
    ELSE
       BUDGET = SIZE(ORDER)
    END IF
    ! Set the randomness.
    IF (PRESENT(RANDOMNESS)) THEN
       RAND_PROB = MAX(0.0_R32, MIN(1.0_R32,RANDOMNESS)) / 2.0_R32
    ELSE IF (BUDGET .LT. SIZE(ORDER)) THEN
       RAND_PROB = 0.0_R32
    ELSE
       RAND_PROB = 0.0_R32
    END IF
    ! Compute the number of threads.
    NT = MIN(SIZE(IWORK,2), SIZE(RWORK,2))
    ! For each point in this set, use the recursive branching
    ! algorithm to identify the nearest elements of TREE.
    !$OMP PARALLEL DO PRIVATE(B,T) NUM_THREADS(NT)
    DO I = 1, SIZE(POINTS,2)
       B = BUDGET
       T = OMP_GET_THREAD_NUM()+1
       CALL PT_NEAREST(POINTS(:,I), K, TREE, SQ_SUMS, RADII, MEDIANS, ORDER, &
            LEAF_SIZE, IWORK(:,T), RWORK(:,T), RAND_PROB, CHECKS=B)
       ! Sort the first K elements of the temporary arry for return.
       INDICES(:,I) = IWORK(:K,T)
       DISTS(:,I) = RWORK(:K,T)
    END DO
  END SUBROUTINE NEAREST

  ! Compute the K nearest elements of TREE to each point in POINTS.
  RECURSIVE SUBROUTINE PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, MEDIANS, &
       ORDER, LEAF_SIZE, INDICES, DISTS, RANDOMNESS, CHECKS, FOUND, PT_SS, D_ROOT)
    REAL(KIND=R32), INTENT(IN), DIMENSION(:) :: POINT
    REAL(KIND=R32), INTENT(IN), DIMENSION(:,:) :: TREE
    REAL(KIND=R32), INTENT(IN), DIMENSION(:) :: SQ_SUMS
    REAL(KIND=R32), INTENT(IN), DIMENSION(:) :: RADII
    REAL(KIND=R32), INTENT(IN), DIMENSION(:) :: MEDIANS
    INTEGER(KIND=I64), INTENT(IN), DIMENSION(:) :: ORDER
    INTEGER(KIND=I64), INTENT(IN) :: K, LEAF_SIZE
    INTEGER(KIND=I64), INTENT(OUT), DIMENSION(:) :: INDICES
    REAL(KIND=R32), INTENT(OUT), DIMENSION(:) :: DISTS
    REAL(KIND=R32), INTENT(IN) :: RANDOMNESS
    INTEGER(KIND=I64), INTENT(INOUT), OPTIONAL   :: CHECKS
    INTEGER(KIND=I64), INTENT(INOUT), OPTIONAL   :: FOUND
    REAL(KIND=R32), INTENT(IN), OPTIONAL :: PT_SS
    REAL(KIND=R32), INTENT(IN), OPTIONAL :: D_ROOT
    ! Local variables
    INTEGER(KIND=I64) :: F, I, I1, I2, MID, ALLOWED_CHECKS
    REAL(KIND=R32) :: D, D0, D1, D2, PS, R
    F = 0
    R = 0.0_R32 ! Initialize R (random number).
    ! Initialize FOUND for first call, if FOUND is present then
    ! this must not be first and all are present.
    INITIALIZE : IF (PRESENT(FOUND)) THEN
       ALLOWED_CHECKS = CHECKS
       IF (ALLOWED_CHECKS .LE. 0) RETURN
       F = FOUND
       PS = PT_SS
       D0 = D_ROOT
    ELSE IF (SIZE(ORDER) .GT. 0) THEN ! There is at least one point to check, this is root.
       ! Initialize the remaining checks to search.
       IF (PRESENT(CHECKS)) THEN ; ALLOWED_CHECKS = CHECKS - 1
       ELSE ; ALLOWED_CHECKS = SIZE(ORDER) - 1 ; END IF
       ! Start at index 0 (added onto current index). Compute squared sum.
       PS = SUM(POINT(:)**2)
       ! Measure distance to root.
       INDICES(1) = ORDER(1)
       DISTS(1) = SQRT(PS + SQ_SUMS(ORDER(1)) - &
            2*DOT_PRODUCT(POINT(:), TREE(:,ORDER(1))))
       ! Set the "points found" to be 1.
       F = 1
       D0 = DISTS(1)
    END IF INITIALIZE
    ! If this is NOT a leaf node, then recurse.
    BRANCH_OR_LEAF : IF (SIZE(ORDER) .GT. LEAF_SIZE) THEN
       ALLOWED_CHECKS = ALLOWED_CHECKS - 1
       ! Measure distance to inner child.
       I1 = ORDER(2)
       D1 = SQRT(PS + SQ_SUMS(I1) - &
            2*DOT_PRODUCT(POINT(:),TREE(:,I1)))
       ! Store this distance calculation and index.
       F = F + 1
       INDICES(F) = I1
       DISTS(F) = D1
       ! Measure distance to outer child the same as above, after
       ! checking to see if there *is* an outer child.
       MID = (SIZE(ORDER) + 2) / 2
       IF (MID+1 > SIZE(ORDER)) THEN
          I2 = 0
          D2 = HUGE(D2)
       ELSE
          I2 = ORDER(MID+1)
          IF (I2 .NE. I1) THEN
             ALLOWED_CHECKS = ALLOWED_CHECKS - 1
             D2 = SQRT(PS + SQ_SUMS(I2) - &
                  2*DOT_PRODUCT(POINT(:),TREE(:,I2)))
             ! Store this distance calculation and index.
             F = F + 1
             INDICES(F) = I2
             DISTS(F) = D2
          ELSE ; D2 = HUGE(D2)
          END IF
       END IF
       ! Re-organize the list of closest points, pushing them to first K spots.
       CALL ARGSELECT(DISTS(:F), INDICES(:F), MIN(K,F))
       F = MIN(K,F)
       ! Store the maximum distance.
       D = MAXVAL(DISTS(:F),1)
       ! Generate a random number for randomized traversal of the tree.
       IF (RANDOMNESS .GT. 0.0_R32) CALL RANDOM_NUMBER(R)
       ! Determine which child to search (depth-first search) based
       ! on which child region the point lands in from the root.
       INNER_CHILD_CLOSER : IF ((R .LT. RANDOMNESS) .OR. &
            ((R .LT. 1.0_R32-RANDOMNESS) .AND. (D0 .LE. MEDIANS(ORDER(1))))) THEN ! tree heuristic
          ! Search the inner child if it could contain a nearer point.
          SEARCH_INNER1 : IF ((F .LT. K) .OR. (D .GT. D1 - RADII(I1))) THEN
             CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, MEDIANS, ORDER(2:MID), &
                  LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D1)
          END IF SEARCH_INNER1
          ! Search the outer child if it could contain a nearer point.
          SEARCH_OUTER1 : IF ((I2 .GT. 0) .AND. (I2 .NE. I1)) THEN
             IF ((F .LT. K) .OR. (D .GT. D2 - RADII(I2))) THEN
                CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, MEDIANS, ORDER(MID+1:), &
                     LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D2)
             END IF
          END IF SEARCH_OUTER1
       ELSE
          ! Search the outer child if it could contain a nearer point.
          SEARCH_OUTER2 : IF ((I2 .GT. 0) .AND. (I2 .NE. I1)) THEN
             IF ((F .LT. K) .OR. (D .GT. D2 - RADII(I2))) THEN
                CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, MEDIANS, ORDER(MID+1:), &
                     LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D2)
             END IF
          END IF SEARCH_OUTER2
          ! Search the inner child if it could contain a nearer point.
          SEARCH_INNER2 : IF ((F .LT. K) .OR. (D .GT. D1 - RADII(I1))) THEN
             CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, MEDIANS, ORDER(2:MID), &
                  LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D1)
          END IF SEARCH_INNER2
       END IF INNER_CHILD_CLOSER
    ! Since this is a leaf node, we measure distance to all children.
    ELSE
       ! TODO: Refactor this code to use matrix multiplication instead of a loop.
       DIST_TO_CHILDREN : DO I = 2, SIZE(ORDER)
          IF (ALLOWED_CHECKS .LE. 0) EXIT DIST_TO_CHILDREN
          ALLOWED_CHECKS = ALLOWED_CHECKS - 1
          ! Measure distance to all children of this node.
          D = SQRT(PS + SQ_SUMS(ORDER(I)) - &
               2*DOT_PRODUCT(POINT(:),TREE(:,ORDER(I))))
          ! Store this distance.
          F = F + 1
          DISTS(F) = D
          INDICES(F) = ORDER(I)
       END DO DIST_TO_CHILDREN
       ! Reduce the kept points to only those that are closest.
       CALL ARGSELECT(DISTS(:F), INDICES(:F), MIN(K,F))
       F = MIN(K, F)
    END IF BRANCH_OR_LEAF
    ! Handle closing operations..
    SORT_K : IF (PRESENT(FOUND)) THEN
       ! This is not the root, we need to pass the updated value of
       ! FOUND back up the recrusion stack.
       FOUND = F
       CHECKS = ALLOWED_CHECKS
    ELSE
       ! This is the root, initial caller. Sort the distances for return.
       CALL ARGSELECT(DISTS(:F), INDICES(:F), K)
       CALL ARGSORT(DISTS(:K), INDICES(:K))
       ! Set all unused dists and indices to default values.
       INDICES(F+1:) = 0
       DISTS(F+1:) = HUGE(DISTS(1))
    END IF SORT_K
  END SUBROUTINE PT_NEAREST


  ! Reorganize a built tree so that it is packed in order in memory.
  SUBROUTINE FIX_ORDER(POINTS, SQ_SUMS, RADII, MEDIANS, ORDER, COPY)
    REAL(KIND=R32),   INTENT(INOUT), DIMENSION(:,:) :: POINTS
    REAL(KIND=R32),   INTENT(INOUT), DIMENSION(:) :: SQ_SUMS
    REAL(KIND=R32),   INTENT(INOUT), DIMENSION(:) :: RADII
    REAL(KIND=R32),   INTENT(INOUT), DIMENSION(:) :: MEDIANS
    INTEGER(KIND=I64), INTENT(INOUT), DIMENSION(:) :: ORDER
    LOGICAL, INTENT(IN), OPTIONAL :: COPY
    LOGICAL :: SHOULD_COPY
    INTEGER(KIND=I64) :: I
    ! Default to copy (in memory) if there is less than 1 GB of data.
    IF (PRESENT(COPY)) THEN ; SHOULD_COPY = COPY
    ELSE ; SHOULD_COPY = SIZEOF(POINTS) .LE. MAX_COPY_BYTES
    END IF
    ! Reorder all of the data. Use a scratch file for large data sets.
    IF (SHOULD_COPY) THEN
       POINTS(:,:) = POINTS(:,ORDER)
    ELSE
       ! Open scratch file for writing all of the points in order.
       OPEN(UNIT=1, STATUS='SCRATCH', ACTION='READWRITE', FORM='UNFORMATTED', ACCESS='STREAM')
       ! Write all points to a scratch file in the correct order.
       DO I = 1, SIZE(ORDER)
          WRITE(UNIT=1) POINTS(:,ORDER(I))
       END DO
       ! Read all points from file (they are now ordered correctly).
       READ(UNIT=1,POS=1) POINTS(:,:)
       ! Close scratch file.
       CLOSE(UNIT=1)
    END IF
    ! Always copy the square sums and the radii in memory (only a problem with many billions of points).
    SQ_SUMS(:) = SQ_SUMS(ORDER)
    RADII(:) = RADII(ORDER)
    MEDIANS(:) = MEDIANS(ORDER)
    ! Reset the order because now it is the expected format.
    FORALL (I=1:SIZE(ORDER)) ORDER(I) = I
  END SUBROUTINE FIX_ORDER

  ! Increment the counts for the number of times various indices are referenced.
  !   Example:
  !     usage = [0, 0, 0, 0, 0]  ! counters for ball tree over 5 points
  !     indices = [1, 4, 2, 1]   ! indices of points to increment
  !     CALL BINCOUNT(indices, usage)
  !     usage = [2, 1, 0, 0, 1]  ! updated counters for usage over 5 points
  SUBROUTINE BINCOUNT(INDICES, USAGE)
    INTEGER(KIND=I64), INTENT(IN), DIMENSION(:) :: INDICES ! (K, SIZE(POINTS,2))
    INTEGER(KIND=I64), INTENT(INOUT), DIMENSION(:) :: USAGE
    INTEGER(KIND=I64) :: I
    DO I = 1, SIZE(INDICES, KIND=I64)
       USAGE(INDICES(I)) = USAGE(INDICES(I)) + 1
    END DO
  END SUBROUTINE BINCOUNT

END MODULE BALL_TREE

