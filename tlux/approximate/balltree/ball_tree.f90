MODULE BALL_TREE
  USE ISO_FORTRAN_ENV, ONLY: REAL32, INT64
  USE PRUNE,       ONLY: LEVEL
  USE SWAP,        ONLY: SWAP_I64
  USE FAST_SELECT, ONLY: ARGSELECT
  USE FAST_SORT,   ONLY: ARGSORT
  IMPLICIT NONE

  ! Max bytes for which a doubling of memory footprint (during copy)
  !  is allowed to happen (switches to using scratch file instead).
  INTEGER(KIND=INT64) :: MAX_COPY_BYTES = 2_INT64 ** 32_INT64

CONTAINS

  ! Re-arrange elements of POINTS into a binary ball tree.
  RECURSIVE SUBROUTINE BUILD_TREE(POINTS, SQ_SUMS, RADII, SPLITS, ORDER,&
       ROOT, LEAF_SIZE, COMPUTED_SQ_SUMS)
    REAL(KIND=REAL32),   INTENT(INOUT), DIMENSION(:,:) :: POINTS
    REAL(KIND=REAL32),   INTENT(OUT),   DIMENSION(:) :: SQ_SUMS
    REAL(KIND=REAL32),   INTENT(OUT),   DIMENSION(:) :: RADII
    REAL(KIND=REAL32),   INTENT(OUT),   DIMENSION(:) :: SPLITS
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:) :: ORDER
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: ROOT, LEAF_SIZE
    LOGICAL,             INTENT(IN), OPTIONAL :: COMPUTED_SQ_SUMS
    ! Local variables
    INTEGER(KIND=INT64) :: CENTER_IDX, MID, I, J, LS
    REAL(KIND=REAL32), DIMENSION(SIZE(POINTS,1)) :: PT
    REAL(KIND=REAL32), DIMENSION(SIZE(ORDER)) :: SQ_DISTS
    REAL(KIND=REAL32) :: MAX_SQ_DIST, SQ_DIST, SHIFT
    EXTERNAL :: DGEMM
    ! Set the leaf size to 1 by default (most possible work required,
    ! but guarantees successful use with any leaf size).
    IF (PRESENT(LEAF_SIZE)) THEN ; LS = LEAF_SIZE
    ELSE                         ; LS = 1
    END IF
    ! If no squared sums were provided, compute them.
    IF (.NOT. PRESENT(COMPUTED_SQ_SUMS) .OR. &
         .NOT. COMPUTED_SQ_SUMS) THEN
       !$OMP PARALLEL DO
       DO I = 1, SIZE(POINTS,2)
          SQ_SUMS(I) = SUM(POINTS(:,I)**2)
       END DO
       !$OMP END PARALLEL DO
    END IF
    ! Set the index of the 'root' of the tree.
    IF (PRESENT(ROOT)) THEN ; CENTER_IDX = ROOT
    ELSE
       ! 1) Compute distances between first point (random) and all others.
       ! 2) Pick the furthest point (on conv hull) from first as the center node.
       J = ORDER(1)
       PT(:) = POINTS(:,J)
       SQ_DISTS(1) = 0.0_REAL32
       !$OMP PARALLEL DO
       ROOT_TO_ALL : DO I = 2, SIZE(ORDER)
          SQ_DISTS(I) = SQ_SUMS(J) + SQ_SUMS(ORDER(I)) - &
               2 * DOT_PRODUCT(POINTS(:,ORDER(I)), PT(:))
       END DO ROOT_TO_ALL
       !$OMP END PARALLEL DO
       CENTER_IDX = MAXLOC(SQ_DISTS(:),1)
       ! Now CENTER_IDX is the selected center for this node in tree.
    END IF

    ! Move the "center" to the first position.
    CALL SWAP_I64(ORDER(1), ORDER(CENTER_IDX))
    ! Measure squared distance beween "center" node and all other points.
    J = ORDER(1)
    PT(:) = POINTS(:,J)
    SQ_DISTS(1) = 0.0_REAL32

    !$OMP PARALLEL DO
    CENTER_TO_ALL : DO I = 2, SIZE(ORDER)
       SQ_DISTS(I) = SQ_SUMS(J) + SQ_SUMS(ORDER(I)) - &
            2 * DOT_PRODUCT(POINTS(:,ORDER(I)), PT(:))
    END DO CENTER_TO_ALL
    !$OMP END PARALLEL DO

    ! Base case for recursion, once we have few enough points, exit.
    IF (SIZE(ORDER) .LE. LS) THEN
       SQ_DISTS(1) = MAXVAL(SQ_DISTS(:))
       RADII(ORDER(1)) = SQRT(SQ_DISTS(1))
       SPLITS(ORDER(1)) = RADII(ORDER(1))
       IF (SIZE(ORDER) .GT. 1) THEN
          RADII(ORDER(2:)) = 0.0_REAL32
          SPLITS(ORDER(2:)) = 0.0_REAL32
       END IF
       RETURN
    ELSE IF (SIZE(ORDER) .EQ. 2) THEN
       ! If the leaf size is 1 and there are only 2 elements, store
       ! the radius and exit (since there are no further steps.
       RADII(ORDER(1)) = SQRT(SQ_DISTS(2))
       SPLITS(ORDER(1)) = RADII(ORDER(1))
       RADII(ORDER(2)) = 0.0_REAL32
       SPLITS(ORDER(2)) = 0.0_REAL32
       RETURN
    END IF

    ! Rearrange "SQ_DISTS" about the median value.
    ! Compute the last index that will belong "inside" this node.
    MID = (SIZE(ORDER) + 2) / 2
    CALL ARGSELECT(SQ_DISTS(2:), ORDER(2:), MID - 1)
    SPLITS(ORDER(1)) = SQRT(SQ_DISTS(MID))
    ! Now ORDER has been rearranged such that the median distance
    ! element of POINTS is at the median location.
    ! Identify the furthest point (must be in second half of list).
    I = MID + MAXLOC(SQ_DISTS(MID+1:),1)
    ! Store the "radius" of this ball, the furthest point.
    RADII(ORDER(1)) = SQRT(SQ_DISTS(I))
    ! Move the median point (furthest "interior") to the front (inner root).
    CALL SWAP_I64(ORDER(2), ORDER(MID))
    ! Move the furthest point into the spot after the median (outer root).
    CALL SWAP_I64(ORDER(MID+1), ORDER(I))

    !$OMP PARALLEL NUM_THREADS(2)
    !$OMP SECTIONS
    !$OMP SECTION
    ! Recurisively create this tree.
    !   build a tree with the root being the furthest from this center
    !   for the remaining "interior" points of this center node.
    CALL BUILD_TREE(POINTS, SQ_SUMS, RADII, SPLITS, ORDER(2:MID), 1_INT64, LS, .TRUE.)
    !$OMP SECTION
    !   build a tree with the root being the furthest from this center
    !   for the remaining "exterior" points of this center node.
    !   Only perform this operation if there are >0 points available.
    IF (MID < SIZE(ORDER)) &
         CALL BUILD_TREE(POINTS, SQ_SUMS, RADII, SPLITS, &
         ORDER(MID+1:), 1_INT64, LS, .TRUE.)
    !$OMP END SECTIONS
    !$OMP END PARALLEL
  END SUBROUTINE BUILD_TREE


  ! Compute the K nearest elements of TREE to each point in POINTS.
  SUBROUTINE NEAREST(POINTS, K, TREE, SQ_SUMS, RADII, SPLITS, ORDER, &
       LEAF_SIZE, INDICES, DISTS, TO_SEARCH, RANDOMNESS)
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:,:) :: POINTS, TREE
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:)   :: SQ_SUMS
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:)   :: RADII
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:)   :: SPLITS
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: ORDER
    INTEGER(KIND=INT64), INTENT(IN)               :: K, LEAF_SIZE
    INTEGER(KIND=INT64), INTENT(OUT), DIMENSION(K,SIZE(POINTS,2)) :: INDICES
    REAL(KIND=REAL32),   INTENT(OUT), DIMENSION(K,SIZE(POINTS,2)) :: DISTS
    INTEGER(KIND=INT64), INTENT(IN),  OPTIONAL    :: TO_SEARCH
    REAL(KIND=REAL32),   INTENT(IN),  OPTIONAL    :: RANDOMNESS
    ! Local variables.
    INTEGER(KIND=INT64) :: I, B, BUDGET
    INTEGER(KIND=INT64), DIMENSION(K+LEAF_SIZE+2) :: INDS_BUFFER
    REAL(KIND=REAL32),   DIMENSION(K+LEAF_SIZE+2) :: DISTS_BUFFER
    REAL(KIND=REAL32) :: RAND_PROB
    IF (PRESENT(TO_SEARCH)) THEN ; BUDGET = MAX(K, TO_SEARCH)
    ELSE ; BUDGET = SIZE(ORDER) ; END IF
    IF (PRESENT(RANDOMNESS)) THEN ; RAND_PROB = MAX(0.0_REAL32, MIN(1.0_REAL32,RANDOMNESS)) / 2.0_REAL32
    ELSE IF (BUDGET .LT. SIZE(ORDER)) THEN ; RAND_PROB = 0.01_REAL32
    ELSE ; RAND_PROB = 0.0_REAL32
    END IF
    ! For each point in this set, use the recursive branching
    ! algorithm to identify the nearest elements of TREE.
    !$OMP PARALLEL DO PRIVATE(INDS_BUFFER, DISTS_BUFFER, B)
    DO I = 1, SIZE(POINTS,2)
       B = BUDGET
       CALL PT_NEAREST(POINTS(:,I), K, TREE, SQ_SUMS, RADII, SPLITS, ORDER, &
            LEAF_SIZE, INDS_BUFFER, DISTS_BUFFER, RAND_PROB, CHECKS=B)
       ! Sort the first K elements of the temporary arry for return.
       INDICES(:,I) = INDS_BUFFER(:K)
       DISTS(:,I) = DISTS_BUFFER(:K)
    END DO
    !$OMP END PARALLEL DO
  END SUBROUTINE NEAREST

  ! Compute the K nearest elements of TREE to each point in POINTS.
  RECURSIVE SUBROUTINE PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, SPLITS, &
       ORDER, LEAF_SIZE, INDICES, DISTS, RANDOMNESS, CHECKS, FOUND, PT_SS, D_ROOT)
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:)   :: POINT
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:,:) :: TREE
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:)   :: SQ_SUMS
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:)   :: RADII
    REAL(KIND=REAL32), INTENT(IN), DIMENSION(:)   :: SPLITS
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: ORDER
    INTEGER(KIND=INT64), INTENT(IN)               :: K, LEAF_SIZE
    INTEGER(KIND=INT64), INTENT(OUT), DIMENSION(:) :: INDICES
    REAL(KIND=REAL32),   INTENT(OUT), DIMENSION(:) :: DISTS
    REAL(KIND=REAL32),   INTENT(IN)                :: RANDOMNESS
    INTEGER(KIND=INT64), INTENT(INOUT), OPTIONAL   :: CHECKS
    INTEGER(KIND=INT64), INTENT(INOUT), OPTIONAL   :: FOUND
    REAL(KIND=REAL32),   INTENT(IN),    OPTIONAL   :: PT_SS
    REAL(KIND=REAL32),   INTENT(IN),    OPTIONAL   :: D_ROOT
    ! Local variables
    INTEGER(KIND=INT64) :: F, I, I1, I2, MID, ALLOWED_CHECKS
    REAL(KIND=REAL32)   :: D, D0, D1, D2
    REAL(KIND=REAL32)   :: PS, R
    R = 0.0_REAL32 ! Initialize R (random number).
    ! Initialize FOUND for first call, if FOUND is present then
    ! this must not be first and all are present.
    INITIALIZE : IF (PRESENT(FOUND)) THEN
       ALLOWED_CHECKS = CHECKS
       IF (ALLOWED_CHECKS .LE. 0) RETURN
       F = FOUND
       PS = PT_SS
       D0 = D_ROOT
    ELSE
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
       IF (RANDOMNESS .GT. 0.0_REAL32) CALL RANDOM_NUMBER(R)
       ! Determine which child to search (depth-first search) based
       ! on which child region the point lands in from the root.
       INNER_CHILD_CLOSER : IF ((R .LT. RANDOMNESS) .OR. &
            ((R .LT. 1.0_REAL32-RANDOMNESS) .AND. (D0 .LE. SPLITS(ORDER(1))) )) THEN ! tree heuristic
          ! Search the inner child if it could contain a nearer point.
          SEARCH_INNER1 : IF ((F .LT. K) .OR. (D .GT. D1 - RADII(I1))) THEN
             CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, SPLITS, ORDER(2:MID), &
                  LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D1)
          END IF SEARCH_INNER1
          ! Search the outer child if it could contain a nearer point.
          SEARCH_OUTER1 : IF (((F .LT. K) .OR. (D .GT. D2 - RADII(I2))) &
               .AND. (I2 .NE. I1) .AND. (I2 .GT. 0)) THEN
             CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, SPLITS, ORDER(MID+1:), &
                  LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D2)
          END IF SEARCH_OUTER1
       ELSE
          ! Search the outer child if it could contain a nearer point.
          SEARCH_OUTER2 : IF (((F .LT. K) .OR. (D .GT. D2 - RADII(I2))) &
               .AND. (I2 .NE. I1) .AND. (I2 .GT. 0)) THEN
             CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, SPLITS, ORDER(MID+1:), &
                  LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D2)
          END IF SEARCH_OUTER2
          ! Search the inner child if it could contain a nearer point.
          SEARCH_INNER2 : IF ((F .LT. K) .OR. (D .GT. D1 - RADII(I1))) THEN
             CALL PT_NEAREST(POINT, K, TREE, SQ_SUMS, RADII, SPLITS, ORDER(2:MID), &
                  LEAF_SIZE, INDICES, DISTS, RANDOMNESS, ALLOWED_CHECKS, F, PS, D1)
          END IF SEARCH_INNER2
       END IF INNER_CHILD_CLOSER
    ! Since this is a leaf node, we measure distance to all children.
    ELSE
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
    END IF SORT_K
  END SUBROUTINE PT_NEAREST


  ! Re-organize a built tree so that it is more usefully packed in memory.
  SUBROUTINE FIX_ORDER(POINTS, SQ_SUMS, RADII, SPLITS, ORDER, COPY)
    REAL(KIND=REAL32),   INTENT(INOUT), DIMENSION(:,:) :: POINTS
    REAL(KIND=REAL32),   INTENT(INOUT), DIMENSION(:) :: SQ_SUMS
    REAL(KIND=REAL32),   INTENT(INOUT), DIMENSION(:) :: RADII
    REAL(KIND=REAL32),   INTENT(INOUT), DIMENSION(:) :: SPLITS
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:) :: ORDER
    LOGICAL, INTENT(IN), OPTIONAL :: COPY
    LOGICAL :: SHOULD_COPY
    INTEGER(KIND=INT64) :: I
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
    ! Always copy the square sums and the radii in memory (shouldn't be bad).
    SQ_SUMS(:) = SQ_SUMS(ORDER)
    RADII(:) = RADII(ORDER)
    SPLITS(:) = SPLITS(ORDER)
    ! Reset the order because now it is the expected format.
    FORALL (I=1:SIZE(ORDER)) ORDER(I) = I
  END SUBROUTINE FIX_ORDER

END MODULE BALL_TREE

