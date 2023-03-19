! Module for matrix multiplication (absolutely crucial for APOS speed).
! Includes routines for orthogonalization, computing the SVD, and
! radializing data matrices with the SVD.
MODULE MATRIX_OPERATIONS
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE

CONTAINS

  ! Convenience wrapper routine for calling matrix multiply.
  SUBROUTINE GEMM(OP_A, OP_B, OUT_ROWS, OUT_COLS, INNER_DIM, &
       AB_MULT, A, A_ROWS, B, B_ROWS, C_MULT, C, C_ROWS)
    CHARACTER, INTENT(IN) :: OP_A, OP_B
    INTEGER, INTENT(IN) :: OUT_ROWS, OUT_COLS, INNER_DIM, A_ROWS, B_ROWS, C_ROWS
    REAL(KIND=RT), INTENT(IN) :: AB_MULT, C_MULT
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: B
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: C
    ! Call external single-precision matrix-matrix multiplication
    !  (should be provided by hardware manufacturer, if not use custom).
    EXTERNAL :: SGEMM 
    CALL SGEMM(OP_A, OP_B, OUT_ROWS, OUT_COLS, INNER_DIM, &
       AB_MULT, A, A_ROWS, B, B_ROWS, C_MULT, C, C_ROWS)
    ! ! Fortran intrinsic version of general matrix multiplication routine,
    ! !   first compute the initial values in the output matrix,
    ! C(:,:) = C_MULT * C(:)
    ! !   then compute the matrix multiplication.
    ! IF (OP_A .EQ. 'N') THEN
    !    IF (OP_B .EQ. 'N') THEN
    !       C(:,:) = C(:,:) + AB_MULT * MATMUL(A(:,:), B(:,:))
    !    ELSE
    !       C(:,:) = C(:,:) + AB_MULT * MATMUL(A(:,:), TRANSPOSE(B(:,:)))
    !    END IF
    ! ELSE
    !    IF (OP_B .EQ. 'N') THEN
    !       C(:,:) = C(:,:) + AB_MULT * MATMUL(TRANSPOSE(A(:,:)), B(:,:))
    !    ELSE
    !       C(:,:) = C(:,:) + AB_MULT * MATMUL(TRANSPOSE(A(:,:)), TRANSPOSE(B(:,:)))
    !    END IF
    ! END IF
  END SUBROUTINE GEMM

  ! Orthogonalize and normalize column vectors of A in order.
  SUBROUTINE ORTHONORMALIZE(A)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
    REAL(KIND=RT), DIMENSION(SIZE(A,2)) :: MULTIPLIERS
    REAL(KIND=RT) :: LEN
    INTEGER :: I, J
    DO I = 1, SIZE(A,2)
       LEN = NORM2(A(:,I))
       IF (LEN .GT. 0.0_RT) THEN
          A(:,I) = A(:,I) / LEN
          IF (I .LT. SIZE(A,2)) THEN
             MULTIPLIERS(I+1:) = MATMUL(A(:,I), A(:,I+1:))
             DO J = I+1, SIZE(A,2)
                A(:,J) = A(:,J) - MULTIPLIERS(J) * A(:,I)
             END DO
          END IF
       END IF
    END DO
  END SUBROUTINE ORTHONORMALIZE

  ! Generate randomly distributed vectors on the N-sphere.
  SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
    REAL(KIND=RT), DIMENSION(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2)) :: TEMP_VECS
    REAL(KIND=RT), PARAMETER :: PI = 3.141592653589793
    INTEGER :: I, J
    ! Skip empty vector sets.
    IF (SIZE(COLUMN_VECTORS) .LE. 0) RETURN
    ! Generate random numbers in the range [0,1].
    CALL RANDOM_NUMBER(COLUMN_VECTORS(:,:))
    CALL RANDOM_NUMBER(TEMP_VECS(:,:))
    ! Map the random uniform numbers to a normal distribution.
    COLUMN_VECTORS(:,:) = SQRT(-LOG(COLUMN_VECTORS(:,:))) * COS(PI * TEMP_VECS(:,:))
    ! Make the vectors uniformly distributed on the unit ball (for dimension > 1).
    IF (SIZE(COLUMN_VECTORS,1) .GT. 1) THEN
       ! Normalize all vectors to have unit length.
       DO I = 1, SIZE(COLUMN_VECTORS,2)
          COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / NORM2(COLUMN_VECTORS(:,I))
       END DO
    END IF
    ! Orthonormalize the first components of the column
    !  vectors to ensure those are well spaced.
    I = MIN(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2))
    IF (I .GT. 1) CALL ORTHONORMALIZE(COLUMN_VECTORS(:,1:I))
  END SUBROUTINE RANDOM_UNIT_VECTORS

  ! Orthogonalize and normalize column vectors of A with pivoting.
  SUBROUTINE ORTHOGONALIZE(A, LENGTHS, RANK, ORDER)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(OUT), DIMENSION(SIZE(A,2)) :: LENGTHS
    INTEGER, INTENT(OUT), OPTIONAL :: RANK
    INTEGER, INTENT(OUT), DIMENSION(SIZE(A,2)), OPTIONAL :: ORDER
    REAL(KIND=RT) :: L, VEC(SIZE(A,1)) 
    INTEGER :: I, J, K
    IF (PRESENT(RANK)) RANK = 0
    IF (PRESENT(ORDER)) THEN
       FORALL (I=1:SIZE(A,2)) ORDER(I) = I
    END IF
    column_orthogonolization : DO I = 1, SIZE(A,2)
       LENGTHS(I:) = SUM(A(:,I:)**2, 1)
       ! Pivot the largest magnitude vector to the front.
       J = I-1+MAXLOC(LENGTHS(I:),1)
       IF (J .NE. I) THEN
          IF (PRESENT(ORDER)) THEN
             K = ORDER(I)
             ORDER(I) = ORDER(J)
             ORDER(J) = K
          END IF
          L = LENGTHS(I)
          LENGTHS(I) = LENGTHS(J)
          LENGTHS(J) = L
          VEC(:) = A(:,I)
          A(:,I) = A(:,J)
          A(:,J) = VEC(:)
       END IF
       ! Subtract the first vector from all others.
       IF (LENGTHS(I) .GT. EPSILON(1.0_RT)) THEN
          LENGTHS(I) = SQRT(LENGTHS(I))
          A(:,I) = A(:,I) / LENGTHS(I)
          IF (I .LT. SIZE(A,2)) THEN
             LENGTHS(I+1:) = MATMUL(A(:,I), A(:,I+1:))
             DO J = I+1, SIZE(A,2)
                A(:,J) = A(:,J) - LENGTHS(J) * A(:,I)
             END DO
          END IF
          IF (PRESENT(RANK)) RANK = RANK + 1
       ELSE
          LENGTHS(I:) = 0.0_RT
          EXIT column_orthogonolization
       END IF
    END DO column_orthogonolization
  END SUBROUTINE ORTHOGONALIZE

  ! Compute the singular values and right singular vectors for matrix A.
  SUBROUTINE SVD(A, S, VT, RANK, STEPS, BIAS)
    IMPLICIT NONE
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(OUT), DIMENSION(MIN(SIZE(A,1),SIZE(A,2))) :: S
    REAL(KIND=RT), INTENT(OUT), DIMENSION(MIN(SIZE(A,1),SIZE(A,2)),MIN(SIZE(A,1),SIZE(A,2))) :: VT
    INTEGER, INTENT(OUT), OPTIONAL :: RANK
    INTEGER, INTENT(IN), OPTIONAL :: STEPS
    REAL(KIND=RT), INTENT(IN), OPTIONAL :: BIAS
    ! Local variables.
    REAL(KIND=RT), DIMENSION(MIN(SIZE(A,1),SIZE(A,2)),MIN(SIZE(A,1),SIZE(A,2))) :: ATA, Q
    INTEGER :: I, J, K, NUM_STEPS
    REAL(KIND=RT) :: MULTIPLIER
    EXTERNAL :: SSYRK
    ! Set the number of steps.
    IF (PRESENT(STEPS)) THEN
       NUM_STEPS = STEPS
    ELSE
       NUM_STEPS = 1
    END IF
    ! Set "K" (the number of components).
    K = MIN(SIZE(A,1),SIZE(A,2))
    ! Find the multiplier on A.
    MULTIPLIER = MAXVAL(ABS(A(:,:)))
    IF (MULTIPLIER .EQ. 0.0_RT) THEN
       S(:) = 0.0_RT
       VT(:,:) = 0.0_RT
       RETURN
    END IF
    IF (PRESENT(BIAS)) MULTIPLIER = MULTIPLIER / BIAS
    MULTIPLIER = 1.0_RT / MULTIPLIER
    ! Compute ATA.
    IF (SIZE(A,1) .LE. SIZE(A,2)) THEN
       ! ATA(:,:) = MATMUL(AT(:,:), TRANSPOSE(AT(:,:)))
       CALL SSYRK('U', 'N', K, SIZE(A,2), MULTIPLIER**2, A(:,:), &
            SIZE(A,1), 0.0_RT, ATA(:,:), K)
    ELSE
       ! ATA(:,:) = MATMUL(TRANSPOSE(A(:,:)), A(:,:))
       CALL SSYRK('U', 'T', K, SIZE(A,1), MULTIPLIER**2, A(:,:), &
            SIZE(A,1), 0.0_RT, ATA(:,:), K)
    END IF
    ! Copy the upper diagnoal portion into the lower diagonal portion.
    DO I = 1, K
       ATA(I+1:,I) = ATA(I,I+1:)
    END DO
    ! Compute initial right singular vectors.
    VT(:,:) = ATA(:,:)
    ! Orthogonalize and reorder by magnitudes.
    CALL ORTHOGONALIZE(VT(:,:), S(:), RANK)
    ! Do power iterations.
    power_iteration : DO I = 1, NUM_STEPS
       Q(:,:) = VT(:,:)
       ! Q(:,:) = MATMUL(TRANSPOSE(ATA(:,:)), QTEMP(:,:))
       CALL GEMM('N', 'N', K, K, K, 1.0_RT, &
            ATA(:,:), K, Q(:,:), K, 0.0_RT, &
            VT(:,:), K)
       CALL ORTHOGONALIZE(VT(:,:), S(:), RANK)
    END DO power_iteration
    ! Compute the singular values.
    WHERE (S(:) .NE. 0.0_RT)
       S(:) = SQRT(S(:)) / MULTIPLIER
    END WHERE
  END SUBROUTINE SVD

  ! If there are at least as many data points as dimension, then
  ! compute the principal components and rescale the data by
  ! projecting onto those and rescaling so that each component has
  ! identical singular values (this makes the data more "radially
  ! symmetric").
  SUBROUTINE RADIALIZE(X, SHIFT, VECS, INVERT_RESULT, FLATTEN, STEPS)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: SHIFT
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: VECS
    LOGICAL, INTENT(IN), OPTIONAL :: INVERT_RESULT
    LOGICAL, INTENT(IN), OPTIONAL :: FLATTEN
    INTEGER, INTENT(IN), OPTIONAL :: STEPS
    ! Local variables.
    LOGICAL :: INVERSE, FLAT
    REAL(KIND=RT), DIMENSION(SIZE(VECS,1),SIZE(VECS,2)) :: TEMP_VECS
    REAL(KIND=RT), DIMENSION(SIZE(X,1)) :: VALS
    REAL(KIND=RT), DIMENSION(SIZE(X,1), SIZE(X,2)) :: X1
    REAL(KIND=RT) :: RN
    INTEGER :: I, D
    ! Set the default value for "INVERSE".
    IF (PRESENT(INVERT_RESULT)) THEN
       INVERSE = INVERT_RESULT
    ELSE
       INVERSE = .FALSE.
    END IF
    ! Set the default value for "FLAT".
    IF (PRESENT(FLATTEN)) THEN
       FLAT = FLATTEN
    ELSE
       FLAT = .TRUE.
    END IF
    ! Shift the data to be be centered about the origin.
    D = SIZE(X,1)
    RN = REAL(SIZE(X,2),RT)
    SHIFT(:) = -SUM(X(:,:),2) / RN
    DO I = 1, D
       X(I,:) = X(I,:) + SHIFT(I)
    END DO
    ! Set the unused portion of the "VECS" matrix to the identity.
    VECS(D+1:,D+1:) = 0.0_RT
    DO I = D+1, SIZE(VECS,1)
       VECS(I,I) = 1.0_RT
    END DO
    ! Find the directions along which the data is most elongated.
    IF (PRESENT(STEPS)) THEN
       CALL SVD(X, VALS, VECS(:D,:D), STEPS=STEPS)
    ELSE
       CALL SVD(X, VALS, VECS(:D,:D), STEPS=10)
    END IF
    ! Normalize the values to make the output componentwise unit mean squared magnitude.
    IF (FLAT) THEN
       VALS(:) = VALS(:) / SQRT(RN)
       ! For all nonzero vectors, rescale them so that the
       !  average squared distance from zero is exactly 1.
       DO I = 1, D
          IF (VALS(I) .GT. 0.0_RT) THEN
             VECS(:,I) = VECS(:,I) / VALS(I)
          END IF
       END DO
    ELSE
       ! Rescale all vectors by the average magnitude.
       VALS(:) = SUM(VALS(:)) / (SQRT(RN) * REAL(D,RT))
       ! ! Rescale all vectors by the magnitude of the first.
       ! VALS(:) = VALS(1) / SQRT(RN)
       IF (VALS(1) .GT. 0.0_RT) THEN
          VECS(:,:) = VECS(:,:) / VALS(1)
       END IF
    END IF
    ! Apply the column vectors to the data to make it radially symmetric.
    X1(:,:) = X(:,:)
    CALL GEMM('T', 'N', D, SIZE(X,2), D, 1.0_RT, &
         VECS(:D,:D), D, &
         X1(:,:), D, &
         0.0_RT, X(:,:), D)
    ! Compute the inverse of the transformation if requested.
    IF (INVERSE) THEN
       VALS(:) = VALS(:)**2
       DO I = 1, D
          IF (VALS(I) .GT. 0.0_RT) THEN
             VECS(:D,I) = VECS(:D,I) * VALS(I)
          END IF
       END DO
       VECS(:D,:D) = TRANSPOSE(VECS(:D,:D))
       SHIFT(:) = -SHIFT(:)
    END IF
  END SUBROUTINE RADIALIZE

END MODULE MATRIX_OPERATIONS

! ---------------------------------------------------------------------------

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


! ---------------------------------------------------------------------------


! An apositional (/aggregate) and positional piecewise linear regression model.
MODULE APOS
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, INT64, INT8
  USE MATRIX_OPERATIONS, ONLY: GEMM, RANDOM_UNIT_VECTORS, ORTHOGONALIZE, RADIALIZE
  USE SORT_AND_SELECT, ONLY: ARGSORT, ARGSELECT

  IMPLICIT NONE

  PRIVATE :: MODEL_GRADIENT

  ! Model configuration, internal sizes and fit parameters.
  TYPE, BIND(C) :: MODEL_CONFIG
     ! Apositional model configuration.
     INTEGER :: ADN      ! apositional dimension numeric (input)
     INTEGER :: ADI      ! apositional dimension of input
     INTEGER :: ADS = 32 ! apositional dimension of state
     INTEGER :: ANS = 8  ! apositional number of states
     INTEGER :: ADSO     ! apositional dimension of state output
     INTEGER :: ADO      ! apositional dimension of output
     INTEGER :: ANE = 0  ! apositional number of embeddings
     INTEGER :: ADE = 0  ! apositional dimension of embeddings
     ! (Positional) model configuration.
     INTEGER :: MDN      ! model dimension numeric (input)
     INTEGER :: MDI      ! model dimension of input
     INTEGER :: MDS = 32 ! model dimension of state
     INTEGER :: MNS = 8  ! model number of states
     INTEGER :: MDSO     ! model dimension of state output
     INTEGER :: MDO      ! model dimension of output
     INTEGER :: MNE = 0  ! model number of embeddings
     INTEGER :: MDE = 0  ! model dimension of embeddings
     ! Summary numbers that are computed.
     INTEGER :: TOTAL_SIZE
     INTEGER :: NUM_VARS
     ! Index subsets of total size vector naming scheme:
     !   M___ -> model,   A___ -> apositional (/ aggregate) model
     !   _S__ -> start,   _E__ -> end
     !   __I_ -> input,   __S_ -> states, __O_ -> output, __E_ -> embedding
     !   ___V -> vectors, ___S -> shifts
     INTEGER :: ASIV, AEIV, ASIS, AEIS ! apositional input
     INTEGER :: ASSV, AESV, ASSS, AESS ! apositional states
     INTEGER :: ASOV, AEOV             ! apositional output
     INTEGER :: ASEV, AEEV             ! apositional embedding
     INTEGER :: MSIV, MEIV, MSIS, MEIS ! model input
     INTEGER :: MSSV, MESV, MSSS, MESS ! model states
     INTEGER :: MSOV, MEOV             ! model output
     INTEGER :: MSEV, MEEV             ! model embedding
     ! Index subsets for input and output shifts.
     ! M___ -> model,       A___ -> apositional (/ aggregate) model
     ! _IS_ -> input shift, _OS_ -> output shift
     ! ___S -> start,       ___E -> end
     INTEGER :: AISS, AISE, AOSS, AOSE
     INTEGER :: MISS, MISE, MOSS, MOSE
     ! Function parameter.
     REAL(KIND=RT) :: DISCONTINUITY = 0.0_RT
     ! Initialization related parameters.
     REAL(KIND=RT) :: INITIAL_SHIFT_RANGE = 1.0_RT
     REAL(KIND=RT) :: INITIAL_OUTPUT_SCALE = 0.1_RT
     ! Optimization related parameters.
     REAL(KIND=RT) :: STEP_FACTOR = 0.001_RT
     REAL(KIND=RT) :: STEP_MEAN_CHANGE = 0.1_RT
     REAL(KIND=RT) :: STEP_CURV_CHANGE = 0.01_RT
     REAL(KIND=RT) :: STEP_AY_CHANGE = 0.05_RT
     REAL(KIND=RT) :: FASTER_RATE = 1.01_RT
     REAL(KIND=RT) :: SLOWER_RATE = 0.99_RT
     REAL(KIND=RT) :: MIN_UPDATE_RATIO = 0.05_RT
     INTEGER :: MIN_STEPS_TO_STABILITY = 1
     INTEGER :: NUM_THREADS = 1
     INTEGER :: PRINT_DELAY_SEC = 3
     INTEGER :: STEPS_TAKEN = 0
     INTEGER :: LOGGING_STEP_FREQUENCY = 10
     INTEGER :: NUM_TO_UPDATE = HUGE(0)
     LOGICAL(KIND=INT8) :: AX_NORMALIZED = .FALSE.
     LOGICAL(KIND=INT8) :: AXI_NORMALIZED = .FALSE.
     LOGICAL(KIND=INT8) :: AY_NORMALIZED = .FALSE.
     LOGICAL(KIND=INT8) :: X_NORMALIZED = .FALSE.
     LOGICAL(KIND=INT8) :: XI_NORMALIZED = .FALSE.
     LOGICAL(KIND=INT8) :: Y_NORMALIZED = .FALSE.
     LOGICAL(KIND=INT8) :: ENCODE_NORMALIZATION = .TRUE.
     LOGICAL(KIND=INT8) :: APPLY_SHIFT = .TRUE.
     LOGICAL(KIND=INT8) :: KEEP_BEST = .TRUE.
     LOGICAL(KIND=INT8) :: EARLY_STOP = .TRUE.
     ! Descriptions of the number of points that can be in one batch.
     INTEGER(KIND=INT64) :: RWORK_SIZE = 0
     INTEGER(KIND=INT64) :: IWORK_SIZE = 0
     INTEGER(KIND=INT64) :: NA = 0
     INTEGER(KIND=INT64) :: NM = 0
     ! Optimization work space start and end indices.
     INTEGER(KIND=INT64) :: SMG, EMG ! MODEL_GRAD(NUM_VARS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMGM, EMGM ! MODEL_GRAD_MEAN(NUM_VARS)
     INTEGER(KIND=INT64) :: SMGC, EMGC ! MODEL_GRAD_CURV(NUM_VARS)
     INTEGER(KIND=INT64) :: SBM, EBM ! BEST_MODEL(NUM_VARS)
     INTEGER(KIND=INT64) :: SAXS, EAXS ! A_STATES(NA,ADS,ANS+1)
     INTEGER(KIND=INT64) :: SAXG, EAXG ! A_GRADS(NA,ADS,ANS+1)
     INTEGER(KIND=INT64) :: SAY, EAY ! AY(NA,ADO)
     INTEGER(KIND=INT64) :: SAYG, EAYG ! AY_GRAD(NA,ADO)
     INTEGER(KIND=INT64) :: SMXS, EMXS ! M_STATES(NM,MDS,MNS+1)
     INTEGER(KIND=INT64) :: SMXG, EMXG ! M_GRADS(NM,MDS,MNS+1)
     INTEGER(KIND=INT64) :: SYG, EYG ! Y_GRADIENT(MDO,NM)
     INTEGER(KIND=INT64) :: SAXR, EAXR ! AX_RESCALE(ADN,ADN)
     INTEGER(KIND=INT64) :: SAXIS, EAXIS ! AXI_SHIFT(ADE)
     INTEGER(KIND=INT64) :: SAXIR, EAXIR ! AXI_RESCALE(ADE,ADE)
     INTEGER(KIND=INT64) :: SAYR, EAYR ! AY_RESCALE(ADO)
     INTEGER(KIND=INT64) :: SMXR, EMXR ! X_RESCALE(MDN,MDN)
     INTEGER(KIND=INT64) :: SMXIS, EMXIS ! XI_SHIFT(MDE)
     INTEGER(KIND=INT64) :: SMXIR, EMXIR ! XI_RESCALE(MDE,MDE)
     INTEGER(KIND=INT64) :: SYR, EYR ! Y_RESCALE(MDO,MDO)
     INTEGER(KIND=INT64) :: SAL, EAL ! A_LENGTHS(ADS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SML, EML ! M_LENGTHS(MDS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SAST, EAST ! A_STATE_TEMP(NA,ADS)
     INTEGER(KIND=INT64) :: SMST, EMST ! M_STATE_TEMP(NM,MDS)
     ! Integer workspace (for optimization).
     INTEGER(KIND=INT64) :: SUI, EUI ! UPDATE_INDICES(NUM_VARS)
     INTEGER(KIND=INT64) :: SBAS, EBAS ! BATCHA_STARTS(NUM_THREADS)
     INTEGER(KIND=INT64) :: SBAE, EBAE ! BATCHA_ENDS(NUM_THREADS)
     INTEGER(KIND=INT64) :: SBMS, EBMS ! BATCHM_STARTS(NUM_THREADS)
     INTEGER(KIND=INT64) :: SBME, EBME ! BATCHM_ENDS(NUM_THREADS)
     INTEGER(KIND=INT64) :: SAO, EAO ! A_ORDER(ADS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMO, EMO ! M_ORDER(MDS,NUM_THREADS)
  END TYPE MODEL_CONFIG

  ! Function that is defined by OpenMP.
  INTERFACE
     FUNCTION OMP_GET_MAX_THREADS()
       INTEGER :: OMP_GET_MAX_THREADS
     END FUNCTION OMP_GET_MAX_THREADS
  END INTERFACE

CONTAINS

  ! FUNCTION OMP_GET_MAX_THREADS()
  !   INTEGER :: OMP_GET_MAX_THREADS
  !   OMP_GET_MAX_THREADS = 1
  ! END FUNCTION OMP_GET_MAX_THREADS
  ! FUNCTION OMP_GET_THREAD_NUM()
  !   INTEGER :: OMP_GET_THREAD_NUM
  !   OMP_GET_THREAD_NUM = 0
  ! END FUNCTION OMP_GET_THREAD_NUM

  ! Generate a model configuration given state parameters for the model.
  SUBROUTINE NEW_MODEL_CONFIG(ADN, ADE, ANE, ADS, ANS, ADO, &
       MDN, MDE, MNE, MDS, MNS, MDO, NUM_THREADS, CONFIG)
     ! Size related parameters.
     INTEGER, INTENT(IN) :: ADN, MDN
     INTEGER, INTENT(IN) :: MDO
     INTEGER, OPTIONAL, INTENT(IN) :: ADO
     INTEGER, OPTIONAL, INTENT(IN) :: ADS, MDS
     INTEGER, OPTIONAL, INTENT(IN) :: ANS, MNS
     INTEGER, OPTIONAL, INTENT(IN) :: ANE, MNE
     INTEGER, OPTIONAL, INTENT(IN) :: ADE, MDE
     INTEGER, OPTIONAL, INTENT(IN) :: NUM_THREADS
     ! Output
     TYPE(MODEL_CONFIG), INTENT(OUT) :: CONFIG
     ! ---------------------------------------------------------------
     ! ANE
     IF (PRESENT(ANE)) CONFIG%ANE = ANE
     ! ADE
     IF (PRESENT(ADE)) THEN
        CONFIG%ADE = ADE
     ELSE IF (CONFIG%ANE .GT. 0) THEN
        ! Compute a reasonable default dimension (tied to volume of space).
        CONFIG%ADE = MAX(1, 2 * CEILING(LOG(REAL(CONFIG%ANE,RT)) / LOG(2.0_RT)))
        IF (CONFIG%ANE .LE. 3) CONFIG%ADE = CONFIG%ADE - 1
     END IF
     ! ADN, ADI
     CONFIG%ADN = ADN
     CONFIG%ADI = CONFIG%ADN + CONFIG%ADE
     ! ANS
     IF (PRESENT(ANS)) THEN
        CONFIG%ANS = ANS
     ELSE IF (CONFIG%ADI .EQ. 0) THEN
        CONFIG%ANS = 0
     END IF
     ! ADS
     IF (PRESENT(ADS)) THEN
        CONFIG%ADS = ADS
     END IF
     CONFIG%ADSO = CONFIG%ADS
     IF (CONFIG%ANS .EQ. 0) THEN
        CONFIG%ADS = 0
        CONFIG%ADSO = CONFIG%ADI
     END IF
     ! ADO
     IF (PRESENT(ADO)) THEN
        CONFIG%ADO = ADO
     ELSE IF (CONFIG%ADI .EQ. 0) THEN
        CONFIG%ADO = 0
     ELSE
        CONFIG%ADO = MIN(32, CONFIG%ADS)
     END IF
     ! ---------------------------------------------------------------
     ! MNE
     IF (PRESENT(MNE)) CONFIG%MNE = MNE
     ! MDE
     IF (PRESENT(MDE)) THEN
        CONFIG%MDE = MDE
     ELSE IF (CONFIG%MNE .GT. 0) THEN
        ! Compute a reasonable default dimension (tied to volume of space).
        CONFIG%MDE = MAX(1, 1 + CEILING(LOG(REAL(CONFIG%MNE,RT)) / LOG(2.0_RT)))
        IF (CONFIG%MNE .LE. 3) CONFIG%MDE = CONFIG%MDE - 1
     END IF
     ! MDN, MDI
     CONFIG%MDN = MDN
     CONFIG%MDI = CONFIG%MDN + CONFIG%MDE + CONFIG%ADO
     ! MNS
     IF (PRESENT(MNS)) CONFIG%MNS = MNS
     ! MDS
     IF (PRESENT(MDS)) THEN
        CONFIG%MDS = MDS
     END IF
     CONFIG%MDSO = CONFIG%MDS
     IF (CONFIG%MNS .EQ. 0) THEN
        CONFIG%MDS = 0
        CONFIG%MDSO = CONFIG%MDI
     END IF
     ! MDO
     CONFIG%MDO = MDO
     IF (CONFIG%MDO .EQ. 0) THEN
        CONFIG%MDI = 0
        CONFIG%MDE = 0
        CONFIG%MDS = 0
        CONFIG%MNS = 0
        CONFIG%MDSO = 0
     END IF
     ! ---------------------------------------------------------------
     ! NUM_THREADS
     IF (PRESENT(NUM_THREADS)) THEN
        CONFIG%NUM_THREADS = NUM_THREADS
     ELSE
        CONFIG%NUM_THREADS = OMP_GET_MAX_THREADS()
     END IF
     ! Compute indices related to the parameter vector for this model.
     CONFIG%TOTAL_SIZE = 0
     ! ---------------------------------------------------------------
     !   apositional input vecs
     CONFIG%ASIV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEIV = CONFIG%ASIV-1  +  CONFIG%ADI * CONFIG%ADS
     CONFIG%TOTAL_SIZE = CONFIG%AEIV
     !   apositional input shift
     CONFIG%ASIS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEIS = CONFIG%ASIS-1  +  CONFIG%ADS
     CONFIG%TOTAL_SIZE = CONFIG%AEIS
     !   apositional state vecs
     CONFIG%ASSV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AESV = CONFIG%ASSV-1  +  CONFIG%ADS * CONFIG%ADS * MAX(0,CONFIG%ANS-1)
     CONFIG%TOTAL_SIZE = CONFIG%AESV
     !   apositional state shift
     CONFIG%ASSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AESS = CONFIG%ASSS-1  +  CONFIG%ADS * MAX(0,CONFIG%ANS-1)
     CONFIG%TOTAL_SIZE = CONFIG%AESS
     !   apositional output vecs
     CONFIG%ASOV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEOV = CONFIG%ASOV-1  +  CONFIG%ADSO * CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AEOV
     !   apositional embedding vecs
     CONFIG%ASEV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEEV = CONFIG%ASEV-1  +  CONFIG%ADE * CONFIG%ANE
     CONFIG%TOTAL_SIZE = CONFIG%AEEV
     ! ---------------------------------------------------------------
     !   model input vecs
     CONFIG%MSIV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MEIV = CONFIG%MSIV-1  +  CONFIG%MDI * CONFIG%MDS
     CONFIG%TOTAL_SIZE = CONFIG%MEIV
     !   model input shift
     CONFIG%MSIS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MEIS = CONFIG%MSIS-1  +  CONFIG%MDS
     CONFIG%TOTAL_SIZE = CONFIG%MEIS
     !   model state vecs
     CONFIG%MSSV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MESV = CONFIG%MSSV-1  +  CONFIG%MDS * CONFIG%MDS * MAX(0,CONFIG%MNS-1)
     CONFIG%TOTAL_SIZE = CONFIG%MESV
     !   model state shift
     CONFIG%MSSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MESS = CONFIG%MSSS-1  +  CONFIG%MDS * MAX(0,CONFIG%MNS-1)
     CONFIG%TOTAL_SIZE = CONFIG%MESS
     !   model output vecs
     CONFIG%MSOV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MEOV = CONFIG%MSOV-1  +  CONFIG%MDSO * CONFIG%MDO
     CONFIG%TOTAL_SIZE = CONFIG%MEOV
     !   model embedding vecs
     CONFIG%MSEV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MEEV = CONFIG%MSEV-1  +  CONFIG%MDE * CONFIG%MNE
     CONFIG%TOTAL_SIZE = CONFIG%MEEV
     ! THIS IS SPECIAL, IT IS PART OF MODEL AND CHANGES DURING TRAINING
     !   apositional output shift
     CONFIG%AOSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AOSE = CONFIG%AOSS-1 + CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AOSE
     ! ---------------------------------------------------------------
     !   number of variables
     CONFIG%NUM_VARS = CONFIG%TOTAL_SIZE
     ! ---------------------------------------------------------------
     !   apositional input shift
     CONFIG%AISS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AISE = CONFIG%AISS-1 + CONFIG%ADN
     CONFIG%TOTAL_SIZE = CONFIG%AISE
     !   model input shift
     CONFIG%MISS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MISE = CONFIG%MISS-1 + CONFIG%MDN
     CONFIG%TOTAL_SIZE = CONFIG%MISE
     !   model output shift
     CONFIG%MOSS = 1 + CONFIG%TOTAL_SIZE
     IF (CONFIG%MDO .GT. 0) THEN
        CONFIG%MOSE = CONFIG%MOSS-1 + CONFIG%MDO
     ELSE
        CONFIG%MOSE = CONFIG%MOSS-1 + CONFIG%ADO
     END IF
     CONFIG%TOTAL_SIZE = CONFIG%MOSE
  END SUBROUTINE NEW_MODEL_CONFIG

  ! Given a number of X points "NM", and a number of apositional X points
  ! "NA", update the "RWORK_SIZE" and "IWORK_SIZE" attributes in "CONFIG"
  ! as well as all related work indices for that size data.
  SUBROUTINE NEW_FIT_CONFIG(NM, NA, CONFIG)
    INTEGER, INTENT(IN) :: NM, NA
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    INTEGER :: DY
    CONFIG%NM = NM
    CONFIG%NA = NA
    ! Get the output dimension.
    IF (CONFIG%MDO .EQ. 0) THEN
       DY = CONFIG%ADO
    ELSE
       DY = CONFIG%MDO
    END IF
    ! ------------------------------------------------------------
    ! Set up the real valued work array.
    CONFIG%RWORK_SIZE = 0
    ! model gradient
    CONFIG%SMG = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMG = CONFIG%SMG-1 + CONFIG%NUM_VARS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EMG
    ! model gradient mean
    CONFIG%SMGM = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMGM = CONFIG%SMGM-1 + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EMGM
    ! model gradient curvature
    CONFIG%SMGC = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMGC = CONFIG%SMGC-1 + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EMGC
    ! best model
    CONFIG%SBM = 1 + CONFIG%RWORK_SIZE
    CONFIG%EBM = CONFIG%SBM-1 + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EBM
    ! apositional states
    CONFIG%SAXS = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAXS = CONFIG%SAXS-1 + CONFIG%NA * CONFIG%ADS * (CONFIG%ANS+1)
    CONFIG%RWORK_SIZE = CONFIG%EAXS
    ! apositional gradients at states
    CONFIG%SAXG = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAXG = CONFIG%SAXG-1 + CONFIG%NA * CONFIG%ADS * (CONFIG%ANS+1)
    CONFIG%RWORK_SIZE = CONFIG%EAXG
    ! AY
    CONFIG%SAY = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAY = CONFIG%SAY-1 + CONFIG%NA * CONFIG%ADO
    CONFIG%RWORK_SIZE = CONFIG%EAY
    ! AY gradient
    CONFIG%SAYG = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAYG = CONFIG%SAYG-1 + CONFIG%NA * CONFIG%ADO
    CONFIG%RWORK_SIZE = CONFIG%EAYG
    ! model states
    CONFIG%SMXS = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMXS = CONFIG%SMXS-1 + CONFIG%NM * CONFIG%MDS * (CONFIG%MNS+1)
    CONFIG%RWORK_SIZE = CONFIG%EMXS
    ! model gradients at states
    CONFIG%SMXG = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMXG = CONFIG%SMXG-1 + CONFIG%NM * CONFIG%MDS * (CONFIG%MNS+1)
    CONFIG%RWORK_SIZE = CONFIG%EMXG
    ! Y gradient
    CONFIG%SYG = 1 + CONFIG%RWORK_SIZE
    CONFIG%EYG = CONFIG%SYG-1 + DY * CONFIG%NM
    CONFIG%RWORK_SIZE = CONFIG%EYG
    ! AX rescale
    CONFIG%SAXR = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAXR = CONFIG%SAXR-1 + CONFIG%ADN * CONFIG%ADN
    CONFIG%RWORK_SIZE = CONFIG%EAXR
    ! AXI shift
    CONFIG%SAXIS = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAXIS = CONFIG%SAXIS-1 + CONFIG%ADE
    CONFIG%RWORK_SIZE = CONFIG%EAXIS
    ! AXI rescale
    CONFIG%SAXIR = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAXIR = CONFIG%SAXIR-1 + CONFIG%ADE * CONFIG%ADE
    CONFIG%RWORK_SIZE = CONFIG%EAXIR
    ! AY rescale
    CONFIG%SAYR = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAYR = CONFIG%SAYR-1 + CONFIG%ADO
    CONFIG%RWORK_SIZE = CONFIG%EAYR
    ! X rescale
    CONFIG%SMXR = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMXR = CONFIG%SMXR-1 + CONFIG%MDN * CONFIG%MDN
    CONFIG%RWORK_SIZE = CONFIG%EMXR
    ! XI shift
    CONFIG%SMXIS = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMXIS = CONFIG%SMXIS-1 + CONFIG%MDE
    CONFIG%RWORK_SIZE = CONFIG%EMXIS
    ! XI rescale
    CONFIG%SMXIR = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMXIR = CONFIG%SMXIR-1 + CONFIG%MDE * CONFIG%MDE
    CONFIG%RWORK_SIZE = CONFIG%EMXIR
    ! Y rescale
    CONFIG%SYR = 1 + CONFIG%RWORK_SIZE
    CONFIG%EYR = CONFIG%SYR-1 + DY * DY
    CONFIG%RWORK_SIZE = CONFIG%EYR
    ! A lengths (lengths of state values after orthogonalization)
    CONFIG%SAL = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAL = CONFIG%SAL-1 + CONFIG%ADS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EAL
    ! M lengths (lengths of state values after orthogonalization)
    CONFIG%SML = 1 + CONFIG%RWORK_SIZE
    CONFIG%EML = CONFIG%SML-1 + CONFIG%MDS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EML
    ! A state temp holder (for orthogonality computation)
    CONFIG%SAST = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAST = CONFIG%SAST-1 + CONFIG%NA * CONFIG%ADS
    CONFIG%RWORK_SIZE = CONFIG%EAST
    ! M state temp holder (for orthogonality computation)
    CONFIG%SMST = 1 + CONFIG%RWORK_SIZE
    CONFIG%EMST = CONFIG%SMST-1 + CONFIG%NM * CONFIG%MDS
    CONFIG%RWORK_SIZE = CONFIG%EMST
    ! ------------------------------------------------------------
    ! Set up the integer valued work array.
    CONFIG%IWORK_SIZE = 0
    ! update indices of model 
    CONFIG%SUI = 1 + CONFIG%IWORK_SIZE
    CONFIG%EUI = CONFIG%SUI-1 + CONFIG%NUM_VARS
    CONFIG%IWORK_SIZE = CONFIG%EUI
    ! apositional batch starts
    CONFIG%SBAS = 1 + CONFIG%IWORK_SIZE
    CONFIG%EBAS = CONFIG%SBAS-1 + CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EBAS
    ! apositional batch ends
    CONFIG%SBAE = 1 + CONFIG%IWORK_SIZE
    CONFIG%EBAE = CONFIG%SBAE-1 + CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EBAE
    ! model batch starts
    CONFIG%SBMS = 1 + CONFIG%IWORK_SIZE
    CONFIG%EBMS = CONFIG%SBMS-1 + CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EBMS
    ! model batch ends
    CONFIG%SBME = 1 + CONFIG%IWORK_SIZE
    CONFIG%EBME = CONFIG%SBME-1 + CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EBME
    ! A order (for orthogonalization)
    CONFIG%SAO = 1 + CONFIG%IWORK_SIZE
    CONFIG%EAO = CONFIG%SAO-1 + CONFIG%ADS * CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EAO
    ! M order (for orthogonalization)
    CONFIG%SMO = 1 + CONFIG%IWORK_SIZE
    CONFIG%EMO = CONFIG%SMO-1 + CONFIG%MDS * CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EMO
  END SUBROUTINE NEW_FIT_CONFIG

  ! Initialize the weights for a model, optionally provide a random seed.
  SUBROUTINE INIT_MODEL(CONFIG, MODEL, SEED)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    INTEGER, INTENT(IN), OPTIONAL :: SEED
    !  Storage for seeding the random number generator (for repeatability).
    INTEGER, DIMENSION(:), ALLOCATABLE :: SEED_ARRAY
    ! Local iterator.
    INTEGER :: I
    ! Set a random seed, if one was provided (otherwise leave default).
    IF (PRESENT(SEED)) THEN
       CALL RANDOM_SEED(SIZE=I)
       ALLOCATE(SEED_ARRAY(I))
       SEED_ARRAY(:) = SEED
       CALL RANDOM_SEED(PUT=SEED_ARRAY(:))
    END IF
    ! Unpack the model vector into its parts.
    CALL UNPACKED_INIT_MODEL(&
         CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, &
         CONFIG%MDSO, CONFIG%MDO, CONFIG%MDE, CONFIG%MNE, &
         MODEL(CONFIG%MSIV:CONFIG%MEIV), &
         MODEL(CONFIG%MSIS:CONFIG%MEIS), &
         MODEL(CONFIG%MSSV:CONFIG%MESV), &
         MODEL(CONFIG%MSSS:CONFIG%MESS), &
         MODEL(CONFIG%MSOV:CONFIG%MEOV), &
         MODEL(CONFIG%MSEV:CONFIG%MEEV))
    ! Initialize the apositional model.
    CALL UNPACKED_INIT_MODEL(&
         CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, &
         CONFIG%ADSO, CONFIG%ADO, CONFIG%ADE, CONFIG%ANE, &
         MODEL(CONFIG%ASIV:CONFIG%AEIV), &
         MODEL(CONFIG%ASIS:CONFIG%AEIS), &
         MODEL(CONFIG%ASSV:CONFIG%AESV), &
         MODEL(CONFIG%ASSS:CONFIG%AESS), &
         MODEL(CONFIG%ASOV:CONFIG%AEOV), &
         MODEL(CONFIG%ASEV:CONFIG%AEEV))

  CONTAINS
    ! Initialize the model after unpacking it into its constituent parts.
    SUBROUTINE UNPACKED_INIT_MODEL(MDI, MDS, MNS, MDSO, MDO, MDE, MNE, &
         INPUT_VECS, INPUT_SHIFT, STATE_VECS, STATE_SHIFT, &
         OUTPUT_VECS, EMBEDDINGS)
      INTEGER, INTENT(IN) :: MDI, MDS, MNS, MDSO, MDO, MDE, MNE
      REAL(KIND=RT), DIMENSION(MDI, MDS) :: INPUT_VECS
      REAL(KIND=RT), DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), DIMENSION(MDS, MDS, MAX(0,MNS-1)) :: STATE_VECS
      REAL(KIND=RT), DIMENSION(MDS, MAX(0,MNS-1)) :: STATE_SHIFT
      REAL(KIND=RT), DIMENSION(MDSO, MDO) :: OUTPUT_VECS
      REAL(KIND=RT), DIMENSION(MDE, MNE) :: EMBEDDINGS
      ! Local holder for "origin" at each layer.
      REAL(KIND=RT), DIMENSION(MDS) :: ORIGIN
      INTEGER,       DIMENSION(MDS) :: ORDER
      INTEGER :: I, J
      ! Generate well spaced random unit-length vectors (no scaling biases)
      ! for all initial variables in the input, internal, output, and embedings.
      CALL RANDOM_UNIT_VECTORS(INPUT_VECS(:,:))
      DO I = 1, MNS-1
         CALL RANDOM_UNIT_VECTORS(STATE_VECS(:,:,I))
      END DO
      CALL RANDOM_UNIT_VECTORS(OUTPUT_VECS(:,:))
      CALL RANDOM_UNIT_VECTORS(EMBEDDINGS(:,:))
      ! Make the output vectors have very small magnitude initially.
      OUTPUT_VECS(:,:) = OUTPUT_VECS(:,:) * CONFIG%INITIAL_OUTPUT_SCALE
      ! Generate random shifts for inputs and internal layers, zero
      !  shift for the output layer (first two will be rescaled).
      DO I = 1, MDS
         INPUT_SHIFT(I) = 2.0_RT * CONFIG%INITIAL_SHIFT_RANGE * & ! 2 * shift *
              (REAL(I-1,RT) / MAX(1.0_RT, REAL(MDS-1, RT))) &     ! range [0, 1]
              - CONFIG%INITIAL_SHIFT_RANGE                        ! - shift
      END DO
      ! Set the state shifts based on translation of the origin, always try
      !  to apply translations to bring the origin back closer to center
      !  (to prevent terrible conditioning of models with many layers).
      ORIGIN(:) = INPUT_SHIFT(:)
      DO J = 1, MNS-1
         ORIGIN(:) = MATMUL(ORIGIN(:), STATE_VECS(:,:,J))
         ! Argsort the values of origin, adding the most to the smallest values.
         CALL ARGSORT(ORIGIN(:), ORDER(:))
         DO I = 1, MDS
            STATE_SHIFT(ORDER(MDS-I+1),J) = INPUT_SHIFT(I) ! range [-shift, shift]
         END DO
         ORIGIN(:) = ORIGIN(:) + STATE_SHIFT(:,J)
      END DO
    END SUBROUTINE UNPACKED_INIT_MODEL
  END SUBROUTINE INIT_MODEL


  ! Returnn nonzero INFO if any shapes or values do not match expectations.
  SUBROUTINE CHECK_SHAPE(CONFIG, MODEL, AX, AXI, SIZES, X, XI, Y, INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    INTEGER,       INTENT(OUT) :: INFO
    INFO = 0
    ! Compute whether the shape matches the CONFIG.
    IF (SIZE(MODEL) .NE. CONFIG%TOTAL_SIZE) THEN
       INFO = 1 ! Model size does not match model configuration.
    ELSE IF (SIZE(X,2) .NE. SIZE(Y,2)) THEN
       INFO = 2 ! Input arrays do not match in size.
    ELSE IF (SIZE(X,1) .NE. CONFIG%MDI) THEN
       INFO = 3 ! X input dimension is bad.
    ELSE IF ((CONFIG%MDO .GT. 0) .AND. (SIZE(Y,1) .NE. CONFIG%MDO)) THEN
       INFO = 4 ! Output dimension is bad.
    ELSE IF ((CONFIG%MDO .EQ. 0) .AND. (SIZE(Y,1) .NE. CONFIG%ADO)) THEN
       INFO = 5 ! Output dimension is bad.
    ELSE IF ((CONFIG%MNE .GT. 0) .AND. (SIZE(XI,2) .NE. SIZE(X,2))) THEN
       INFO = 6 ! Input integer XI size does not match X.
    ELSE IF ((MINVAL(XI) .LT. 0) .OR. (MAXVAL(XI) .GT. CONFIG%MNE)) THEN
       INFO = 7 ! Input integer X index out of range.
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (SIZE(SIZES) .NE. SIZE(Y,2))) THEN
       INFO = 8 ! SIZES has wrong size.
    ELSE IF (SIZE(AX,2) .NE. SUM(SIZES)) THEN
       INFO = 9 ! AX and SUM(SIZES) do not match.
    ELSE IF (SIZE(AX,1) .NE. CONFIG%ADI) THEN
       INFO = 10 ! AX input dimension is bad.
    ELSE IF (SIZE(AXI,2) .NE. SIZE(AX,2)) THEN
       INFO = 11 ! Input integer AXI size does not match AX.
    ELSE IF ((MINVAL(AXI) .LT. 0) .OR. (MAXVAL(AXI) .GT. CONFIG%ANE)) THEN
       INFO = 12 ! Input integer AX index out of range.
    END IF
  END SUBROUTINE CHECK_SHAPE

 
  ! Given a number of batches, compute the batch start and ends for
  !  the apositional and positional inputs. Store in (2,_) arrays.
  SUBROUTINE COMPUTE_BATCHES(NUM_BATCHES, NA, NM, SIZES, &
       BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS, INFO)
    INTEGER, INTENT(IN) :: NUM_BATCHES
    INTEGER(KIND=INT64), INTENT(IN) :: NA, NM
    INTEGER, INTENT(IN),  DIMENSION(:) :: SIZES
    INTEGER, INTENT(OUT), DIMENSION(:) :: BATCHA_STARTS, BATCHA_ENDS
    INTEGER, INTENT(OUT), DIMENSION(:) :: BATCHM_STARTS, BATCHM_ENDS
    INTEGER, INTENT(INOUT) :: INFO
    ! Local variables.
    INTEGER :: BATCH, BE, BN, BS, I
    ! Check for errors.
    IF (NUM_BATCHES .GT. NM) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Requested number of batches is too large.', NUM_BATCHES, NM, NA
       INFO = -1
       RETURN
    ELSE IF (NUM_BATCHES .NE. SIZE(BATCHA_STARTS)) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Number of batches does not match BATCHA.', NUM_BATCHES, SIZE(BATCHA_STARTS)
       INFO = -2
       RETURN
    ELSE IF (NUM_BATCHES .NE. SIZE(BATCHM_STARTS)) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Number of batches does not match BATCHM.', NUM_BATCHES, SIZE(BATCHM_STARTS)
       INFO = -3
       RETURN
    ELSE IF (NUM_BATCHES .LT. 1) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Number of batches is negative.', NUM_BATCHES
       INFO = -4
       RETURN
    END IF
    ! Construct batches for data sets with apositional inputs.
    IF (NA .GT. 0) THEN
       IF (NUM_BATCHES .EQ. 1) THEN
          BATCHA_STARTS(1) = 1
          BATCHA_ENDS(1) = NA
          BATCHM_STARTS(1) = 1
          BATCHM_ENDS(1) = NM
       ELSE
          BN = (NA + NUM_BATCHES - 1) / NUM_BATCHES ! = CEIL(NA / NUM_BATCHES)
          ! Compute how many X points are associated with each Y.
          BS = 1
          BE = SIZES(1)
          BATCH = 1
          BATCHM_STARTS(BATCH) = 1
          DO I = 2, NM
             ! If a fair share of the points have been aggregated, OR
             !   there are only as many sets left as there are batches.
             IF ((BE-BS .GT. BN) .OR. (1+NM-I .LE. (NUM_BATCHES-BATCH))) THEN
                BATCHM_ENDS(BATCH) = I-1
                BATCHA_STARTS(BATCH) = BS
                BATCHA_ENDS(BATCH) = BE
                BATCH = BATCH+1
                BATCHM_STARTS(BATCH) = I
                BS = BE+1
                BE = BS - 1
             END IF
             BE = BE + SIZES(I)
          END DO
          BATCHM_ENDS(BATCH) = NM
          BATCHA_STARTS(BATCH) = BS
          BATCHA_ENDS(BATCH) = BE
       END IF
    ! Construct batches for data sets that only have positional inputs.
    ELSE
       BN = (NM + NUM_BATCHES - 1) / NUM_BATCHES ! = CEIL(NM / NUM_BATCHES)
       DO BATCH = 1, NUM_BATCHES
          BATCHM_STARTS(BATCH) = BN*(BATCH-1) + 1
          BATCHM_ENDS(BATCH) = MIN(NM, BN*BATCH)
       END DO
       BATCHA_STARTS(:) = 0
       BATCHA_ENDS(:) = -1
    END IF
  END SUBROUTINE COMPUTE_BATCHES


  ! Given a model and mixed real and integer inputs, embed the integer
  !  inputs into their appropriate real-value-only formats.
  SUBROUTINE EMBED(CONFIG, MODEL, AXI, XI, AX, X)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:) :: MODEL
    INTEGER,       INTENT(IN),  DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN),  DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: AX ! ADI, SIZE(AX,2)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: X  ! MDI, SIZE(X,2)
    ! If there is AXInteger input, unpack it into X.
    IF (CONFIG%ADE .GT. 0) THEN
       CALL UNPACK_EMBEDDINGS(CONFIG%ADE, CONFIG%ANE, &
            MODEL(CONFIG%ASEV:CONFIG%AEEV), &
            AXI(:,:), AX(CONFIG%ADN+1:,:))
    END IF
    ! If there is XInteger input, unpack it into end of X.
    IF (CONFIG%MDE .GT. 0) THEN
       CALL UNPACK_EMBEDDINGS(CONFIG%MDE, CONFIG%MNE, &
            MODEL(CONFIG%MSEV:CONFIG%MEEV), &
            XI(:,:), X(CONFIG%MDN+1:CONFIG%MDN+CONFIG%MDE,:))
    END IF
  CONTAINS
    ! Given integer inputs and embedding vectors, put embeddings in
    !  place of integer inputs inside of a real matrix.
    SUBROUTINE UNPACK_EMBEDDINGS(MDE, MNE, EMBEDDINGS, INT_INPUTS, EMBEDDED)
      INTEGER, INTENT(IN) :: MDE, MNE
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDE, MNE) :: EMBEDDINGS
      INTEGER, INTENT(IN), DIMENSION(:,:) :: INT_INPUTS
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: EMBEDDED
      INTEGER :: N, D, E
      REAL(KIND=RT) :: RD
      RD = REAL(SIZE(INT_INPUTS,1),RT)
      ! Add together appropriate embedding vectors based on integer inputs.
      EMBEDDED(:,:) = 0.0_RT
      DO N = 1, SIZE(INT_INPUTS,2)
         DO D = 1, SIZE(INT_INPUTS,1)
            E = INT_INPUTS(D,N)
            IF (E .GT. 0) THEN
               EMBEDDED(:,N) = EMBEDDED(:,N) + EMBEDDINGS(:,E)
            END IF
         END DO
         EMBEDDED(:,N) = EMBEDDED(:,N) / RD
      END DO
    END SUBROUTINE UNPACK_EMBEDDINGS
  END SUBROUTINE EMBED


  ! Evaluate the piecewise linear regression model, assume already-embedded inputs.
  SUBROUTINE EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, Y, A_STATES, M_STATES, INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:) :: AY
    INTEGER,       INTENT(IN),    DIMENSION(:)   :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:,:) :: A_STATES ! SIZE(AX,2), ADS, (ANS|2)
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:,:) :: M_STATES ! SIZE(X, 2), MDS, (MNS|2)
    INTEGER, INTENT(INOUT) :: INFO
    ! Internal values.
    INTEGER :: I, BATCH, NB, BN, BS, BE, BT, GS, GE, NT, E
    INTEGER, DIMENSION(:), ALLOCATABLE :: BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS
    ! Set up batching for parallelization.
    NB = MIN(SIZE(Y,2), CONFIG%NUM_THREADS)
    NT = MIN(CONFIG%NUM_THREADS, NB)
    ! Compute the batch start and end indices.
    ALLOCATE(BATCHA_STARTS(NB), BATCHA_ENDS(NB), BATCHM_STARTS(NB), BATCHM_ENDS(NB))
    CALL COMPUTE_BATCHES(NB, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS, INFO)
    IF (INFO .NE. 0) RETURN
    !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT, GS, GE) IF(NB > 1)
    batch_evaluation : DO BATCH = 1, NB
       ! Apositional model forward pass.
       IF (CONFIG%ADI .GT. 0) THEN
          BS = BATCHA_STARTS(BATCH)
          BE = BATCHA_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .EQ. 0) CYCLE batch_evaluation
          ! Apply shift terms to apositional inputs.
          IF ((CONFIG%APPLY_SHIFT) .AND. (CONFIG%ADN .GT. 0)) THEN
             DO I = BS, BE
                AX(:CONFIG%ADN,I) = AX(:CONFIG%ADN,I) + MODEL(CONFIG%AISS:CONFIG%AISE)
             END DO
          END IF
          ! Evaluate the apositional model.
          CALL UNPACKED_EVALUATE(BT, &
               CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADSO, CONFIG%ADO, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASIS:CONFIG%AEIS), &
               MODEL(CONFIG%ASSV:CONFIG%AESV), &
               MODEL(CONFIG%ASSS:CONFIG%AESS), &
               MODEL(CONFIG%ASOV:CONFIG%AEOV), &
               AX(:,BS:BE), AY(BS:BE,:), A_STATES(BS:BE,:,:), YTRANS=.TRUE.)
          ! Unapply shift terms to apositional inputs.
          IF ((CONFIG%APPLY_SHIFT) .AND. (CONFIG%ADN .GT. 0)) THEN
             DO I = BS, BE
                AX(:CONFIG%ADN,I) = AX(:CONFIG%ADN,I) - MODEL(CONFIG%AISS:CONFIG%AISE)
             END DO
          END IF
          GS = BS ! First group start is the batch start.
          ! Take the mean of all outputs from the apositional model, store
          !   as input to the model that proceeds this aggregation.
          IF (CONFIG%MDO .GT. 0) THEN
             ! Apply zero-mean shift terms to apositional model outputs.
             DO I = 1, CONFIG%ADO
                AY(BS:BE,I) = AY(BS:BE,I) + MODEL(CONFIG%AOSS + I-1)
             END DO
             E = CONFIG%MDN+CONFIG%MDE+1 ! <- start of apositional output
             DO I = BATCHM_STARTS(BATCH), BATCHM_ENDS(BATCH)
                GE = GS + SIZES(I) - 1
                X(E:,I) = SUM(AY(GS:GE,:), 1) / REAL(SIZES(I),RT) 
                GS = GE + 1
             END DO
          ! If there is no model after this, place results directly in Y.
          ELSE
             DO I = BATCHM_STARTS(BATCH), BATCHM_ENDS(BATCH)
                GE = GS + SIZES(I) - 1
                Y(:,I) = SUM(AY(GS:GE,:), 1) / REAL(SIZES(I),RT) 
                GS = GE + 1
             END DO
          END IF
       END IF
       ! Update "BS", "BE", and "BT" to coincide with the model.
       BS = BATCHM_STARTS(BATCH)
       BE = BATCHM_ENDS(BATCH)
       BT = BE-BS+1
       ! Positional model forward pass.
       IF (CONFIG%MDO .GT. 0) THEN
          IF (BT .EQ. 0) CYCLE batch_evaluation
          ! Apply shift terms to numeric model inputs.
          IF ((CONFIG%APPLY_SHIFT) .AND. (CONFIG%MDN .GT. 0)) THEN
             DO I = BS, BE
                X(:CONFIG%MDN,I) = X(:CONFIG%MDN,I) + MODEL(CONFIG%MISS:CONFIG%MISE)
             END DO
          END IF
          ! Run the positional model.
          CALL UNPACKED_EVALUATE(BT, &
               CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDSO, CONFIG%MDO, &
               MODEL(CONFIG%MSIV:CONFIG%MEIV), &
               MODEL(CONFIG%MSIS:CONFIG%MEIS), &
               MODEL(CONFIG%MSSV:CONFIG%MESV), &
               MODEL(CONFIG%MSSS:CONFIG%MESS), &
               MODEL(CONFIG%MSOV:CONFIG%MEOV), &
               X(:,BS:BE), Y(:,BS:BE), M_STATES(BS:BE,:,:), YTRANS=.FALSE.)
          ! Unapply the X shifts.
          IF ((CONFIG%APPLY_SHIFT) .AND. (CONFIG%MDN .GT. 0)) THEN
             DO I = BS, BE
                X(:CONFIG%MDN,I) = X(:CONFIG%MDN,I) - MODEL(CONFIG%MISS:CONFIG%MISE)
             END DO
          END IF
       END IF
       ! Apply shift terms to final outputs.
       IF (CONFIG%APPLY_SHIFT) THEN
          DO I = BS, BE
             Y(:,I) = Y(:,I) + MODEL(CONFIG%MOSS:CONFIG%MOSE)
          END DO
       END IF
    END DO batch_evaluation
    !$OMP END PARALLEL DO
    DEALLOCATE(BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS)

  CONTAINS

    SUBROUTINE UNPACKED_EVALUATE(N, MDI, MDS, MNS, MDSO, MDO, INPUT_VECS, &
         INPUT_SHIFT, STATE_VECS, STATE_SHIFT, OUTPUT_VECS, X, Y, &
         STATES, YTRANS)
      INTEGER, INTENT(IN) :: N, MDI, MDS, MNS, MDSO, MDO
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDI, MDS) :: INPUT_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MDS, MAX(0,MNS-1)) :: STATE_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MAX(0,MNS-1)) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDSO, MDO) :: OUTPUT_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(:,:) :: X
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: Y
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:,:) :: STATES
      LOGICAL, INTENT(IN) :: YTRANS
      ! Local variables to evaluating a single batch.
      INTEGER :: D, L, S1, S2, S3
      LOGICAL :: REUSE_STATES
      REUSE_STATES = (SIZE(STATES,3) .LT. MNS)
      IF (MNS .GT. 0) THEN
         ! Compute the input transformation.
         CALL GEMM('T', 'N', N, MDS, MDI, 1.0_RT, &
              X(:,:), SIZE(X,1), &
              INPUT_VECS(:,:), SIZE(INPUT_VECS,1), &
              0.0_RT, STATES(:,:,1), N)
         ! Apply the rectification.
         DO D = 1, MDS
            STATES(:,D,1) = MAX(STATES(:,D,1) + INPUT_SHIFT(D), CONFIG%DISCONTINUITY)
         END DO
         ! Compute the next set of internal values with a rectified activation.
         DO L = 1, MNS-1
            ! Determine the storage locations of values based on number of states.
            IF (REUSE_STATES) THEN ; S1 = 1 ; S2 = 2   ; S3 = 1
            ELSE                   ; S1 = L ; S2 = L+1 ; S3 = L+1
            END IF
            ! Compute all vectors.
            CALL GEMM('N', 'N', N, MDS, MDS, 1.0_RT, &
                 STATES(:,:,S1), N, &
                 STATE_VECS(:,:,L), SIZE(STATE_VECS,1), &
                 0.0_RT, STATES(:,:,S2), N)
            ! Compute all piecewise linear functions and apply the rectification.
            DO D = 1, MDS
               STATES(:,D,S3) = MAX(STATES(:,D,S2) + STATE_SHIFT(D,L), CONFIG%DISCONTINUITY)
            END DO
         END DO
         ! Set the location of the "previous state output".
         IF (REUSE_STATES) THEN ; S3 = 1
         ELSE                   ; S3 = MNS
         END IF
         ! Return the final output (default to assuming Y is contiguous
         !   by component unless PRESENT(YTRANS) and YTRANS = .TRUE.
         !   then assume it is contiguous by individual sample).
         IF (YTRANS) THEN
            CALL GEMM('N', 'N', N, MDO, MDS, 1.0_RT, &
                 STATES(:,:,S3), N, &
                 OUTPUT_VECS(:,:), SIZE(OUTPUT_VECS,1), &
                 0.0_RT, Y(:,:), SIZE(Y,1))
         ELSE
            CALL GEMM('T', 'T', MDO, N, MDS, 1.0_RT, &
                 OUTPUT_VECS(:,:), SIZE(OUTPUT_VECS,1), &
                 STATES(:,:,S3), N, &
                 0.0_RT, Y(:,:), SIZE(Y,1))
         END IF
      ! Handle the linear model case, where there are only output vectors.
      ELSE
         ! Return the final output (default to assuming Y is contiguous
         !   by component unless PRESENT(YTRANS) and YTRANS = .TRUE.
         !   then assume it is contiguous by individual sample).
         IF (YTRANS) THEN
            CALL GEMM('T', 'N', N, MDO, MDI, 1.0_RT, &
                 X(:,:), SIZE(X,1), &
                 OUTPUT_VECS(:,:), SIZE(OUTPUT_VECS,1), &
                 0.0_RT, Y(:,:), SIZE(Y,1))
         ELSE
            CALL GEMM('T', 'N', MDO, N, MDI, 1.0_RT, &
                 OUTPUT_VECS(:,:), SIZE(OUTPUT_VECS,1), &
                 X(:,:), SIZE(X,1), &
                 0.0_RT, Y(:,:), SIZE(Y,1))
         END IF
      END IF
    END SUBROUTINE UNPACKED_EVALUATE
  END SUBROUTINE EVALUATE


  ! Given the values at all internal states in the model and an output
  !  gradient, propogate the output gradient through the model and
  !  return the gradient of all basis functions.
  SUBROUTINE BASIS_GRADIENT(CONFIG, MODEL, Y, X, AX, SIZES, &
       M_STATES, A_STATES, AY, GRAD)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: M_STATES ! SIZE(X, 2), MDS, MNS+1
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_STATES ! SIZE(AX,2), ADS, ANS+1
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY ! SIZE(AX,2), ADO
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:) :: GRAD
    ! Set the dimension of the X gradient that should be calculated.
    INTEGER :: I, J, GS, GE, XDG
    ! Propogate the gradient through the positional model.
    IF (CONFIG%MDO .GT. 0) THEN
       XDG = CONFIG%MDE
       IF (CONFIG%ADI .GT. 0) THEN
          XDG = XDG + CONFIG%ADO
       END IF
       ! Do the backward gradient calculation assuming "Y" contains output gradient.
       CALL UNPACKED_BASIS_GRADIENT( Y(:,:), M_STATES(:,:,:), X(:,:), &
            CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDSO, CONFIG%MDO, XDG, &
            MODEL(CONFIG%MSIV:CONFIG%MEIV), &
            MODEL(CONFIG%MSIS:CONFIG%MEIS), &
            MODEL(CONFIG%MSSV:CONFIG%MESV), &
            MODEL(CONFIG%MSSS:CONFIG%MESS), &
            MODEL(CONFIG%MSOV:CONFIG%MEOV), &
            GRAD(CONFIG%MSIV:CONFIG%MEIV), &
            GRAD(CONFIG%MSIS:CONFIG%MEIS), &
            GRAD(CONFIG%MSSV:CONFIG%MESV), &
            GRAD(CONFIG%MSSS:CONFIG%MESS), &
            GRAD(CONFIG%MSOV:CONFIG%MEOV), &
            YTRANS=.FALSE.) ! Y is in COLUMN vector format.
    END IF
    ! Propogate the gradient form X into the aggregate outputs.
    IF (CONFIG%ADI .GT. 0) THEN
       ! Propogate gradient from the input to the positional model.
       IF (CONFIG%MDO .GT. 0) THEN
          XDG = SIZE(X,1) - CONFIG%ADO + 1
          GS = 1
          DO I = 1, SIZE(SIZES)
             GE = GS + SIZES(I) - 1
             DO J = GS, GE
                AY(J,:) = X(XDG:,I) / REAL(SIZES(I),RT)
             END DO
             GS = GE + 1
          END DO
       ! Propogate gradient direction from the aggregate output.
       ELSE
          GS = 1
          DO I = 1, SIZE(SIZES)
             GE = GS + SIZES(I) - 1
             DO J = GS, GE
                AY(J,:) = Y(:,I) / REAL(SIZES(I),RT)
             END DO
             GS = GE + 1
          END DO
       END IF
       ! Do the backward gradient calculation assuming "AY" contains output gradient.
       CALL UNPACKED_BASIS_GRADIENT( AY(:,:), A_STATES(:,:,:), AX(:,:), &
            CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADSO, CONFIG%ADO, CONFIG%ADE, &
            MODEL(CONFIG%ASIV:CONFIG%AEIV), &
            MODEL(CONFIG%ASIS:CONFIG%AEIS), &
            MODEL(CONFIG%ASSV:CONFIG%AESV), &
            MODEL(CONFIG%ASSS:CONFIG%AESS), &
            MODEL(CONFIG%ASOV:CONFIG%AEOV), &
            GRAD(CONFIG%ASIV:CONFIG%AEIV), &
            GRAD(CONFIG%ASIS:CONFIG%AEIS), &
            GRAD(CONFIG%ASSV:CONFIG%AESV), &
            GRAD(CONFIG%ASSS:CONFIG%AESS), &
            GRAD(CONFIG%ASOV:CONFIG%AEOV), &
            YTRANS=.TRUE.) ! AY is in ROW vector format.
    END IF

  CONTAINS
    ! Compute the model gradient.
    SUBROUTINE UNPACKED_BASIS_GRADIENT( Y, STATES, X, &
         MDI, MDS, MNS, MDSO, MDO, MDE, &
         INPUT_VECS, INPUT_SHIFT, &
         STATE_VECS, STATE_SHIFT, OUTPUT_VECS, &
         INPUT_VECS_GRADIENT, INPUT_SHIFT_GRADIENT, &
         STATE_VECS_GRADIENT, STATE_SHIFT_GRADIENT, &
         OUTPUT_VECS_GRADIENT, YTRANS )
      REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: STATES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
      INTEGER, INTENT(IN) :: MDI, MDS, MNS, MDSO, MDO, MDE
      ! Model variables.
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDI,MDS) :: INPUT_VECS
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS,MDS,MAX(0,MNS-1)) :: STATE_VECS
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS,MAX(0,MNS-1)) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDSO,MDO) :: OUTPUT_VECS
      ! Model variable gradients.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI,MDS) :: INPUT_VECS_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS) :: INPUT_SHIFT_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MDS,MAX(0,MNS-1)) :: STATE_VECS_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MAX(0,MNS-1)) :: STATE_SHIFT_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDSO,MDO) :: OUTPUT_VECS_GRADIENT
      LOGICAL, INTENT(IN) :: YTRANS
      ! D   - dimension index
      ! L   - layer index
      ! LP1 - layer index "plus 1" -> "P1"
      INTEGER :: D, L, LP1
      CHARACTER :: YT
      ! Number of points.
      REAL(KIND=RT) :: N, NORM
      ! Get the number of points.
      N = REAL(SIZE(X,2), RT)
      ! Set default for assuming Y is transposed (row vectors).
      IF (YTRANS) THEN ; YT = 'N'
      ELSE             ; YT = 'T'
      END IF
      ! Handle propogation of the gradient through internal states.
      IF (MNS .GT. 0) THEN
         ! Compute the gradient of variables with respect to the "output gradient"
         CALL GEMM('T', YT, MDS, MDO, SIZE(X,2), 1.0_RT / N, &
              STATES(:,:,MNS), SIZE(STATES,1), &
              Y(:,:), SIZE(Y,1), &
              1.0_RT, OUTPUT_VECS_GRADIENT(:,:), SIZE(OUTPUT_VECS_GRADIENT,1))
         ! Propogate the gradient back to the last internal vector space.
         CALL GEMM(YT, 'T', SIZE(X,2), MDS, MDO, 1.0_RT, &
              Y(:,:), SIZE(Y,1), &
              OUTPUT_VECS(:,:), SIZE(OUTPUT_VECS,1), &
              0.0_RT, STATES(:,:,MNS+1), SIZE(STATES,1))
         ! Cycle over all internal layers.
         STATE_REPRESENTATIONS : DO L = MNS-1, 1, -1
            LP1 = L+1
            DO D = 1, MDS
               ! Propogate the error gradient back to the preceding vectors.
               WHERE (STATES(:,D,LP1) .GT. CONFIG%DISCONTINUITY)
                  STATES(:,D,LP1) = STATES(:,D,MNS+1)
               END WHERE
            END DO
            ! Compute the shift gradient.
            STATE_SHIFT_GRADIENT(:,L) = SUM(STATES(:,:,LP1), 1) / N &
                 + STATE_SHIFT_GRADIENT(:,L)
            ! Compute the gradient with respect to each output and all inputs.
            CALL GEMM('T', 'N', MDS, MDS, SIZE(X,2), 1.0_RT / N, &
                 STATES(:,:,L), SIZE(STATES,1), &
                 STATES(:,:,LP1), SIZE(STATES,1), &
                 1.0_RT, STATE_VECS_GRADIENT(:,:,L), SIZE(STATE_VECS_GRADIENT,1))
            ! Propogate the gradient to the immediately preceding layer.
            CALL GEMM('N', 'T', SIZE(X,2), MDS, MDS, 1.0_RT, &
                 STATES(:,:,LP1), SIZE(STATES,1), &
                 STATE_VECS(:,:,L), SIZE(STATE_VECS,1), &
                 0.0_RT, STATES(:,:,MNS+1), SIZE(STATES,1))
         END DO STATE_REPRESENTATIONS
         ! Compute the gradients going into the first layer.
         DO D = 1, MDS
            ! Propogate the error gradient back to the preceding vectors.
            WHERE (STATES(:,D,1) .GT. CONFIG%DISCONTINUITY)
               STATES(:,D,1) = STATES(:,D,MNS+1)
            END WHERE
         END DO
         ! Compute the input shift variable gradients.
         INPUT_SHIFT_GRADIENT(:) = SUM(STATES(:,:,1), 1) / N &
              + INPUT_SHIFT_GRADIENT(:)
         ! Compute the gradient of all input variables.
         !   [the X are transposed already, shape = (MDI,N)]
         CALL GEMM('N', 'N', MDI, MDS, SIZE(X,2), 1.0_RT / N, &
              X(:,:), SIZE(X,1), &
              STATES(:,:,1), SIZE(STATES,1), &
              1.0_RT, INPUT_VECS_GRADIENT(:,:), SIZE(INPUT_VECS_GRADIENT,1))
         ! Compute the gradient at the input if there are embeddings.
         IF (MDE .GT. 0) THEN
            LP1 = SIZE(X,1)-MDE+1
            CALL GEMM('N', 'T', MDE, SIZE(X,2), MDS, 1.0_RT, &
                 INPUT_VECS(LP1:,:), MDE, &
                 STATES(:,:,1), SIZE(STATES,1), &
                 0.0_RT, X(LP1:,:), MDE)
         END IF
      ! Handle the purely linear case (no internal states).
      ELSE
         ! Compute the gradient of variables with respect to the "output gradient"
         CALL GEMM('N', YT, MDI, MDO, SIZE(X,2), 1.0_RT / N, &
              X(:,:), SIZE(X,1), &
              Y(:,:), SIZE(Y,1), &
              1.0_RT, OUTPUT_VECS_GRADIENT(:,:), SIZE(OUTPUT_VECS_GRADIENT,1))
         ! Propogate the gradient back to the input embeddings.
         IF (MDE .GT. 0) THEN
            LP1 = SIZE(X,1)-MDE+1
            IF (YTRANS) THEN ; YT = 'T'
            ELSE             ; YT = 'N'
            END IF
            CALL GEMM('N', YT, MDE, SIZE(X,2), MDO, 1.0_RT, &
                 OUTPUT_VECS(LP1:,:), SIZE(OUTPUT_VECS,1), &
                 Y(:,:), SIZE(Y,1), & ! MDO by N
                 0.0_RT, X(LP1:,:), MDE)
         END IF
      END IF

    END SUBROUTINE UNPACKED_BASIS_GRADIENT
  END SUBROUTINE BASIS_GRADIENT


  ! Compute the gradient with respect to embeddings given the input
  !  gradient by aggregating over the repeated occurrences of the embedding.
  SUBROUTINE EMBEDDING_GRADIENT(MDE, MNE, INT_INPUTS, GRAD, EMBEDDING_GRAD)
    INTEGER, INTENT(IN) :: MDE, MNE
    INTEGER, INTENT(IN), DIMENSION(:,:) :: INT_INPUTS
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: GRAD
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDE,MNE) :: EMBEDDING_GRAD
    ! Local variables.
    REAL(KIND=RT), DIMENSION(MDE,MNE) :: TEMP_GRAD
    REAL(KIND=RT), DIMENSION(MNE) :: COUNTS
    INTEGER :: N, D, E
    REAL(KIND=RT) :: RD
    ! Accumulate the gradients for all embedding vectors.
    COUNTS(:) = 0.0_RT
    TEMP_GRAD(:,:) = 0.0_RT
    RD = REAL(SIZE(INT_INPUTS,1),RT)
    DO N = 1, SIZE(INT_INPUTS,2)
       DO D = 1, SIZE(INT_INPUTS,1)
          E = INT_INPUTS(D,N)
          IF (E .GT. 0) THEN
             COUNTS(E) = COUNTS(E) + 1.0_RT
             TEMP_GRAD(:,E) = TEMP_GRAD(:,E) + GRAD(:,N) / RD
          END IF
       END DO
    END DO
    ! Average the embedding gradient by dividing by the sum of occurrences.
    DO E = 1, MNE
       IF (COUNTS(E) .GT. 0.0_RT) THEN
          EMBEDDING_GRAD(:,E) = EMBEDDING_GRAD(:,E) + TEMP_GRAD(:,E) / COUNTS(E)
       END IF
    END DO
  END SUBROUTINE EMBEDDING_GRADIENT


  ! Compute the gradient of the sum of squared error of this regression
  ! model with respect to its variables given input and output pairs.
  SUBROUTINE MODEL_GRADIENT(CONFIG, MODEL, AX, AXI, AY, SIZES, X, XI, Y, &
       SUM_SQUARED_GRADIENT, MODEL_GRAD, &
       AY_GRAD, Y_GRADIENT, A_STATES, A_GRADS, M_STATES, M_GRADS, &
       INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: AXI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY
    INTEGER,       INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    ! Sum (over all data) squared error (summed over dimensions).
    REAL(KIND=RT), INTENT(INOUT) :: SUM_SQUARED_GRADIENT
    ! Gradient of the model parameters.
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: MODEL_GRAD
    ! Work space.
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(SIZE(AX,2),CONFIG%ADO) :: AY_GRAD
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(SIZE(Y,1),SIZE(Y,2)) :: Y_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(SIZE(X,2),CONFIG%MDS,CONFIG%MNS+1) :: M_STATES, M_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(SIZE(AX,2),CONFIG%ADS,CONFIG%ANS+1) :: A_STATES, A_GRADS
    ! Output and optional inputs.
    INTEGER, INTENT(INOUT) :: INFO
    INTEGER :: L, D
    ! Embed all integer inputs into real vector inputs.
    CALL EMBED(CONFIG, MODEL, AXI, XI, AX, X)
    ! Evaluate the model, storing internal states (for gradient calculation).
    CALL EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, Y_GRADIENT, A_STATES, M_STATES, INFO)
    ! Compute the gradient of the model outputs, overwriting "Y_GRADIENT"
    Y_GRADIENT(:,:) = Y_GRADIENT(:,:) - Y(:,:) ! squared error gradient
    SUM_SQUARED_GRADIENT = SUM_SQUARED_GRADIENT + SUM(Y_GRADIENT(:,:)**2)
    ! Copy the state values into holders for the gradients.
    A_GRADS(:,:,:) = A_STATES(:,:,:)
    AY_GRAD(:,:) = AY(:,:)
    M_GRADS(:,:,:) = M_STATES(:,:,:)
    ! Compute the gradient with respect to the model basis functions.
    CALL BASIS_GRADIENT(CONFIG, MODEL, Y_GRADIENT, X, AX, &
         SIZES, M_GRADS, A_GRADS, AY_GRAD, MODEL_GRAD)
    ! Convert the computed input gradients into average gradients for each embedding.
    IF (CONFIG%MDE .GT. 0) THEN
       CALL EMBEDDING_GRADIENT(CONFIG%MDE, CONFIG%MNE, &
            XI, X(CONFIG%MDI-CONFIG%ADO-CONFIG%MDE+1:CONFIG%MDI-CONFIG%ADO,:), &
            MODEL_GRAD(CONFIG%MSEV:CONFIG%MEEV))
    END IF
    ! Convert the computed input gradients into average gradients for each embedding.
    IF (CONFIG%ADE .GT. 0) THEN
       CALL EMBEDDING_GRADIENT(CONFIG%ADE, CONFIG%ANE, &
            AXI, AX(CONFIG%ADI-CONFIG%ADE+1:CONFIG%ADI,:), &
            MODEL_GRAD(CONFIG%ASEV:CONFIG%AEEV))
    END IF
  END SUBROUTINE MODEL_GRADIENT

  
  ! Make inputs and outputs radially symmetric (to make initialization
  !  more well spaced and lower the curvature of the error gradient).
  ! 
  SUBROUTINE NORMALIZE_DATA(CONFIG, MODEL, AX, AXI, AY, SIZES, X, XI, Y, &
       AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_RESCALE, X_RESCALE, &
       XI_SHIFT, XI_RESCALE, Y_RESCALE, &
       A_STATES, A_EMB_VECS, M_EMB_VECS, A_OUT_VECS, INFO)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: AXI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY
    INTEGER,       INTENT(IN),    DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AXI_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AXI_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AY_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: XI_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: XI_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_STATES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADE,CONFIG%ANE) :: A_EMB_VECS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDE,CONFIG%MNE) :: M_EMB_VECS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADSO,CONFIG%ADO) :: A_OUT_VECS
    INTEGER, INTENT(INOUT) :: INFO
    INTEGER :: D, E
    ! Encode embeddings if the are provided.
    IF ((CONFIG%MDE + CONFIG%ADE .GT. 0) .AND. (&
         (.NOT. CONFIG%XI_NORMALIZED) .OR. (.NOT. CONFIG%AXI_NORMALIZED))) THEN
       CALL EMBED(CONFIG, MODEL, AXI, XI, AX, X)
    END IF
    ! 
    !$OMP PARALLEL NUM_THREADS(5)
    !$OMP SECTIONS PRIVATE(D)
    !$OMP SECTION
    IF ((.NOT. CONFIG%AX_NORMALIZED) .AND. (CONFIG%ADN .GT. 0)) THEN
       CALL RADIALIZE(AX(:CONFIG%ADN,:), &
            MODEL(CONFIG%AISS:CONFIG%AISE), AX_RESCALE(:,:))
       CONFIG%AX_NORMALIZED = .TRUE.
    ELSE IF (CONFIG%ADN .GT. 0) THEN
       MODEL(CONFIG%AISS:CONFIG%AISE) = 0.0_RT
       AX_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:CONFIG%ADN) AX_RESCALE(D,D) = 1.0_RT
    END IF
    !$OMP SECTION
    IF ((.NOT. CONFIG%AXI_NORMALIZED) .AND. (CONFIG%ADE .GT. 0)) THEN
       CALL RADIALIZE(AX(CONFIG%ADN+1:CONFIG%ADN+CONFIG%ADE,:), &
            AXI_SHIFT(:), AXI_RESCALE(:,:))
       ! Apply the shift to the source embeddings.
       DO D = 1, CONFIG%ADE
          A_EMB_VECS(D,:) = A_EMB_VECS(D,:) + AXI_SHIFT(D)
       END DO
       ! Apply the transformation to the source embeddings.
       A_EMB_VECS(:,:) = MATMUL(TRANSPOSE(AXI_RESCALE(:,:)), A_EMB_VECS(:,:))
       CONFIG%AXI_NORMALIZED = .TRUE.
    END IF
    !$OMP SECTION
    IF ((.NOT. CONFIG%X_NORMALIZED) .AND. (CONFIG%MDN .GT. 0)) THEN
       CALL RADIALIZE(X(:CONFIG%MDN,:), MODEL(CONFIG%MISS:CONFIG%MISE), X_RESCALE(:,:))
       CONFIG%X_NORMALIZED = .TRUE.
    ELSE IF (CONFIG%MDN .GT. 0) THEN
       MODEL(CONFIG%MISS:CONFIG%MISE) = 0.0_RT
       X_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:CONFIG%MDN) X_RESCALE(D,D) = 1.0_RT
    END IF
    !$OMP SECTION
    IF ((.NOT. CONFIG%XI_NORMALIZED) .AND. (CONFIG%MDE .GT. 0)) THEN
       CALL RADIALIZE(X(CONFIG%MDN+1:CONFIG%MDN+CONFIG%MDE,:), &
            XI_SHIFT(:), XI_RESCALE(:,:))
       ! Apply the shift to the source embeddings.
       DO D = 1, CONFIG%MDE
          M_EMB_VECS(D,:) = M_EMB_VECS(D,:) + XI_SHIFT(D)
       END DO
       ! Apply the transformation to the source embeddings.
       M_EMB_VECS(:,:) = MATMUL(TRANSPOSE(XI_RESCALE(:,:)), M_EMB_VECS(:,:))
       CONFIG%XI_NORMALIZED = .TRUE.
    END IF
    !$OMP SECTION
    IF (.NOT. CONFIG%Y_NORMALIZED) THEN
       CALL RADIALIZE(Y(:,:), MODEL(CONFIG%MOSS:CONFIG%MOSE), &
            Y_RESCALE(:,:), INVERT_RESULT=.TRUE., FLATTEN=.FALSE.)
       CONFIG%Y_NORMALIZED = .TRUE.
    ELSE
       MODEL(CONFIG%MOSS:CONFIG%MOSE) = 0.0_RT
       Y_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:SIZE(Y,1)) Y_RESCALE(D,D) = 1.0_RT
    END IF
    !$OMP END SECTIONS
    !$OMP END PARALLEL
    ! 
    ! Normalize AY outside the parallel region (AX must already be
    !  normalized, and EVALUATE contains parallelization).
    IF ((.NOT. CONFIG%AY_NORMALIZED) .AND. (CONFIG%ADO .GT. 0)) THEN
       MODEL(CONFIG%AOSS:CONFIG%AOSE) = 0.0_RT
       ! Only apply the normalization to AY if there is a model afterwards.
       IF (CONFIG%MDO .GT. 0) THEN
          E = CONFIG%MDN + CONFIG%MDE + 1 ! <- beginning of ADO storage in X
          ! Disable "model" evaluation for this forward pass.
          !   (Give "A_STATES" for the "M_STATES" argument, since it will not be unused.)
          D = CONFIG%MDO ; CONFIG%MDO = 0
          CALL EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, X(E:,:), A_STATES, A_STATES, INFO)
          CONFIG%MDO = D
          ! Compute AY shift as the mean of mean-outputs, apply it.
          MODEL(CONFIG%AOSS:CONFIG%AOSE) = -SUM(X(E:,:),2) / REAL(SIZE(X,2),RT)
          DO D = 0, CONFIG%ADO-1
             X(E+D,:) = X(E+D,:) + MODEL(CONFIG%AOSS + D)
          END DO
          ! Compute the AY scale as the standard deviation of mean-outputs.
          X(E:,1) = SUM(X(E:,:)**2,2) / REAL(SIZE(X,2),RT)
          WHERE (X(E:,1) .GT. 0.0_RT)
             X(E:,1) = SQRT(X(E:,1))
          ELSEWHERE
             X(E:,1) = 1.0_RT
          END WHERE
          ! Apply the factor to the output vectors (and the shift values).
          DO D = 1, CONFIG%ADO
             A_OUT_VECS(:,D) = A_OUT_VECS(:,D) / X(E+D-1,1)
             MODEL(CONFIG%AOSS+D-1) = MODEL(CONFIG%AOSS+D-1) / X(E+D-1,1)
          END DO
       END IF
       CONFIG%AY_NORMALIZED = .TRUE.
    END IF
  END SUBROUTINE NORMALIZE_DATA

  
  ! Performing conditioning related operations on this model 
  !  (ensure that mean squared error is effectively reduced).
  SUBROUTINE CONDITION_MODEL(CONFIG, MODEL, NUM_THREADS, FIT_STEP, AY, &
       A_STATES, M_STATES, A_GRADS, M_GRADS, &
       A_LENGTHS, M_LENGTHS, A_STATE_TEMP, M_STATE_TEMP, A_ORDER, M_ORDER, &
       NB, BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS, &
       TOTAL_EVAL_RANK, TOTAL_GRAD_RANK)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), DIMENSION(:) :: MODEL
    INTEGER, INTENT(IN) :: NUM_THREADS, FIT_STEP
    REAL(KIND=RT), DIMENSION(:,:) :: AY 
    REAL(KIND=RT), DIMENSION(:,:,:) :: A_STATES, M_STATES
    REAL(KIND=RT), DIMENSION(:,:,:) :: A_GRADS, M_GRADS
    REAL(KIND=RT), DIMENSION(:,:) :: A_LENGTHS, M_LENGTHS
    REAL(KIND=RT), DIMENSION(:,:) :: A_STATE_TEMP, M_STATE_TEMP
    INTEGER, DIMENSION(:,:) :: A_ORDER, M_ORDER
    INTEGER, INTENT(IN) :: NB
    INTEGER, INTENT(IN), DIMENSION(CONFIG%NUM_THREADS) :: &
         BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS
    INTEGER :: TOTAL_EVAL_RANK, TOTAL_GRAD_RANK
    ! Local variables.
    INTEGER :: I, VS, VE, J, R, NT, N, BS, BE, BN, BATCH, TER, TGR
    ! Maintain a constant max-norm across the magnitue of input and internal vectors.
    CALL UNIT_MAX_NORM(CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, &
         MODEL(CONFIG%MSIV:CONFIG%MEIV), &
         MODEL(CONFIG%MSSV:CONFIG%MESV))
    IF (CONFIG%ADI .GT. 0) THEN
       CALL UNIT_MAX_NORM(CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, &
            MODEL(CONFIG%ASIV:CONFIG%AEIV), &
            MODEL(CONFIG%ASSV:CONFIG%AESV))
    END IF
    ! Update the apositional model output shift to 
    !  produce componentwise mean-zero values (prevent divergence).
    IF ((CONFIG%MDO .GT. 0) .AND. (CONFIG%ADO .GT. 0)) THEN
       MODEL(CONFIG%AOSS:CONFIG%AOSE) = &
            CONFIG%STEP_AY_CHANGE * (-SUM(AY(:,:),1) / REAL(SIZE(AY,1),RT)) &
            + (1.0_RT - CONFIG%STEP_AY_CHANGE * MODEL(CONFIG%AOSS:CONFIG%AOSE))
    END IF
    ! -------------------------------------------------------------
    ! TODO:
    !  - Parallelize orthogonalization by same batches as training, 
    !    find those nodes that are agreeaby deleted, the total rank
    !    is the layer size less those agreeably deleted.
    ! 
    !  - Using the computed rank of values and gradients, delete the
    !    redundant basis functions and initialize with a combination
    !    of uncaptured previous layer values with gradients (first nonzero
    !    gradient components, then remaining nonzero input components).
    ! 
    ! - When measuring alignment of two vectors come up with way to
    !   quickly find the "most aligned" shift term (the shift that
    !   maximizes the dot product of the vectors assuming rectification).
    ! 
    IF ((CONFIG%LOGGING_STEP_FREQUENCY .GT. 0) .AND. &
         (MOD(FIT_STEP-1,CONFIG%LOGGING_STEP_FREQUENCY) .EQ. 0)) THEN
       TOTAL_EVAL_RANK = 0
       TOTAL_GRAD_RANK = 0
       ! Check the rank of all internal apositional states.
       J = CONFIG%ANS+1
       ! Batch computation formula.
       N = SIZE(A_STATE_TEMP,1)
       BN = (N + NUM_THREADS - 1) / NUM_THREADS ! = CEIL(NM / NUM_BATCHES)
       DO I = 1, CONFIG%ANS
          TER = 0; TGR = 0;
          !$OMP PARALLEL DO PRIVATE(R,BS,BE) NUM_THREADS(NUM_THREADS) &
          !$OMP& REDUCTION(MAX: TER, TGR)
          DO BATCH = 1, NUM_THREADS
             BS = BN*(BATCH-1) + 1
             BE = MIN(N, BN*BATCH)
             ! Compute model state rank.
             A_STATE_TEMP(BS:BE,:) = A_STATES(BS:BE,:,I)
             CALL ORTHOGONALIZE(A_STATE_TEMP(BS:BE,:), A_LENGTHS(:,BATCH), TER, A_ORDER(:,BATCH))
             ! Compute grad state rank.
             A_STATE_TEMP(BS:BE,:) = A_GRADS(BS:BE,:,I)
             CALL ORTHOGONALIZE(A_STATE_TEMP(BS:BE,:), A_LENGTHS(:,BATCH), TGR, A_ORDER(:,BATCH))
          END DO
          !$OMP END PARALLEL DO
          TOTAL_EVAL_RANK = TOTAL_EVAL_RANK + TER
          TOTAL_GRAD_RANK = TOTAL_GRAD_RANK + TGR
       END DO
       ! 
       ! Check the rank of all internal model states.
       N = SIZE(M_STATE_TEMP,1)
       BN = (N + NUM_THREADS - 1) / NUM_THREADS ! = CEIL(NM / NUM_BATCHES)
       DO I = 1, CONFIG%MNS
          TER = 0; TGR = 0;
          !$OMP PARALLEL DO PRIVATE(R,BS,BE) NUM_THREADS(NUM_THREADS) &
          !$OMP& REDUCTION(MAX: TER, TGR)
          DO BATCH = 1, NUM_THREADS
             BS = BN*(BATCH-1) + 1
             BE= MIN(N, BN*BATCH)
             ! Compute model state rank.
             M_STATE_TEMP(BS:BE,:) = M_STATES(BS:BE,:,I)
             CALL ORTHOGONALIZE(M_STATE_TEMP(BS:BE,:), M_LENGTHS(:,BATCH), TER, M_ORDER(:,BATCH))
             ! Compute grad state rank.
             M_STATE_TEMP(BS:BE,:) = M_GRADS(BS:BE,:,I)
             CALL ORTHOGONALIZE(M_STATE_TEMP(BS:BE,:), M_LENGTHS(:,BATCH), TGR, M_ORDER(:,BATCH))
          END DO
          !$OMP END PARALLEL DO
          TOTAL_EVAL_RANK = TOTAL_EVAL_RANK + TER
          TOTAL_GRAD_RANK = TOTAL_GRAD_RANK + TGR
       END DO
    END IF
  CONTAINS

    ! Set the input vectors and the state vectors to 
    SUBROUTINE UNIT_MAX_NORM(MDI, MDS, MNS, INPUT_VECS, STATE_VECS)
      INTEGER, INTENT(IN) :: MDI, MDS, MNS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI,MDS)              :: INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MDS,MAX(0,MNS-1)) :: STATE_VECS
      REAL(KIND=RT) :: SCALAR
      INTEGER :: L
      !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) PRIVATE(SCALAR)
      DO L = 1, MNS
         IF (L .LT. MNS) THEN
            SCALAR = SQRT(MAXVAL(SUM(STATE_VECS(:,:,L)**2, 1)))
            STATE_VECS(:,:,L) = STATE_VECS(:,:,L) / SCALAR
         ELSE
            SCALAR = SQRT(MAXVAL(SUM(INPUT_VECS(:,:)**2, 1)))
            INPUT_VECS(:,:) = INPUT_VECS(:,:) / SCALAR
         END IF
      END DO
      !$OMP END PARALLEL DO
    END SUBROUTINE UNIT_MAX_NORM

  END SUBROUTINE CONDITION_MODEL



  ! Fit input / output pairs by minimizing mean squared error.
  SUBROUTINE MINIMIZE_MSE(CONFIG, MODEL, RWORK, IWORK, &
       AX, AXI, SIZES, X, XI, Y, &
       STEPS, RECORD, SUM_SQUARED_ERROR, INFO)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: RWORK
    INTEGER,       INTENT(INOUT), DIMENSION(:) :: IWORK
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN),    DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    INTEGER,       INTENT(IN) :: STEPS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(6,STEPS), OPTIONAL :: RECORD
    REAL(KIND=RT), INTENT(OUT) :: SUM_SQUARED_ERROR
    INTEGER,       INTENT(OUT) :: INFO
    ! Local variables.
    !    measured gradient contribution of all input points
    REAL(KIND=RT), DIMENSION(SIZE(AX,2)) :: AX_CONTRIB
    REAL(KIND=RT), DIMENSION(SIZE(X,2)) :: X_CONTRIB
    !    count of how many steps have been taken since last usage
    INTEGER, DIMENSION(SIZE(AX,2)) :: AX_UNUSED_STEPS
    INTEGER, DIMENSION(SIZE(X,2)) :: X_UNUSED_STEPS
    !    indices (used for sorting and selecting points for gradient computation)
    INTEGER, DIMENSION(SIZE(AX,2)) :: AX_INDICES
    INTEGER, DIMENSION(SIZE(X,2)) :: X_INDICES
    !    "backspace" character array for printing to the same line repeatedly
    CHARACTER(LEN=*), PARAMETER :: RESET_LINE = REPEAT(CHAR(8),27)
    !    temporary holders for overwritten CONFIG attributes
    LOGICAL :: APPLY_SHIFT
    INTEGER :: NUM_THREADS
    !    miscellaneous (hard to concisely categorize)
    LOGICAL :: DID_PRINT
    INTEGER :: STEP, BATCH, NB, NS, SS, SE, MIN_TO_UPDATE, D, VS, VE, T
    INTEGER :: TOTAL_RANK, TOTAL_EVAL_RANK, TOTAL_GRAD_RANK
    INTEGER(KIND=INT64) :: CURRENT_TIME, CLOCK_RATE, CLOCK_MAX, LAST_PRINT_TIME, WAIT_TIME
    REAL(KIND=RT) :: MSE, PREV_MSE, BEST_MSE
    REAL(KIND=RT) :: STEP_MEAN_REMAIN, STEP_CURV_REMAIN

    ! Check for a valid data shape given the model.
    INFO = 0
    ! Check the shape of all inputs (to make sure they match this model).
    CALL CHECK_SHAPE(CONFIG, MODEL, AX, AXI, SIZES, X, XI, Y, INFO)
    ! Do shape checks on the work space provided.
    IF (SIZE(RWORK) .LT. CONFIG%RWORK_SIZE) THEN
       INFO = 13
    ELSE IF (SIZE(IWORK) .LT. CONFIG%IWORK_SIZE) THEN
       INFO = 14
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (CONFIG%NA .LT. 1)) THEN
       INFO = 15
    ELSE IF ((CONFIG%MDI .GT. 0) .AND. (CONFIG%NM .LT. 1)) THEN
       INFO = 16
    END IF
    IF (INFO .NE. 0) RETURN
    ! Unpack all of the work storage into the expected shapes.
    CALL UNPACKED_MINIMIZE_MSE(&
         RWORK(CONFIG%SMG : CONFIG%EMG), & ! MODEL_GRAD(NUM_VARS,NUM_THREADS)
         RWORK(CONFIG%SMGM : CONFIG%EMGM), & ! MODEL_GRAD_MEAN(NUM_VARS)
         RWORK(CONFIG%SMGC : CONFIG%EMGC), & ! MODEL_GRAD_CURV(NUM_VARS)
         RWORK(CONFIG%SBM : CONFIG%EBM), & ! BEST_MODEL(NUM_VARS)
         RWORK(CONFIG%SYG : CONFIG%EYG), & ! Y_GRADIENT(MDO,NM)
         RWORK(CONFIG%SMXS : CONFIG%EMXS), & ! M_STATES(NM,MDS,MNS+1)
         RWORK(CONFIG%SMXG : CONFIG%EMXG), & ! M_GRADS(NM,MDS,MNS+1)
         RWORK(CONFIG%SAXS : CONFIG%EAXS), & ! A_STATES(NA,ADS,ANS+1)
         RWORK(CONFIG%SAXG : CONFIG%EAXG), & ! A_GRADS(NA,ADS,ANS+1)
         RWORK(CONFIG%SAY : CONFIG%EAY), & ! AY(NA,ADO)
         RWORK(CONFIG%SAYG : CONFIG%EAYG), & ! AY_GRAD(NA,ADO)
         RWORK(CONFIG%SMXR : CONFIG%EMXR), & ! X_RESCALE(MDN,MDN)
         RWORK(CONFIG%SMXIS : CONFIG%EMXIS), & ! XI_SHIFT(MDE)
         RWORK(CONFIG%SMXIR : CONFIG%EMXIR), & ! XI_RESCALE(MDE,MDE)
         RWORK(CONFIG%SAXR : CONFIG%EAXR), & ! AX_RESCALE(ADN,ADN)
         RWORK(CONFIG%SAXIS : CONFIG%EAXIS), & ! AXI_SHIFT(ADE)
         RWORK(CONFIG%SAXIR : CONFIG%EAXIR), & ! AXI_RESCALE(ADE,ADE)
         RWORK(CONFIG%SAYR : CONFIG%EAYR), & ! AY_RESCALE(ADO)
         RWORK(CONFIG%SYR : CONFIG%EYR), & ! Y_RESCALE(MDO,MDO)
         RWORK(CONFIG%SAL : CONFIG%EAL), & ! A_LENGTHS
         RWORK(CONFIG%SML : CONFIG%EML), & ! M_LENGTHS
         RWORK(CONFIG%SAST : CONFIG%EAST), & ! A_STATE_TEMP
         RWORK(CONFIG%SMST : CONFIG%EMST), & ! M_STATE_TEMP
         MODEL(CONFIG%ASIV : CONFIG%AEIV), & ! APOSITIONAL_INPUT_VECS
         MODEL(CONFIG%MSIV : CONFIG%MEIV), & ! MODEL_INPUT_VECS
         MODEL(CONFIG%ASOV : CONFIG%AEOV), & ! APOSITIONAL_OUTPUT_VECS
         MODEL(CONFIG%MSOV : CONFIG%MEOV), & ! MODEL_OUTPUT_VECS
         IWORK(CONFIG%SUI : CONFIG%EUI), & ! UPDATE_INDICES(NUM_VARS)
         IWORK(CONFIG%SBAS : CONFIG%EBAS), & ! BATCHA_STARTS(NUM_THREADS)
         IWORK(CONFIG%SBAE : CONFIG%EBAE), & ! BATCHA_ENDS(NUM_THREADS)
         IWORK(CONFIG%SBMS : CONFIG%EBMS), & ! BATCHM_STARTS(NUM_THREADS)
         IWORK(CONFIG%SBME : CONFIG%EBME), & ! BATCHM_ENDS(NUM_THREADS)
         IWORK(CONFIG%SAO : CONFIG%EAO), & ! A_ORDER
         IWORK(CONFIG%SMO : CONFIG%EMO) & ! M_ORDER
         )

  CONTAINS

    ! Unpack the work arrays into the proper shapes.
    SUBROUTINE UNPACKED_MINIMIZE_MSE(&
         MODEL_GRAD, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, BEST_MODEL, &
         Y_GRADIENT, M_STATES, M_GRADS, A_STATES, A_GRADS, &
         AY, AY_GRAD, X_RESCALE, XI_SHIFT, XI_RESCALE, &
         AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_RESCALE, Y_RESCALE, &
         A_LENGTHS, M_LENGTHS, A_STATE_TEMP, M_STATE_TEMP, &
         A_IN_VECS, M_IN_VECS, A_OUT_VECS, M_OUT_VECS, &
         UPDATE_INDICES, BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS, &
         A_ORDER, M_ORDER)
      ! Definition of unpacked work storage.
      REAL(KIND=RT), DIMENSION(CONFIG%NUM_VARS,CONFIG%NUM_THREADS) :: MODEL_GRAD
      REAL(KIND=RT), DIMENSION(CONFIG%NUM_VARS) :: MODEL_GRAD_MEAN, MODEL_GRAD_CURV, BEST_MODEL
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADS, CONFIG%ANS+1) :: A_STATES, A_GRADS
      REAL(KIND=RT), DIMENSION(CONFIG%NM, CONFIG%MDS, CONFIG%MNS+1) :: M_STATES, M_GRADS
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADO) :: AY, AY_GRAD
      REAL(KIND=RT), DIMENSION(SIZE(Y,1), CONFIG%NM) :: Y_GRADIENT
      REAL(KIND=RT), DIMENSION(CONFIG%ADN, CONFIG%ADN) :: AX_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADE) :: AXI_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%ADE, CONFIG%ADE) :: AXI_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADO) :: AY_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%MDN, CONFIG%MDN) :: X_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%MDE) :: XI_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%MDE, CONFIG%MDE) :: XI_RESCALE
      REAL(KIND=RT), DIMENSION(SIZE(Y,1), SIZE(Y,1)) :: Y_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADS, CONFIG%NUM_THREADS) :: A_LENGTHS
      REAL(KIND=RT), DIMENSION(CONFIG%MDS, CONFIG%NUM_THREADS) :: M_LENGTHS
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADS) :: A_STATE_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%NM, CONFIG%MDS) :: M_STATE_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%ADI, CONFIG%ADS) :: A_IN_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDI, CONFIG%MDS) :: M_IN_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%ADSO, CONFIG%ADO) :: A_OUT_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDSO, CONFIG%MDO) :: M_OUT_VECS
      INTEGER, DIMENSION(CONFIG%NUM_VARS) :: UPDATE_INDICES
      INTEGER, DIMENSION(CONFIG%NUM_THREADS) :: &
           BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS
      INTEGER, DIMENSION(CONFIG%ADS, CONFIG%NUM_THREADS) :: A_ORDER
      INTEGER, DIMENSION(CONFIG%MDS, CONFIG%NUM_THREADS) :: M_ORDER
      ! 
      ! ----------------------------------------------------------------
      !                 Initialization and preparation
      ! 
      ! Cap the "number [of variables] to update" at the model size.
      CONFIG%NUM_TO_UPDATE = MAX(1,MIN(CONFIG%NUM_TO_UPDATE, CONFIG%NUM_VARS))
      ! Set the "total rank", the number of internal state components.
      TOTAL_RANK = CONFIG%MDS*CONFIG%MNS + CONFIG%ADS*CONFIG%ANS
      ! Compute the minimum number of model parameters to update.
      MIN_TO_UPDATE = MAX(1,INT(CONFIG%MIN_UPDATE_RATIO * REAL(CONFIG%NUM_VARS,RT)))
      ! Set the initial "number of steps taken since best" counter.
      NS = 0
      ! Set the num batches (NB).
      NB = MIN(CONFIG%NUM_THREADS, SIZE(Y,2))
      CALL COMPUTE_BATCHES(NB, CONFIG%NA, CONFIG%NM, SIZES, &
           BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS, INFO)
      IF (INFO .NE. 0) THEN
         Y(:,:) = 0.0_RT
         RETURN
      END IF
      ! Store the start time of this routine (to make sure updates can
      !  be shown to the user at a reasonable frequency).
      CALL SYSTEM_CLOCK(LAST_PRINT_TIME, CLOCK_RATE, CLOCK_MAX)
      WAIT_TIME = CLOCK_RATE * CONFIG%PRINT_DELAY_SEC
      DID_PRINT = .FALSE.
      ! Initial rates of change of mean and variance values.
      STEP_MEAN_REMAIN = 1.0_RT - CONFIG%STEP_MEAN_CHANGE
      STEP_CURV_REMAIN = 1.0_RT - CONFIG%STEP_CURV_CHANGE
      ! Initial mean squared error is "max representable value".
      PREV_MSE = HUGE(PREV_MSE)
      BEST_MSE = HUGE(BEST_MSE)
      ! Set the default size start and end indices for when it is absent.
      IF (SIZE(SIZES) .EQ. 0) THEN
         SS = 1
         SE = -1
      END IF
      ! Disable the application of SHIFT (since data is / will be normalized).
      APPLY_SHIFT = CONFIG%APPLY_SHIFT
      CONFIG%APPLY_SHIFT = .FALSE.
      ! Normalize the data.
      CALL NORMALIZE_DATA(CONFIG, MODEL, AX, AXI, AY, SIZES, X, XI, Y, &
           AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_RESCALE, X_RESCALE, &
           XI_SHIFT, XI_RESCALE, Y_RESCALE, A_STATES, &
           MODEL(CONFIG%ASEV:CONFIG%AEEV), &
           MODEL(CONFIG%MSEV:CONFIG%MEEV), &
           MODEL(CONFIG%ASOV:CONFIG%AEOV), INFO)
      IF (INFO .NE. 0) RETURN
      ! Set the number of threads to 1 to prevent nested parallelization.
      NUM_THREADS = CONFIG%NUM_THREADS
      CONFIG%NUM_THREADS = 1
      ! 
      ! ----------------------------------------------------------------
      !                    Minimizing mean squared error
      ! 
      ! Iterate, taking steps with the average gradient over all data.
      fit_loop : DO STEP = 1, STEPS
         ! 
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !                       Compute model gradient 
         ! 
         ! Compute the average gradient over all points.
         SUM_SQUARED_ERROR = 0.0_RT
         ! Set gradients to zero initially.
         MODEL_GRAD(:,:) = 0.0_RT
         !$OMP PARALLEL DO NUM_THREADS(NB) FIRSTPRIVATE(SS, SE) &
         !$OMP& REDUCTION(+: SUM_SQUARED_ERROR)
         DO BATCH = 1, NB
            ! Set the size start and end.
            IF (CONFIG%ADI .GT. 0) THEN
               SS = BATCHM_STARTS(BATCH)
               SE = BATCHM_ENDS(BATCH)
            END IF
            ! Sum the gradient over all data batches.
            CALL MODEL_GRADIENT(CONFIG, MODEL(:), &
                 AX(:,BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH)), &
                 AXI(:,BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH)), &
                 AY(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:), &
                 SIZES(SS:SE), &
                 X(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                 XI(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                 Y(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                 SUM_SQUARED_ERROR, MODEL_GRAD(:,BATCH), &
                 AY_GRAD(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:), &
                 Y_GRADIENT(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                 A_STATES(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:,:), &
                 A_GRADS(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:,:), &
                 M_STATES(BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH),:,:), &
                 M_GRADS(BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH),:,:), &
                 INFO)
         END DO
         !$OMP END PARALLEL DO
         IF (INFO .NE. 0) RETURN
         ! Aggregate over computed batches.
         MODEL_GRAD(:,1) = SUM(MODEL_GRAD(:,:),2)
         ! 
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !           Update the step factors, early stop if appropaite.
         ! 
         ! Convert the sum of squared errors into the mean squared error.
         MSE = SUM_SQUARED_ERROR / REAL(SIZE(Y),RT) ! RNY * SIZE(Y,1)
         ! Adjust exponential sliding windows based on change in error.
         IF (MSE .LE. PREV_MSE) THEN
            CONFIG%STEP_FACTOR = CONFIG%STEP_FACTOR * CONFIG%FASTER_RATE
            CONFIG%STEP_MEAN_CHANGE = CONFIG%STEP_MEAN_CHANGE * CONFIG%SLOWER_RATE
            STEP_MEAN_REMAIN = 1.0_RT - CONFIG%STEP_MEAN_CHANGE
            CONFIG%STEP_CURV_CHANGE = CONFIG%STEP_CURV_CHANGE * CONFIG%SLOWER_RATE
            STEP_CURV_REMAIN = 1.0_RT - CONFIG%STEP_CURV_CHANGE
            CONFIG%NUM_TO_UPDATE = CONFIG%NUM_TO_UPDATE + INT(0.05_RT * REAL(CONFIG%NUM_VARS,RT))
         ELSE
            CONFIG%STEP_FACTOR = CONFIG%STEP_FACTOR * CONFIG%SLOWER_RATE
            CONFIG%STEP_MEAN_CHANGE = CONFIG%STEP_MEAN_CHANGE * CONFIG%FASTER_RATE
            STEP_MEAN_REMAIN = 1.0_RT - CONFIG%STEP_MEAN_CHANGE
            CONFIG%STEP_CURV_CHANGE = CONFIG%STEP_CURV_CHANGE * CONFIG%FASTER_RATE
            STEP_CURV_REMAIN = 1.0_RT - CONFIG%STEP_CURV_CHANGE
            CONFIG%NUM_TO_UPDATE = CONFIG%NUM_TO_UPDATE - INT(0.05_RT * REAL(CONFIG%NUM_VARS,RT))
         END IF
         ! Project the number of variables to update into allowable bounds.
         CONFIG%NUM_TO_UPDATE = MIN(CONFIG%NUM_VARS, MAX(MIN_TO_UPDATE, CONFIG%NUM_TO_UPDATE))
         ! Store the previous error for tracking the best-so-far.
         PREV_MSE = MSE
         ! Update the step number.
         NS = NS + 1
         ! Update the saved "best" model based on error.
         IF (MSE .LT. BEST_MSE) THEN
            NS = 0
            BEST_MSE = MSE
            IF (CONFIG%KEEP_BEST) THEN
               BEST_MODEL(:) = MODEL(1:CONFIG%NUM_VARS)
            END IF
         ! Early stop if we don't expect to see a better solution
         !  by the time the fit operation is complete.
         ELSE IF (CONFIG%EARLY_STOP .AND. (NS .GT. STEPS - STEP)) THEN
            EXIT fit_loop
         END IF
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !              Modify the model parameters (take step).
         ! 
         ! Convert the summed gradient to average gradient.
         MODEL_GRAD(:,1) = MODEL_GRAD(:,1) / REAL(NB,RT)
         MODEL_GRAD_MEAN(:) = STEP_MEAN_REMAIN * MODEL_GRAD_MEAN(:) &
              + CONFIG%STEP_MEAN_CHANGE * MODEL_GRAD(:,1)
         MODEL_GRAD_CURV(:) = STEP_CURV_REMAIN * MODEL_GRAD_CURV(:) &
              + CONFIG%STEP_CURV_CHANGE * (MODEL_GRAD_MEAN(:) - MODEL_GRAD(:,1))**2
         MODEL_GRAD_CURV(:) = MAX(MODEL_GRAD_CURV(:), EPSILON(CONFIG%STEP_FACTOR))
         ! Set the step as the mean direction (over the past few steps).
         MODEL_GRAD(:,1) = MODEL_GRAD_MEAN(:)
         ! Start scaling by step magnitude by curvature once enough data is collected.
         IF (STEP .GE. CONFIG%MIN_STEPS_TO_STABILITY) THEN
            MODEL_GRAD(:,1) = MODEL_GRAD(:,1) / SQRT(MODEL_GRAD_CURV(:))
         END IF
         ! Update as many parameters as it seems safe to update (and still converge).
         IF (CONFIG%NUM_TO_UPDATE .LT. CONFIG%NUM_VARS) THEN
            ! Identify the subset of components that will be updapted this step.
            CALL ARGSELECT(-ABS(MODEL_GRAD(:,1)), &
                 CONFIG%NUM_TO_UPDATE, UPDATE_INDICES(:))
            ! Take the gradient steps (based on the computed "step" above).
            MODEL(UPDATE_INDICES(1:CONFIG%NUM_TO_UPDATE)) = &
                 MODEL(UPDATE_INDICES(1:CONFIG%NUM_TO_UPDATE)) &
                 - MODEL_GRAD(UPDATE_INDICES(1:CONFIG%NUM_TO_UPDATE),1) &
                 * CONFIG%STEP_FACTOR
         ELSE
            ! Take the gradient steps (based on the computed "step" above).
            MODEL(1:CONFIG%NUM_VARS) = MODEL(1:CONFIG%NUM_VARS) &
                 - MODEL_GRAD(:,1) * CONFIG%STEP_FACTOR
         END IF
         CONFIG%STEPS_TAKEN = CONFIG%STEPS_TAKEN + 1
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         ! Rescale internal vectors to have a maximum 2-norm of 1.
         ! Center the outputs of the apositional model about the origin.
         CALL CONDITION_MODEL(CONFIG, MODEL, NUM_THREADS, CONFIG%STEPS_TAKEN, AY, &
              A_STATES(:,:,:), M_STATES(:,:,:), A_GRADS(:,:,:), M_GRADS(:,:,:), &
              A_LENGTHS(:,:), M_LENGTHS(:,:), A_STATE_TEMP(:,:), M_STATE_TEMP(:,:), &
              A_ORDER(:,:), M_ORDER(:,:), NB, BATCHA_STARTS(:), BATCHA_ENDS(:), &
              BATCHM_STARTS(:), BATCHM_ENDS(:), TOTAL_EVAL_RANK, TOTAL_GRAD_RANK)
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         ! Record various statistics that are currently of interest (for research).
         IF (PRESENT(RECORD)) THEN
            ! Store the mean squared error at this iteration.
            RECORD(1,STEP) = MSE
            ! Store the current multiplier on the step.
            RECORD(2,STEP) = CONFIG%STEP_FACTOR
            ! Store the norm of the step that was taken (intermittently).
            IF (MOD(CONFIG%STEPS_TAKEN-1,CONFIG%LOGGING_STEP_FREQUENCY) .EQ. 0) THEN
               RECORD(3,STEP) = SQRT(MAX(EPSILON(0.0_RT), SUM(MODEL_GRAD(:,1)**2))) / SQRT(REAL(CONFIG%NUM_VARS,RT))
            ELSE
               RECORD(3,STEP) = RECORD(3,STEP-1)
            END IF
            ! Store the percentage of parameters updated in this step.
            RECORD(4,STEP) = REAL(CONFIG%NUM_TO_UPDATE,RT) / REAL(CONFIG%NUM_VARS)
            ! Store the evaluative utilization rate (total data rank over full rank)
            RECORD(5,STEP) = REAL(TOTAL_EVAL_RANK,RT) / REAL(TOTAL_RANK,RT)
            ! Store the gradient utilization rate (total gradient rank over full rank)
            RECORD(6,STEP) = REAL(TOTAL_GRAD_RANK,RT) / REAL(CONFIG%MDS*CONFIG%MNS + CONFIG%ADS*CONFIG%ANS,RT)
         END IF
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         ! Write an update about step and convergence to the command line.
         CALL SYSTEM_CLOCK(CURRENT_TIME, CLOCK_RATE, CLOCK_MAX)
         IF (CURRENT_TIME - LAST_PRINT_TIME .GT. WAIT_TIME) THEN
            IF (DID_PRINT) WRITE (*,'(A)',ADVANCE='NO') RESET_LINE
            WRITE (*,'(I6,"  (",F6.4,") [",F6.4,"]")', ADVANCE='NO') STEP, MSE, BEST_MSE
            LAST_PRINT_TIME = CURRENT_TIME
            DID_PRINT = .TRUE.
         END IF
      END DO fit_loop
      ! 
      ! ----------------------------------------------------------------
      !                 Finalization, prepare for return.
      ! 
      ! Restore the best model seen so far (if enough steps were taken).
      IF (CONFIG%KEEP_BEST .AND. (STEPS .GT. 0)) THEN
         MSE                      = BEST_MSE
         MODEL(1:CONFIG%NUM_VARS) = BEST_MODEL(:)
      END IF
      ! 
      ! Apply the data normalizing scaling factors to the weight
      !  matrices to embed normalization into the model.
      IF (CONFIG%ENCODE_NORMALIZATION) THEN
         IF (CONFIG%ADN .GT. 0) THEN
            IF (CONFIG%ANS .GT. 0) THEN
               A_IN_VECS(:CONFIG%ADN,:) = MATMUL(AX_RESCALE(:,:), A_IN_VECS(:CONFIG%ADN,:))
            ELSE
               A_OUT_VECS(:CONFIG%ADN,:) = MATMUL(AX_RESCALE(:,:), A_OUT_VECS(:CONFIG%ADN,:))
            END IF
         END IF
         IF (CONFIG%MDN .GT. 0) THEN
            IF (CONFIG%MNS .GT. 0) THEN
               M_IN_VECS(:CONFIG%MDN,:) = MATMUL(X_RESCALE(:,:), M_IN_VECS(:CONFIG%MDN,:))
            ELSE
               M_OUT_VECS(:CONFIG%MDN,:) = MATMUL(X_RESCALE(:,:), M_OUT_VECS(:CONFIG%MDN,:))
            END IF
         END IF
         ! Apply the output rescale to whichever part of the model produces output.
         IF (CONFIG%MDO .GT. 0) THEN
            M_OUT_VECS(:,:) = MATMUL(M_OUT_VECS(:,:), Y_RESCALE(:,:))
         ELSE
            A_OUT_VECS(:,:) = MATMUL(A_OUT_VECS(:,:), Y_RESCALE(:,:))
         END IF
      END IF
      ! 
      ! Erase the printed message if one was produced.
      IF (DID_PRINT) WRITE (*,'(A)',ADVANCE='NO') RESET_LINE
      ! 
      ! Reset configuration settings that were modified.
      CONFIG%APPLY_SHIFT = APPLY_SHIFT
      CONFIG%NUM_THREADS = NUM_THREADS
    END SUBROUTINE UNPACKED_MINIMIZE_MSE

  END SUBROUTINE MINIMIZE_MSE


END MODULE APOS

