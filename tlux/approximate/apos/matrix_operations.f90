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

  ! Generate randomly distributed vectors on the N-sphere.
  SUBROUTINE RANDOM_UNIT_VECTORS(COLUMN_VECTORS)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: COLUMN_VECTORS
    ! Local variables.
    REAL(KIND=RT), DIMENSION(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2)) :: TEMP_VECS
    REAL(KIND=RT), PARAMETER :: PI = 3.141592653589793
    REAL(KIND=RT) :: LEN
    INTEGER :: I, J, K
    ! Skip empty vector sets.
    IF (SIZE(COLUMN_VECTORS) .LE. 0) RETURN
    ! Generate random numbers in the range [0,1].
    CALL RANDOM_NUMBER(COLUMN_VECTORS(:,:))
    CALL RANDOM_NUMBER(TEMP_VECS(:,:))
    ! Map the random uniform numbers to a radial distribution.
    COLUMN_VECTORS(:,:) = SQRT(-LOG(COLUMN_VECTORS(:,:))) * COS(PI * TEMP_VECS(:,:))
    ! Orthogonalize the vectors in (random) order.
    IF (SIZE(COLUMN_VECTORS,1) .GT. 1) THEN
       ! Compute the last vector that is part of the orthogonalization.
       K = MIN(SIZE(COLUMN_VECTORS,1), SIZE(COLUMN_VECTORS,2))
       ! Orthogonalize the "lazy way" without column pivoting.
       ! Could result in imperfectly orthogonal vectors (because of
       ! rounding errors being enlarged by upscaling), that is acceptable.
       DO I = 1, K-1
          LEN = NORM2(COLUMN_VECTORS(:,I))
          IF (LEN .GT. 0.0_RT) THEN
             ! Make this column unit length.
             COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
             ! Compute multipliers (store in row of TEMP_VECS) and subtract
             ! from all remaining columns (doing the orthogonalization).
             TEMP_VECS(1,I+1:K) = MATMUL(COLUMN_VECTORS(:,I), COLUMN_VECTORS(:,I+1:K))
             DO J = I+1, K
                COLUMN_VECTORS(:,J) = COLUMN_VECTORS(:,J) - TEMP_VECS(1,J) * COLUMN_VECTORS(:,I)
             END DO
          ELSE
             ! This should not happen (unless the vectors are at least in the
             !   tens of thousands, in which case a different method should be used).
             PRINT *, 'ERROR: Random unit vector failed to initialize correctly, rank deficient.'
          END IF
       END DO
       ! Make the rest of the column vectors unit length.
       DO I = K, SIZE(COLUMN_VECTORS,2)
          LEN = NORM2(COLUMN_VECTORS(:,I))
          IF (LEN .GT. 0.0_RT)  COLUMN_VECTORS(:,I) = COLUMN_VECTORS(:,I) / LEN
       END DO
    END IF
  END SUBROUTINE RANDOM_UNIT_VECTORS

  ! Orthogonalize and normalize column vectors of A with pivoting.
  SUBROUTINE ORTHOGONALIZE(A, LENGTHS, RANK, ORDER)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: LENGTHS ! SIZE(A,2)
    INTEGER, INTENT(OUT), OPTIONAL :: RANK
    INTEGER, INTENT(OUT), DIMENSION(:), OPTIONAL :: ORDER ! SIZE(A,2)
    REAL(KIND=RT) :: L, V
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
          ! Perform the pivot.
          DO K = 1, SIZE(A,1)
             V = A(K,I)
             A(K,I) = A(K,J)
             A(K,J) = V
          END DO
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
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: S ! MIN(SIZE(A,1),SIZE(A,2))
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: VT ! MIN(SIZE(A,1),SIZE(A,2)), MIN(SIZE(A,1),SIZE(A,2))
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
    VT(1:SIZE(ATA,1),1:SIZE(ATA,2)) = ATA(:,:)
    ! Fill remaining columns (if extra were provided) with zeros.
    IF ((SIZE(VT,1) .GT. SIZE(ATA,1)) .OR. (SIZE(VT,2) .GT. SIZE(ATA,2))) THEN
       VT(SIZE(ATA,1)+1:,SIZE(ATA,2)+1:) = 0.0_RT
    END IF
    ! Orthogonalize and reorder by magnitudes.
    CALL ORTHOGONALIZE(VT(:,:), S(:), RANK)
    ! Do power iterations.
    power_iteration : DO I = 1, NUM_STEPS
       Q(:,:) = VT(1:SIZE(Q,1),1:SIZE(Q,2))
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

  ! Perform least squares with LAPACK.
  ! 
  !   A is column vectors (of points) if TRANS='T', and row vectors 
  !     (of points) if TRANS='N'.
  !   B must be COLUMN VECTORS of fit output (1 row = 1 point).
  !   X always has a first dimension that is nonpoint axis size of A,
  !     and the second dimension is determined by B's columns (or rank),
  !     or (if smaller), then B is reduced to its principal components.
  SUBROUTINE LEAST_SQUARES(TRANS, A, B, X)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A, B
    CHARACTER, INTENT(IN) :: TRANS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: X ! MIN(SIZE(A,1),SIZE(A,2)),SIZE(B,2)
    ! Local variables.
    INTEGER :: M, N, R, NRHS, LDA, LDB, LWORK, INFO
    REAL(KIND=RT), DIMENSION(:), ALLOCATABLE :: WORK
    REAL(KIND=RT), DIMENSION(:,:), ALLOCATABLE :: PROJECTION
    EXTERNAL :: SGELS
    ! Reduce the rank of B to the desired size, if appropriate.
    IF (SIZE(X,2) .LT. SIZE(B,2)) THEN
       NRHS = MIN(SIZE(B,1), SIZE(B,2))
       ALLOCATE( WORK(NRHS), PROJECTION(NRHS,NRHS) )
       ! Compute the SVD to use as the projection.
       CALL SVD(B, WORK, PROJECTION, RANK=NRHS, STEPS=10)
       ! Project B down (and zero out the remaining columns).
       NRHS = MIN(NRHS, SIZE(X,2))
       B(:,:NRHS) = MATMUL(B(:,:), PROJECTION(:,:NRHS))
       IF (SIZE(B,2) .GT. NRHS) B(:,NRHS+1:) = 0.0_RT
       DEALLOCATE(PROJECTION, WORK)
    END IF
    ! Set variables for calling least squares routine.
    M = SIZE(A,1)
    N = SIZE(A,2)
    NRHS = R
    LDA = SIZE(A,1)
    LDB = SIZE(B,1)
    ! Allocate the work space for the call.
    LWORK = MAX(1, MIN(M,N) + MAX(MIN(M,N), NRHS))
    ALLOCATE(WORK(LWORK))
    ! Make the call to the least squares routine.
    CALL SGELS( TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK, INFO )
    ! Store the result.
    IF (SIZE(X,2) .LE. SIZE(B,2)) THEN
       X(:,:) = B(:SIZE(X,1),:SIZE(X,2))
    ELSE
       X(:,:SIZE(B,2)) = B(:SIZE(X,1),:SIZE(B,2))
    END IF
  END SUBROUTINE LEAST_SQUARES

END MODULE MATRIX_OPERATIONS
