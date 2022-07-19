! Module for matrix multiplication (absolutely crucial for APOS speed).
! Includes routines for orthogonalization, computing the SVD, and
! radializing data matrices with the SVD.
MODULE MATRIX_OPERATIONS
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, INT64
  USE IEEE_ARITHMETIC, ONLY: IS_NAN => IEEE_IS_NAN, IS_FINITE => IEEE_IS_FINITE
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

    ! Standard SGEMM.
    INTERFACE
       SUBROUTINE SGEMM(OP_A, OP_B, OUT_ROWS, OUT_COLS, INNER_DIM, &
            AB_MULT, A, A_ROWS, B, B_ROWS, C_MULT, C, C_ROWS)
         CHARACTER :: OP_A, OP_B
         INTEGER :: OUT_ROWS, OUT_COLS, INNER_DIM, A_ROWS, B_ROWS, C_ROWS
         REAL :: AB_MULT, C_MULT
         REAL, DIMENSION(*) :: A
         REAL, DIMENSION(*) :: B
         REAL, DIMENSION(*) :: C
       END SUBROUTINE SGEMM
    END INTERFACE
    CALL SGEMM(OP_A, OP_B, OUT_ROWS, OUT_COLS, INNER_DIM, &
       AB_MULT, A, A_ROWS, B, B_ROWS, C_MULT, C, C_ROWS)

    ! ! Numpy SGEMM_64 (with 64-bit integer types).
    ! INTERFACE
    !    SUBROUTINE SGEMM_64(OP_A, OP_B, OUT_ROWS, OUT_COLS, INNER_DIM, &
    !         AB_MULT, A, A_ROWS, B, B_ROWS, C_MULT, C, C_ROWS)
    !      USE ISO_FORTRAN_ENV, ONLY: INT64
    !      CHARACTER :: OP_A, OP_B
    !      INTEGER(KIND=INT64) :: OUT_ROWS, OUT_COLS, INNER_DIM, A_ROWS, B_ROWS, C_ROWS
    !      REAL :: AB_MULT, C_MULT
    !      REAL, DIMENSION(*) :: A
    !      REAL, DIMENSION(*) :: B
    !      REAL, DIMENSION(*) :: C
    !    END SUBROUTINE SGEMM_64
    ! END INTERFACE
    ! CALL SGEMM_64(OP_A, OP_B, INT(OUT_ROWS,KIND=INT64), INT(OUT_COLS,KIND=INT64), INT(INNER_DIM,KIND=INT64), &
    !    AB_MULT, A, INT(A_ROWS,KIND=INT64), B, INT(B_ROWS,KIND=INT64), C_MULT, C, INT(C_ROWS,KIND=INT64))

  END SUBROUTINE GEMM

  ! Orthogonalize and normalize column vectors of A with pivoting.
  SUBROUTINE ORTHOGONALIZE(A, LENGTHS, RANK, ORDER, MULTIPLIERS)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: LENGTHS ! SIZE(A,2)
    INTEGER, INTENT(OUT), OPTIONAL :: RANK
    INTEGER, INTENT(OUT), DIMENSION(:), OPTIONAL :: ORDER ! SIZE(A,2)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:), OPTIONAL :: MULTIPLIERS ! SIZE(A,2), SIZE(A,2)
    REAL(KIND=RT) :: L, V
    INTEGER :: I, J, K
    IF (PRESENT(RANK)) RANK = 0
    IF (PRESENT(ORDER)) THEN
       FORALL (I=1:SIZE(A,2)) ORDER(I) = I
    END IF
    IF (PRESENT(MULTIPLIERS)) THEN
       MULTIPLIERS(:,:) = 0.0_RT
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
          ! Perform the column pivot.
          DO K = 1, SIZE(A,1)
             V = A(K,I)
             A(K,I) = A(K,J)
             A(K,J) = V
          END DO
          ! Do column pivot on multipliers too, if present.
          IF (PRESENT(MULTIPLIERS)) THEN
             DO K = 1, SIZE(MULTIPLIERS,1)
                V = MULTIPLIERS(K,I)
                MULTIPLIERS(K,I) = MULTIPLIERS(K,J)
                MULTIPLIERS(K,J) = V
             END DO
          END IF
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
             ! Store these multipliers if they were requested.
             IF (PRESENT(MULTIPLIERS)) THEN
                MULTIPLIERS(I,I:) = LENGTHS(I:)
             END IF
          END IF
          IF (PRESENT(RANK)) RANK = RANK + 1
       ELSE
          LENGTHS(I:) = 0.0_RT
          ! A(:,I:) = 0.0_RT ! <- Expected or not? Unclear.
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
    ! Local variables. LOCAL ALLOCATION
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
    ! Local variables. LOCAL ALLOCATION
    LOGICAL :: INVERSE, FLAT
    LOGICAL, DIMENSION(SIZE(X,1), SIZE(X,2)) :: NON_NUMBER_MASK
    REAL(KIND=RT), DIMENSION(SIZE(VECS,1),SIZE(VECS,2)) :: TEMP_VECS
    REAL(KIND=RT), DIMENSION(SIZE(X,1)) :: VALS, RN, SCALAR
    REAL(KIND=RT), DIMENSION(SIZE(X,1), SIZE(X,2)) :: X1
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
    ! Identify the location of "bad values" (Inf and NaN).
    NON_NUMBER_MASK(:,:) = IS_NAN(X(:,:)) .OR. (.NOT. IS_FINITE(X(:,:)))
    ! Set all nonnumber values to zero (so they do not affect computed shifts).
    WHERE (NON_NUMBER_MASK(:,:))
       X(:,:) = 0.0_RT
    END WHERE
    ! Shift the data to be be centered about the origin.
    D = SIZE(X,1)
    ! Count the number of valid numbers in each component.
    RN(:) = MAX(1.0_RT, REAL(SIZE(X,2) - COUNT(NON_NUMBER_MASK(:,:), 2), RT))
    ! Invert the mask to select only the valid numbers.
    NON_NUMBER_MASK(:,:) = .NOT. NON_NUMBER_MASK(:,:)
    DO I = 1, D
       ! Rescale the input components individually to have a maximum of 1
       !  to prevent numerical issues with SVD (will embed in VECS or undo
       !  this scaling later when storing the inverse transformation).
       SCALAR(I) = MAXVAL(ABS(X(I,:)))
       IF (SCALAR(I) .NE. 0.0_RT) THEN
          X(I,:) = X(I,:) / SCALAR(I)
       ELSE
          SCALAR(I) = 1.0_RT
       END IF
       ! Shift all valid numbers by the mean.
       !   NOTE: mask was inverted to capture valid numbers.
       SHIFT(I) = -SUM(X(I,:), MASK=NON_NUMBER_MASK(I,:)) / RN(I) 
       WHERE (NON_NUMBER_MASK(I,:))
          X(I,:) = X(I,:) + SHIFT(I)
       END WHERE
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
    ! Normalize the values associated with the singular vectors to
    !  make the output componentwise unit mean squared magnitude.
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
       ! Rescale all vectors by the average singular value.
       VALS(:) = SUM(VALS(:)) / (SQRT(RN) * REAL(D,RT))
       IF (VALS(1) .GT. 0.0_RT) THEN
          VECS(:,:) = VECS(:,:) / VALS(1)
       END IF
    END IF
    ! Apply the scaled singular vectors to the data to make it radially symmetric.
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
          VECS(I,:D) = VECS(I,:D) * SCALAR(I)
       END DO
       VECS(:D,:D) = TRANSPOSE(VECS(:D,:D))
       SHIFT(:) = -SHIFT(:) * SCALAR(:)
    ELSE
       ! Apply the exact same transformation to the vectors
       ! that was already applied to X to normalize original
       ! component scale (maximum absolute values).
       DO I = 1, D
          VECS(I,:D) = VECS(I,:D) / SCALAR(I)
       END DO
       SHIFT(:) = SHIFT(:) * SCALAR(:)
    END IF
  END SUBROUTINE RADIALIZE

  ! Perform least squares with LAPACK.
  ! 
  !   A is column vectors (of points) if TRANS='T', and row vectors 
  !     (of points) if TRANS='N'.
  !   B must be COLUMN VECTORS of fit output (1 row = 1 point).
  !   X always has a first dimension that is nonpoint axis size of A,
  !     and the second dimension is determined by B's columns (or rank),
  !     and if X is smaller then B is reduced to its principal components.
  SUBROUTINE LEAST_SQUARES(TRANS, A, B, X)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A, B
    CHARACTER, INTENT(IN) :: TRANS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: X ! MIN(SIZE(A,1),SIZE(A,2)),SIZE(B,2)
    ! Local variables.
    INTEGER :: M, N, NRHS, LDA, LDB, LWORK, INFO
    REAL(KIND=RT), DIMENSION(:), ALLOCATABLE :: WORK
    REAL(KIND=RT), DIMENSION(:,:), ALLOCATABLE :: PROJECTION
    INTERFACE
       SUBROUTINE SGELS(TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK, INFO)
         USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
         CHARACTER, INTENT(IN) :: TRANS
         INTEGER, INTENT(IN) :: M, N, NRHS, LDA, LDB, LWORK
         REAL(KIND=RT), INTENT(INOUT), DIMENSION(LDA,*) :: A
         REAL(KIND=RT), INTENT(IN), DIMENSION(LDB,*) :: B
         REAL(KIND=RT), INTENT(INOUT), DIMENSION(*) :: WORK
         INTEGER, INTENT(OUT) :: INFO
       END SUBROUTINE SGELS
    END INTERFACE
    ! TODO: Test this function since I redefined the interface and updated some code.
    ! 
    ! Reduce the rank of B to the desired size, if appropriate.
    IF (SIZE(X,2) .LT. SIZE(B,2)) THEN
       NRHS = MIN(SIZE(B,1), SIZE(X,2))
       ALLOCATE( WORK(NRHS), PROJECTION(NRHS,NRHS) )
       ! Compute the SVD to use as the projection.
       CALL SVD(B, WORK, PROJECTION, RANK=NRHS, STEPS=10)
       ! Project B down (and zero out the remaining columns).
       B(:,:NRHS) = MATMUL(B(:,:), PROJECTION(:,:NRHS))
       IF (SIZE(B,2) .GT. NRHS) B(:,NRHS+1:) = 0.0_RT
       DEALLOCATE(PROJECTION, WORK)
    ELSE
       NRHS = MIN(SIZE(B,1), SIZE(B,2))
    END IF
    ! Set variables for calling least squares routine.
    M = SIZE(A,1)
    N = SIZE(A,2)
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
