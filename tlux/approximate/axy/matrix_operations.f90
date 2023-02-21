! Module for matrix multiplication (absolutely crucial for APOS speed).
! Includes routines for orthogonalization, computing the SVD, and
! radializing data matrices with the SVD.
MODULE MATRIX_OPERATIONS
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, INT64, INT32
  USE IEEE_ARITHMETIC, ONLY: IS_NAN => IEEE_IS_NAN, IS_FINITE => IEEE_IS_FINITE
  IMPLICIT NONE

CONTAINS


  ! Compute the mean of a matrix along an axis in a way that is numerically stable.
  SUBROUTINE STABLE_MEAN(MATRIX, MEAN, DIM, MASK)
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: MATRIX
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: MEAN
    INTEGER, INTENT(IN) :: DIM
    LOGICAL, INTENT(IN), OPTIONAL, DIMENSION(:,:) :: MASK
    INTEGER(KIND=INT64) :: I
    REAL(KIND=RT) :: SHIFT, SCALE
    ! Compute the mean in a stable way (scale to the unit box first).
    SHIFT = MINVAL(MATRIX, MASK=MASK)
    SCALE = MAXVAL(MATRIX, MASK=MASK) - SHIFT
    IF (SCALE .EQ. 0.0_RT) THEN
       SCALE = 1.0_RT
    END IF
    MEAN(:) = SUM((MATRIX - SHIFT) / SCALE, DIM=DIM, MASK=MASK) / REAL(SIZE(MATRIX, DIM=DIM), RT)
    MEAN(:) = SHIFT + MEAN(:) * SCALE
  END SUBROUTINE STABLE_MEAN


  ! Convenience wrapper routine for calling matrix multiply.
  ! 
  ! TODO: For very large data, automatically batch the operations to reduce
  !       the size of the temporary space that is needed.
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

    ! INTEGER(KIND=INT64) :: I, J, K
    ! C(:,:) = C(:,:) * C_MULT
    ! IF (OP_A .EQ. 'N') THEN
    !    IF (OP_B .EQ. 'N') THEN
    !       DO I = 1, OUT_ROWS
    !          DO J = 1, OUT_COLS
    !             DO K = 1, INNER_DIM
    !                C(I,J) = C(I,J) + A(I,K) * B(K,J) * AB_MULT
    !             END DO
    !          END DO
    !       END DO
    !    ELSE IF (OP_B .EQ. 'T') THEN
    !       DO I = 1, OUT_ROWS
    !          DO J = 1, OUT_COLS
    !             DO K = 1, INNER_DIM
    !                C(I,J) = C(I,J) + A(I,K) * B(J,K) * AB_MULT
    !             END DO
    !          END DO
    !       END DO
    !    ELSE
    !       PRINT *, 'ERROR: Bad OP_B value:', OP_B
    !    END IF
    ! ELSE IF (OP_A .EQ. 'T') THEN
    !    IF (OP_B .EQ. 'N') THEN
    !       DO I = 1, OUT_ROWS
    !          DO J = 1, OUT_COLS
    !             DO K = 1, INNER_DIM
    !                C(I,J) = C(I,J) + A(K,I) * B(J,K) * AB_MULT
    !             END DO
    !          END DO
    !       END DO
    !    ELSE IF (OP_B .EQ. 'T') THEN
    !       DO I = 1, OUT_ROWS
    !          DO J = 1, OUT_COLS
    !             DO K = 1, INNER_DIM
    !                C(I,J) = C(I,J) + A(K,I) * B(K,J) * AB_MULT
    !             END DO
    !          END DO
    !       END DO
    !    ELSE
    !       PRINT *, 'ERROR: Bad OP_B value:', OP_B
    !    END IF
    ! ELSE
    !    PRINT *, 'ERROR: Bad OP_A value:', OP_A
    ! END IF

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

  
  ! Convenience wrapper routine for calling symmetric matrix multiplication.
  SUBROUTINE SYRK(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC)
    CHARACTER, INTENT(IN) :: UPLO, TRANS
    INTEGER, INTENT(IN) :: N, K, LDA, LDC
    REAL(KIND=RT), INTENT(IN) :: ALPHA, BETA
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: C

    ! Standard SSYRK.
    INTERFACE
       SUBROUTINE SSYRK(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC)
         CHARACTER, INTENT(IN) :: UPLO, TRANS
         INTEGER, INTENT(IN) :: N, K, LDA, LDC
         REAL, INTENT(IN) :: ALPHA, BETA
         REAL, INTENT(IN), DIMENSION(LDA,*) :: A
         REAL, INTENT(INOUT), DIMENSION(LDC,*) :: C
       END SUBROUTINE SSYRK
    END INTERFACE
    CALL SSYRK(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC)

  END SUBROUTINE SYRK


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
       ! Subtract the current vector from all others.
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
          ! A(:,I:) = 0.0_RT ! <- Expected or not? Unclear. They're already practically zero.
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
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:,:) :: ATA, Q
    INTEGER :: I, J, K, NUM_STEPS
    REAL(KIND=RT) :: MULTIPLIER
    ! LOCAL ALLOCATION
    ALLOCATE( &
         ATA(MIN(SIZE(A,1),SIZE(A,2)),MIN(SIZE(A,1),SIZE(A,2))), &
         Q(MIN(SIZE(A,1),SIZE(A,2)),MIN(SIZE(A,1),SIZE(A,2))) &
    )
    ! Set the number of steps.
    IF (PRESENT(STEPS)) THEN
       NUM_STEPS = STEPS
    ELSE
       NUM_STEPS = 10
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
       CALL SYRK('U', 'N', K, SIZE(A,2), MULTIPLIER**2, A(:,:), &
            SIZE(A,1), 0.0_RT, ATA(:,:), K)
    ELSE
       ! ATA(:,:) = MATMUL(TRANSPOSE(A(:,:)), A(:,:))
       CALL SYRK('U', 'T', K, SIZE(A,1), MULTIPLIER**2, A(:,:), &
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
       VT(SIZE(ATA,1)+1:,:) = 0.0_RT
       VT(1:SIZE(ATA,1),SIZE(ATA,2)+1:) = 0.0_RT
    END IF
    ! Orthogonalize and reorder by magnitudes.
    CALL ORTHOGONALIZE(VT(:,:), S(:), RANK)
    ! Do power iterations.
    power_iteration : DO I = 1, NUM_STEPS
       Q(:,:) = VT(1:SIZE(Q,1),1:SIZE(Q,2))
       ! VT(:,:) = MATMUL(ATA(:,:), Q(:,:))
       CALL GEMM('N', 'N', K, K, K, 1.0_RT, &
            ATA(:,:), K, Q(:,:), K, 0.0_RT, &
            VT(:,:), K)
       CALL ORTHOGONALIZE(VT(:,:), S(:), RANK)
    END DO power_iteration
    ! Compute the singular values.
    WHERE (S(:) .NE. 0.0_RT)
       S(:) = SQRT(S(:)) / MULTIPLIER
    END WHERE
    DEALLOCATE(ATA, Q)
  END SUBROUTINE SVD

  ! If there are at least as many data points as dimension, then
  ! compute the principal components and rescale the data by
  ! projecting onto those and rescaling so that each component has
  ! identical singular values (this makes the data more "radially
  ! symmetric").
  SUBROUTINE RADIALIZE(X, SHIFT, VECS, INVERSE, FLATTEN, MAXBOUND, STEPS, MAX_TO_SQUARE)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: SHIFT
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: VECS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:), OPTIONAL :: INVERSE
    LOGICAL, INTENT(IN), OPTIONAL :: FLATTEN, MAXBOUND
    INTEGER, INTENT(IN), OPTIONAL :: STEPS
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: MAX_TO_SQUARE
    ! Local variables.
    LOGICAL :: FLAT, SCALE_BY_AVERAGE
    LOGICAL, ALLOCATABLE, DIMENSION(:,:) :: VALIDITY_MASK
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:,:) :: TEMP_VECS
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:) :: VALS, RN, SCALAR, MINS
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:,:) :: X1
    INTEGER(KIND=INT64) :: I, D, N
    INTEGER(KIND=INT64) :: NMAX
    D = SIZE(X,1,KIND=INT64)
    N = SIZE(X,2,KIND=INT64)
    ! LOCAL ALLOCATION
    ALLOCATE( &
         VALIDITY_MASK(N, D), &
         VALS(D), &
         RN(D), &
         MINS(D), &
         SCALAR(D), &
         X1(D, N) &
    )
    ! Set the default value for "FLAT".
    IF (PRESENT(FLATTEN)) THEN
       FLAT = FLATTEN
    ELSE
       FLAT = .TRUE.
    END IF
    ! Set the default value for "SCALE_BY_AVERAGE".
    IF (PRESENT(MAXBOUND)) THEN
       SCALE_BY_AVERAGE = .NOT. MAXBOUND
    ELSE
       SCALE_BY_AVERAGE = .TRUE.
    END IF
    ! Set default "NMAX".
    IF (PRESENT(MAX_TO_SQUARE)) THEN
       NMAX = MIN(MAX_TO_SQUARE, N)
    ELSE
       NMAX = MIN(10000000_INT64, N)
    END IF
    ! Shift the data to be be centered about the origin.
    !$OMP PARALLEL DO
    DO I = 1, D
       ! Identify the location of "bad values" (Inf and NaN).
       VALIDITY_MASK(:,I) = (&
            IS_NAN(X(I,:)) .OR. &
            (.NOT. IS_FINITE(X(I,:))) .OR. &
            (ABS(X(I,:)) .GT. (HUGE(X(I,1)) / 2.0_RT)))
       ! Set all nonnumber values to zero (so they do not affect computed shifts).
       WHERE (VALIDITY_MASK(:,I))
          X(I,:) = 0.0_RT
       END WHERE
       ! Count the number of valid numbers in each component.
       RN(I) = MAX(1.0_RT, REAL(N - COUNT(VALIDITY_MASK(:,I), KIND=INT64), KIND=RT))
       ! Invert the mask to select only the valid numbers.
       VALIDITY_MASK(:,I) = .NOT. VALIDITY_MASK(:,I)
       ! Compute the minimum value for this column.
       MINS(I) = 0.0_RT
       MINS(I) = MINVAL(X(I,:), MASK=VALIDITY_MASK(:,I))
       ! Rescale the input components individually to be
       !  in [0,1] to prevent numerical issues with SVD.
       SCALAR(I) = 1.0_RT
       SCALAR(I) = MAXVAL(X(I,:), MASK=VALIDITY_MASK(:,I)) - MINS(I)
       IF (SCALAR(I) .NE. 0.0_RT) THEN
          WHERE (VALIDITY_MASK(:,I))
             X(I,:) = (X(I,:) - MINS(I)) / SCALAR(I)
          END WHERE
       ELSE
          WHERE (VALIDITY_MASK(:,I))
             X(I,:) = (X(I,:) - MINS(I))
          END WHERE
          SCALAR(I) = 1.0_RT
       END IF
       ! Shift all valid numbers by the mean.
       !   NOTE: mask was inverted above to capture VALID numbers.
       SHIFT(I) = -SUM(X(I,:), MASK=VALIDITY_MASK(:,I)) / RN(I) 
       WHERE (VALIDITY_MASK(:,I))
          X(I,:) = X(I,:) + SHIFT(I)
       END WHERE
       ! Reincorporate the [0,1] rescaling into the shift term.
       SHIFT(I) = SHIFT(I) * SCALAR(I) - MINS(I)
    END DO
    ! Set the unused portion of the "VECS" matrix to the identity.
    VECS(:,D+1:) = 0.0_RT
    VECS(D+1:,1:D) = 0.0_RT
    DO I = D+1, MIN(SIZE(VECS,1,KIND=INT64), SIZE(VECS,2,KIND=INT64))
       VECS(I,I) = 1.0_RT
    END DO
    ! Find the directions along which the data is most elongated.
    CALL SVD(X(:,:NMAX), VALS, VECS(1:D,1:D), STEPS=STEPS)
    ! Update the singular values associated with each vector (based on desired flatness outcome).
    IF (FLAT) THEN
       VALS(:) = VALS(:) / SQRT(RN)
    ELSE
       IF (SCALE_BY_AVERAGE) THEN
          VALS(:) = SUM(VALS(:)) / (SQRT(RN) * REAL(D,RT))  ! Average singular value.
       ELSE
          VALS(:) = VALS(1) / SQRT(RN)  ! First, max, singular value.
       END IF
    END IF
    ! Compute the inverse of the transformation if requested (BEFORE updating VECS).
    IF (PRESENT(INVERSE)) THEN
       ! Since the vectors are orthonormal, the inverse is the transpose. We also
       ! will divide the vectors by the singular values, so we invert that as well.
       DO I = 1, D
          IF (VALS(I) .GT. 0.0_RT) THEN
             INVERSE(I,1:D) = VALS(I) * VECS(1:D,I) * SCALAR(:)
          ELSE
             INVERSE(I,1:D) = VECS(1:D,I) * SCALAR(:)
          END IF
       END DO
       ! Set all elements of INVERSE that are not touched to zero.
       INVERSE(:,D+1:) = 0.0_RT
       INVERSE(D+1:,1:D) = 0.0_RT
    END IF
    ! Normalize the values associated with the singular vectors to
    !  make the output componentwise unit mean squared magnitude.
    IF (FLAT) THEN
       ! For all nonzero vectors, rescale them so that the
       !  average squared distance from zero is exactly 1.
       DO I = 1, D
          IF (VALS(I) .GT. 0.0_RT) THEN
             VECS(:,I) = VECS(:,I) / VALS(I)
          END IF
       END DO
    ! OR, simply rescale all vectors by the first singular value.
    ELSE
       IF (VALS(1) .GT. 0.0_RT) THEN
          VECS(:,:) = VECS(:,:) / VALS(1)
       END IF
    END IF
    ! Apply the scaled singular vectors to the data to normalize.
    X1(:,:) = X(:,:) 
    CALL GEMM('T', 'N', INT(D,INT32), SIZE(X,2), INT(D,INT32), 1.0_RT, &
         VECS(1:D,1:D), INT(D,INT32), &
         X1(:,:), INT(D,INT32), &
         0.0_RT, X(:,:), INT(D,INT32))
    ! Apply the exact same transformation to the vectors
    ! that was already applied to X to normalize original
    ! component scale, because these vectors should
    ! recreate the entire transformation.
    DO I = 1, D
       VECS(I,1:D) = VECS(I,1:D) / SCALAR(I)
    END DO
    ! Deallocate local memory.
    DEALLOCATE(VALIDITY_MASK, VALS, RN, MINS, SCALAR, X1)
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
    CALL SGELS( TRANS, M, N, INT(NRHS), A, LDA, B, LDB, WORK, LWORK, INFO )
    ! Store the result.
    IF (SIZE(X,2) .LE. SIZE(B,2)) THEN
       X(:,:) = B(:SIZE(X,1),:SIZE(X,2))
    ELSE
       X(:,:SIZE(B,2)) = B(:SIZE(X,1),:SIZE(B,2))
    END IF
  END SUBROUTINE LEAST_SQUARES

END MODULE MATRIX_OPERATIONS
