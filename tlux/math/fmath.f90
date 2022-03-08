
MODULE FMATH
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  IMPLICIT NONE

CONTAINS

  ! Orthogonalize and normalize column vectors of A in order.
  SUBROUTINE ORTHOGONALIZE(A, LENGTHS, RANK)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
    REAL(KIND=RT), INTENT(OUT), DIMENSION(SIZE(A,2)) :: LENGTHS
    INTEGER, INTENT(OUT), OPTIONAL :: RANK
    REAL(KIND=RT) :: L, VEC(SIZE(A,1)) 
    INTEGER :: I, J
    IF (PRESENT(RANK)) RANK = 0
    column_orthogonolization : DO I = 1, SIZE(A,2)
       LENGTHS(I:) = SUM(A(:,I:)**2, 1)
       ! Pivot the largest magnitude vector to the front.
       J = I-1+MAXLOC(LENGTHS(I:),1)
       IF (J .NE. I) THEN
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
    EXTERNAL :: SGEMM, SSYRK
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
       CALL SGEMM('N', 'N', K, K, K, 1.0_RT, &
            ATA(:,:), K, Q(:,:), K, 0.0_RT, &
            VT(:,:), K)
       CALL ORTHOGONALIZE(VT(:,:), S(:), RANK)
    END DO power_iteration
    ! Compute the singular values.
    WHERE (S(:) .NE. 0.0_RT)
       S(:) = SQRT(S(:)) / MULTIPLIER
    END WHERE
  END SUBROUTINE SVD

END MODULE FMATH
