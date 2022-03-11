! TODO:
! 
! - Center and fit inputs and outputs to spherical points (unit length).
! 
! - Enable a model that has no internal states (for linear regression).
! 
! - Enable a apositional without a following model.
! 
! - Change initialization of shift terms to be based on the assumption
!   that input data has spherical shape. Internal shift terms use
!   previous shift and weight matrices to estimate value ranges for
!   each basis function, with largest functions getting most negative
!   shift and smallest functions getting most positive shift.
! 
! - Create 'condition_model' subroutine that takes values and gradients
!   at all internal states, uses linear transformations to identify and
!   remove redundant basis functions and initialize new basis functions
!   that align most with the error function.
! 
! - Get stats on the internal values within the network during training.
!   - step size progression
!   - shift values
!   - vector magnitudes for each node
!   - output weights magnitude for each node
!   - internal node contributions to MSE
!   - data distributions at internal nodes (% less and greater than 0)
! 


! Module for matrix multiplication (absolutely crucial for APOS speed).
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
  SUBROUTINE ORTHONORMALIZE(A, RANK)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A
    INTEGER, INTENT(OUT), OPTIONAL :: RANK
    REAL(KIND=RT), DIMENSION(SIZE(A,2)) :: MULTIPLIERS
    REAL(KIND=RT) :: LEN
    INTEGER :: I, J
    IF (PRESENT(RANK)) RANK = 0
    DO I = 1, SIZE(A,2)
       LEN = NORM2(A(:,I))
       IF (LEN .GT. 0.0_RT) THEN
          A(:,I) = A(:,I) / LEN
          IF (PRESENT(RANK)) RANK = RANK + 1
       END IF
       IF ((LEN .GT. 0.0_RT) .AND. (I .LT. SIZE(A,2))) THEN
          MULTIPLIERS(I+1:) = MATMUL(A(:,I), A(:,I+1:))
          DO J = I+1, SIZE(A,2)
             A(:,J) = A(:,J) - MULTIPLIERS(J) * A(:,I)
          END DO
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

END MODULE MATRIX_OPERATIONS


! An apositional (/aggregate) and positional piecewise linear regression model.
MODULE APOS
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
  USE MATRIX_OPERATIONS, ONLY: GEMM, ORTHONORMALIZE, RANDOM_UNIT_VECTORS
  IMPLICIT NONE

  PRIVATE :: MODEL_GRADIENT

  ! Model configuration, internal sizes and fit parameters.
  TYPE, BIND(C) :: MODEL_CONFIG
     ! (Positional) model configuration.
     INTEGER :: MDI      ! model dimension of input
     INTEGER :: MDS = 32 ! model dimension of state
     INTEGER :: MNS = 8  ! model number of states
     INTEGER :: MDO      ! model dimension of output
     INTEGER :: MNE = 0  ! model number of embeddings
     INTEGER :: MDE = 0  ! model dimension of embeddings
     ! Apositional model configuration.
     INTEGER :: ADI      ! apositional dimension of input
     INTEGER :: ADS = 32 ! apositional dimension of state
     INTEGER :: ANS = 8  ! apositional number of states
     INTEGER :: ADO = 32 ! apositional dimension of output
     INTEGER :: ANE = 0  ! apositional number of embeddings
     INTEGER :: ADE = 0  ! apositional dimension of embeddings
     ! Summary numbers that are computed.
     INTEGER :: TOTAL_SIZE
     INTEGER :: NUM_VARS
     ! Index subsets of total size vector naming scheme:
     !   M___ -> model,   A___ -> apositional (/ aggregate) model
     !   _S__ -> start,   _E__ -> end
     !   __I_ -> input,   __S_ -> states, __O_ -> output, __E_ -> embedding
     !   ___V -> vectors, ___S -> shifts
     INTEGER :: MSIV, MEIV, MSIS, MEIS ! model input
     INTEGER :: MSSV, MESV, MSSS, MESS ! model states
     INTEGER :: MSOV, MEOV             ! model output
     INTEGER :: MSEV, MEEV             ! model embedding
     INTEGER :: ASIV, AEIV, ASIS, AEIS ! apositional input
     INTEGER :: ASSV, AESV, ASSS, AESS ! apositional states
     INTEGER :: ASOV, AEOV             ! apositional output
     INTEGER :: ASEV, AEEV             ! apositional embedding
     ! Index subsets for input and output shifts.
     ! M___ -> model,       A___ -> apositional (/ aggregate) model
     ! _IS_ -> input shift, _OS_ -> output shift
     ! ___S -> start,       ___E -> end
     INTEGER :: MISS, MISE, MOSS, MOSE
     INTEGER :: AISS, AISE
     ! Function parameter.
     REAL(KIND=RT) :: DISCONTINUITY = 0.0_RT
     ! Initialization related parameters.
     REAL(KIND=RT) :: INITIAL_SHIFT_RANGE = 1.0_RT
     REAL(KIND=RT) :: INITIAL_OUTPUT_SCALE = 0.1_RT
     REAL(KIND=RT) :: INITIAL_STEP = 0.001_RT
     REAL(KIND=RT) :: INITIAL_STEP_MEAN_CHANGE = 0.1_RT
     REAL(KIND=RT) :: INITIAL_STEP_CURV_CHANGE = 0.01_RT
     ! Optimization related parameters.
     REAL(KIND=RT) :: FASTER_RATE = 1.01_RT
     REAL(KIND=RT) :: SLOWER_RATE = 0.99_RT
     INTEGER       :: MIN_STEPS_TO_STABILITY = 1
     INTEGER       :: NUM_THREADS = 1
     LOGICAL       :: KEEP_BEST = .TRUE.
     LOGICAL       :: EARLY_STOP = .TRUE.
  END TYPE MODEL_CONFIG

  ! Function that is defined by OpenMP.
  INTERFACE
     FUNCTION OMP_GET_MAX_THREADS()
       INTEGER :: OMP_GET_MAX_THREADS
     END FUNCTION OMP_GET_MAX_THREADS
  END INTERFACE

CONTAINS

  ! Generate a model configuration given state parameters for the model.
  SUBROUTINE NEW_MODEL_CONFIG(MDI, MDO, MDS, MNS, MNE, MDE, &
       ADI, ADO, ADS, ANS, ANE, ADE, NUM_THREADS, CONFIG)
     ! Size related parameters.
     INTEGER, INTENT(IN) :: MDI, ADI
     INTEGER, INTENT(IN) :: MDO
     INTEGER, OPTIONAL, INTENT(IN) :: ADO
     INTEGER, OPTIONAL, INTENT(IN) :: MDS, ADS
     INTEGER, OPTIONAL, INTENT(IN) :: MNS, ANS
     INTEGER, OPTIONAL, INTENT(IN) :: MNE, ANE
     INTEGER, OPTIONAL, INTENT(IN) :: MDE, ADE
     INTEGER, OPTIONAL, INTENT(IN) :: NUM_THREADS
     ! Output
     TYPE(MODEL_CONFIG), INTENT(OUT) :: CONFIG
     ! ---------------------------------------------------------------
     ! MDS
     IF (PRESENT(MDS)) CONFIG%MDS = MDS
     ! MNS
     IF (PRESENT(MNS)) CONFIG%MNS = MNS
     ! MNE
     IF (PRESENT(MNE)) CONFIG%MNE = MNE
     ! MDE
     IF (PRESENT(MDE)) THEN
        CONFIG%MDE = MDE
     ELSE IF (CONFIG%MNE .GT. 0) THEN
        ! Compute a reasonable default dimension (tied to volume of space).
        CONFIG%MDE = MAX(1, 1 + CEILING(LOG(REAL(CONFIG%MNE,RT)) / LOG(2.0_RT)))
        IF (CONFIG%MNE .GT. 2) CONFIG%MDE = CONFIG%MDE + 1
     END IF
     ! ---------------------------------------------------------------
     ! ADO
     IF (PRESENT(ADO)) THEN
        CONFIG%ADO = ADO
     ELSE IF (ADI .EQ. 0) THEN
        CONFIG%ADO = 0
     END IF
     ! ADS
     IF (PRESENT(ADS)) THEN
        CONFIG%ADS = ADS
     ELSE IF (ADI .EQ. 0) THEN
        CONFIG%ADS = 0
     END IF
     ! ANS
     IF (PRESENT(ANS)) THEN
        CONFIG%ANS = ANS
     ELSE IF (ADI .EQ. 0) THEN
        CONFIG%ANS = 0
     END IF
     ! ANE
     IF (PRESENT(ANE)) CONFIG%ANE = ANE
     ! ADE
     IF (PRESENT(ADE)) THEN
        CONFIG%ADE = ADE
     ELSE IF (CONFIG%ANE .GT. 0) THEN
        ! Compute a reasonable default dimension (tied to volume of space).
        CONFIG%ADE = MAX(1, 1 + CEILING(LOG(REAL(CONFIG%ANE,RT)) / LOG(2.0_RT)))
        IF (CONFIG%ANE .GT. 2) CONFIG%ADE = CONFIG%ADE + 1
     END IF
     ! ---------------------------------------------------------------
     ! NUM_THREADS
     IF (PRESENT(NUM_THREADS)) THEN
        CONFIG%NUM_THREADS = NUM_THREADS
     ELSE
        CONFIG%NUM_THREADS = OMP_GET_MAX_THREADS()
     END IF
     ! Declare all required configurations.
     CONFIG%ADI = ADI + CONFIG%ADE
     CONFIG%MDI = MDI + CONFIG%MDE + CONFIG%ADO
     CONFIG%MDO = MDO
     ! Compute indices related to the parameter vector for this model.
     CONFIG%TOTAL_SIZE = 0
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
     CONFIG%MESV = CONFIG%MSSV-1  +  CONFIG%MDS * CONFIG%MDS * (CONFIG%MNS-1)
     CONFIG%TOTAL_SIZE = CONFIG%MESV
     !   model state shift
     CONFIG%MSSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MESS = CONFIG%MSSS-1  +  CONFIG%MDS * (CONFIG%MNS-1)
     CONFIG%TOTAL_SIZE = CONFIG%MESS
     !   model output vecs
     CONFIG%MSOV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MEOV = CONFIG%MSOV-1  +  CONFIG%MDS * CONFIG%MDO
     CONFIG%TOTAL_SIZE = CONFIG%MEOV
     !   model embedding vecs
     CONFIG%MSEV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MEEV = CONFIG%MSEV-1  +  CONFIG%MDE * CONFIG%MNE
     CONFIG%TOTAL_SIZE = CONFIG%MEEV
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
     CONFIG%AESV = CONFIG%ASSV-1  +  CONFIG%ADS * CONFIG%ADS * (CONFIG%ANS-1)
     CONFIG%TOTAL_SIZE = CONFIG%AESV
     !   apositional state shift
     CONFIG%ASSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AESS = CONFIG%ASSS-1  +  CONFIG%ADS * (CONFIG%ANS-1)
     CONFIG%TOTAL_SIZE = CONFIG%AESS
     !   apositional output vecs
     CONFIG%ASOV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEOV = CONFIG%ASOV-1  +  CONFIG%ADS * CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AEOV
     !   apositional embedding vecs
     CONFIG%ASEV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEEV = CONFIG%ASEV-1  +  CONFIG%ADE * CONFIG%ANE
     CONFIG%TOTAL_SIZE = CONFIG%AEEV
     ! ---------------------------------------------------------------
     !   number of variables
     CONFIG%NUM_VARS = CONFIG%TOTAL_SIZE
     ! ---------------------------------------------------------------
     !   model input shift
     CONFIG%MISS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MISE = CONFIG%MISS-1 + CONFIG%MDI-CONFIG%ADO-CONFIG%MDE
     CONFIG%TOTAL_SIZE = CONFIG%MISE
     !   model output shift
     CONFIG%MOSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MOSE = CONFIG%MOSS-1 + CONFIG%MDO
     CONFIG%TOTAL_SIZE = CONFIG%MOSE
     !   apositional input shift
     CONFIG%AISS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AISE = CONFIG%AISS-1 + CONFIG%ADI-CONFIG%ADE
     CONFIG%TOTAL_SIZE = CONFIG%AISE
  END SUBROUTINE NEW_MODEL_CONFIG


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
         CONFIG%MDO, CONFIG%MDE, CONFIG%MNE, &
         MODEL(CONFIG%MSIV:CONFIG%MEIV), &
         MODEL(CONFIG%MSIS:CONFIG%MEIS), &
         MODEL(CONFIG%MSSV:CONFIG%MESV), &
         MODEL(CONFIG%MSSS:CONFIG%MESS), &
         MODEL(CONFIG%MSOV:CONFIG%MEOV), &
         MODEL(CONFIG%MSEV:CONFIG%MEEV))
    ! Initialize the apositional model.
    CALL UNPACKED_INIT_MODEL(&
         CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, &
         CONFIG%ADO, CONFIG%ADE, CONFIG%ANE, &
         MODEL(CONFIG%ASIV:CONFIG%AEIV), &
         MODEL(CONFIG%ASIS:CONFIG%AEIS), &
         MODEL(CONFIG%ASSV:CONFIG%AESV), &
         MODEL(CONFIG%ASSS:CONFIG%AESS), &
         MODEL(CONFIG%ASOV:CONFIG%AEOV), &
         MODEL(CONFIG%ASEV:CONFIG%AEEV))
    ! Reset the output scale for the apositional model to be neutral.
    MODEL(CONFIG%ASOV:CONFIG%AEOV) = MODEL(CONFIG%ASOV:CONFIG%AEOV) &
         / CONFIG%INITIAL_OUTPUT_SCALE

  CONTAINS
    ! Initialize the model after unpacking it into its constituent parts.
    SUBROUTINE UNPACKED_INIT_MODEL(MDI, MDS, MNS, MDO, MDE, MNE, &
         INPUT_VECS, INPUT_SHIFT, STATE_VECS, STATE_SHIFT, &
         OUTPUT_VECS, EMBEDDINGS)
      INTEGER, INTENT(IN) :: MDI, MDS, MNS, MDO, MDE, MNE
      REAL(KIND=RT), DIMENSION(MDI, MDS) :: INPUT_VECS
      REAL(KIND=RT), DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), DIMENSION(MDS, MDS, MNS-1) :: STATE_VECS
      REAL(KIND=RT), DIMENSION(MDS, MNS-1) :: STATE_SHIFT
      REAL(KIND=RT), DIMENSION(MDS, MDO) :: OUTPUT_VECS
      REAL(KIND=RT), DIMENSION(MDE, MNE) :: EMBEDDINGS
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
         INPUT_SHIFT(I) = 2.0_RT * CONFIG%INITIAL_SHIFT_RANGE * &    ! 2 * shift *
              (REAL(I-1,RT) / MAX(1.0_RT, REAL(MDS-1, RT))) & ! range [0, 1]
              - CONFIG%INITIAL_SHIFT_RANGE                           ! - shift
         STATE_SHIFT(I,:) = INPUT_SHIFT(I) ! range [-shift, shift]
      END DO
    END SUBROUTINE UNPACKED_INIT_MODEL
  END SUBROUTINE INIT_MODEL


  ! Returnn nonzero INFO if any shapes or values do not match expectations.
  SUBROUTINE CHECK_SHAPE(CONFIG, MODEL, Y, X, XI, AX, AXI, SIZES, INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN), DIMENSION(:) :: SIZES
    INTEGER,       INTENT(OUT) :: INFO
    INFO = 0
    ! Compute whether the shape matches the CONFIG.
    IF (SIZE(MODEL) .NE. CONFIG%TOTAL_SIZE) THEN
       INFO = 1 ! Model size does not match model configuration.
    ELSE IF (SIZE(X,2) .NE. SIZE(Y,2)) THEN
       INFO = 2 ! Input arrays do not match in size.
    ELSE IF (SIZE(X,1) + CONFIG%MDE + CONFIG%ADO .NE. CONFIG%MDI) THEN
       INFO = 3 ! X input dimension is bad.
    ELSE IF (SIZE(Y,1) .NE. CONFIG%MDO) THEN
       INFO = 4 ! Output dimension is bad.
    ELSE IF ((CONFIG%MNE .GT. 0) .AND. (SIZE(XI,2) .NE. SIZE(X,2))) THEN
       INFO = 5 ! Input integer X size does not match.
    ELSE IF ((MINVAL(XI) .LT. 0) .OR. (MAXVAL(XI) .GT. CONFIG%MNE)) THEN
       INFO = 6 ! Input integer X index out of range.
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (SIZE(X,2) .NE. SIZE(SIZES))) THEN
       INFO = 7 ! X and SIZES do not match.
    ELSE IF (SIZE(AX,2) .NE. SUM(SIZES)) THEN
       INFO = 8 ! AX and SUM(SIZES) do not match.
    ELSE IF (SIZE(AX,1) + CONFIG%ADE .NE. CONFIG%ADI) THEN
       INFO = 9 ! AX input dimension is bad.
    ELSE IF (SIZE(AXI,2) .NE. SIZE(AX,2)) THEN
       INFO = 10 ! Input integer AX size does not match.
    ELSE IF ((MINVAL(AXI) .LT. 0) .OR. (MAXVAL(AXI) .GT. CONFIG%ANE)) THEN
       INFO = 11 ! Input integer AX index out of range.
    END IF
  END SUBROUTINE CHECK_SHAPE

 
  ! Given a model and mixed real and integer inputs, embed the inputs
  !  into their appropriate real-value-only formats.
  SUBROUTINE EMBED(CONFIG, MODEL, X, XI, AX, AXI, XXI, AXXI)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN),  DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),  DIMENSION(:,:) :: AXI
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: XXI  ! MDI, SIZE(X,2)
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: AXXI ! ADI, SIZE(AX,2)
    ! Add the real inputs to the front of each vector before apositional outputs.
    XXI(1:SIZE(X,1),:) = X(:,:)
    ! If there is XInteger input, unpack it into end of XXI.
    CALL UNPACK_EMBEDDINGS(CONFIG%MDE, CONFIG%MNE, &
         MODEL(CONFIG%MSEV:CONFIG%MEEV), &
         XI(:,:), XXI(SIZE(X,1)+1:SIZE(X,1)+CONFIG%MDE,:))
    ! Add the real inputs to the front of each vector.
    AXXI(1:SIZE(AX,1),:) = AX(:,:)
    ! If there is AXInteger input, unpack it into XXI.
    CALL UNPACK_EMBEDDINGS(CONFIG%ADE, CONFIG%ANE, &
         MODEL(CONFIG%ASEV:CONFIG%AEEV), &
         AXI(:,:), AXXI(SIZE(AX,1)+1:,:))
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


  ! Given a number of batches, compute the batch start and ends for
  !  the apositional and positional inputs. Store in (2,_) arrays.
  SUBROUTINE COMPUTE_BATCHES(NUM_BATCHES, X, AX, SIZES, BATCHA, BATCHM, INFO)
    INTEGER,       INTENT(IN) :: NUM_BATCHES
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),  DIMENSION(:)   :: SIZES
    INTEGER,       INTENT(OUT), DIMENSION(:,:) :: BATCHA, BATCHM
    INTEGER,       INTENT(INOUT) :: INFO
    ! Local variables.
    INTEGER :: BATCH, BE, BN, BS, I, MN, AN
    ! Compute sizes of inputs.
    IF ((SIZE(SIZES) .GT. 0) .OR. (SIZE(AX,2) .GT. 0)) THEN
       MN = SIZE(SIZES)
    ELSE
       MN = SIZE(X,2)
    END IF
    AN = SUM(SIZES(:))
    ! Check for errors.
    IF (SIZE(X,2) .NE. MN) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Sizes of X and SIZES do not match.', SIZE(X,2), SIZE(SIZES)
       INFO = -1
    ELSE IF (SIZE(AX,2) .NE. AN) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Size of AX and sum of SIZES do not match.', SIZE(AX,2), AN
       INFO = -2
    ELSE IF (NUM_BATCHES .GT. MN) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Requested number of batches is too large.', NUM_BATCHES, MN, AN
       INFO = -3
    ELSE IF (NUM_BATCHES .NE. SIZE(BATCHA,2)) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Number of batches does not match BATCHA.', NUM_BATCHES, SIZE(BATCHA,2)
       INFO = -4
    ELSE IF (NUM_BATCHES .NE. SIZE(BATCHM,2)) THEN
       PRINT *, 'ERROR (COMPUTE_BATCHES): Number of batches does not match BATCHM.', NUM_BATCHES, SIZE(BATCHM,2)
       INFO = -5
    END IF
    IF (INFO .NE. 0) RETURN
    ! Construct batches for data sets with apositional inputs.
    IF (AN .GT. 0) THEN
       IF (NUM_BATCHES .EQ. 1) THEN
          BATCHA(1,1) = 1
          BATCHA(2,1) = AN
          BATCHM(1,1) = 1
          BATCHM(2,1) = MN
       ELSE
          BN = (AN + NUM_BATCHES - 1) / NUM_BATCHES ! = CEIL(AN / NUM_BATCHES)
          ! Compute how many X points are associated with each Y.
          BS = 1
          BE = SIZES(1)
          BATCH = 1
          BATCHM(1,BATCH) = 1
          DO I = 2, MN
             ! If a fair share of the points have been aggregated, OR
             !   there are only as many sets left as there are batches.
             IF ((BE-BS .GT. BN) .OR. (1+MN-I .LE. (NUM_BATCHES-BATCH))) THEN
                BATCHM(2,BATCH) = I-1
                BATCHA(1,BATCH) = BS
                BATCHA(2,BATCH) = BE
                BATCH = BATCH+1
                BATCHM(1,BATCH) = I
                BS = BE+1
                BE = BS - 1
             END IF
             BE = BE + SIZES(I)
          END DO
          BATCHM(2,BATCH) = MN
          BATCHA(1,BATCH) = BS
          BATCHA(2,BATCH) = BE
       END IF
    ! Construct batches for data sets that only have positional inputs.
    ELSE
       BN = (MN + NUM_BATCHES - 1) / NUM_BATCHES ! = CEIL(MN / NUM_BATCHES)
       DO BATCH = 1, NUM_BATCHES
          BATCHM(1,BATCH) = BN*(BATCH-1) + 1
          BATCHM(2,BATCH) = MIN(MN, BN*BATCH)
       END DO
       BATCHA(1,:) = 0
       BATCHA(2,:) = -1
    END IF
  END SUBROUTINE COMPUTE_BATCHES


  ! Evaluate the piecewise linear regression model, assume already-embedded inputs.
  SUBROUTINE EVALUATE(CONFIG, MODEL, Y, X, AX, SIZES, M_STATES, &
       A_STATES, AY, INFO, SHIFT, THREADS)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:)   :: SIZES
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:,:) :: M_STATES ! SIZE(X, 2), MDS, (MNS|2)
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:,:) :: A_STATES ! SIZE(AX,2), ADS, (ANS|2)
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:) :: AY ! SIZE(AX,2), ADO
    INTEGER, INTENT(INOUT) :: INFO
    LOGICAL, INTENT(IN), OPTIONAL :: SHIFT
    INTEGER, INTENT(IN), OPTIONAL :: THREADS
    ! Internal values.
    LOGICAL :: APPLY_SHIFT
    INTEGER :: I, BATCH, NB, BN, BS, BE, BT, GS, GE, NT
    INTEGER, DIMENSION(:,:), ALLOCATABLE :: BATCHA, BATCHM
    ! Set default for shifting data.
    IF (PRESENT(SHIFT)) THEN
       APPLY_SHIFT = SHIFT
    ELSE
       APPLY_SHIFT = .TRUE.
    END IF
    ! Set default for the number of threads.
    IF (PRESENT(THREADS)) THEN
       NT = THREADS
    ELSE
       NT = CONFIG%NUM_THREADS
    END IF
    ! Set up batching for parallelization.
    NB = MIN(SIZE(Y,2), NT)
    ALLOCATE(BATCHA(2,NB), BATCHM(2,NB))
    ! Compute the batch start and end indices.
    CALL COMPUTE_BATCHES(NB, X, AX, SIZES, BATCHA, BATCHM, INFO)
    IF (INFO .NE. 0) RETURN
    !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NB > 1)
    batch_evaluation : DO BATCH = 1, NB
       ! If there is an apositional model, apply it.
       IF (CONFIG%ADI .GT. 0) THEN
          BS = BATCHA(1,BATCH)
          BE = BATCHA(2,BATCH)
          BT = BE-BS+1
          IF (BT .EQ. 0) CYCLE batch_evaluation
          ! Apply shift terms to apositional inputs.
          IF (APPLY_SHIFT) THEN
             GE = CONFIG%ADI-CONFIG%ADE
             IF (GE .GT. 0) THEN
                DO I = BS, BE
                   AX(:GE,I) = AX(:GE,I) + MODEL(CONFIG%AISS:CONFIG%AISE)
                END DO
             END IF
          END IF
          ! Evaluate the apositional model.
          CALL UNPACKED_EVALUATE(BT, &
               CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADO, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASIS:CONFIG%AEIS), &
               MODEL(CONFIG%ASSV:CONFIG%AESV), &
               MODEL(CONFIG%ASSS:CONFIG%AESS), &
               MODEL(CONFIG%ASOV:CONFIG%AEOV), &
               AX(:,BS:BE), AY(BS:BE,:), A_STATES(BS:BE,:,:), YTRANS=.TRUE.)
          ! Aggregate the outputs from the apositional model.
          BT = SIZE(X,1)-CONFIG%ADO+1 ! <- reuse this variable name
          GS = BS
          DO I = BATCHM(1,BATCH), BATCHM(2,BATCH)
             GE = GS + SIZES(I) - 1
             X(BT:,I) = SUM(AY(GS:GE,:), 1) / REAL(SIZES(I),RT) 
             GS = GE + 1
          END DO
          BS = BATCHM(1,BATCH)
          BE = BATCHM(2,BATCH)
          BT = BE-BS+1
       ELSE
          BS = BATCHM(1,BATCH)
          BE = BATCHM(2,BATCH)
          BT = BE-BS+1
          IF (BT .EQ. 0) CYCLE batch_evaluation
       END IF
       ! Apply shift terms to model inputs.
       IF (APPLY_SHIFT) THEN
          GE = CONFIG%MDI-CONFIG%MDE-CONFIG%ADO
          IF (GE .GT. 0) THEN
             DO I = BS, BE
                X(:GE,I) = X(:GE,I) + MODEL(CONFIG%MISS:CONFIG%MISE)
             END DO
          END IF
       END IF
       ! Run the positional model.
       CALL UNPACKED_EVALUATE(BT, &
            CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDO, &
            MODEL(CONFIG%MSIV:CONFIG%MEIV), &
            MODEL(CONFIG%MSIS:CONFIG%MEIS), &
            MODEL(CONFIG%MSSV:CONFIG%MESV), &
            MODEL(CONFIG%MSSS:CONFIG%MESS), &
            MODEL(CONFIG%MSOV:CONFIG%MEOV), &
            X(:,BS:BE), Y(:,BS:BE), M_STATES(BS:BE,:,:))
       ! Apply shift terms to model outputs.
       IF (APPLY_SHIFT) THEN
          DO I = BS, BE
             Y(:,I) = Y(:,I) + MODEL(CONFIG%MOSS:CONFIG%MOSE)
          END DO
       END IF
    END DO batch_evaluation
    !$OMP END PARALLEL DO
    DEALLOCATE(BATCHA, BATCHM)

  CONTAINS

    SUBROUTINE UNPACKED_EVALUATE(N, MDI, MDS, MNS, MDO, INPUT_VECS, &
         INPUT_SHIFT, STATE_VECS, STATE_SHIFT, OUTPUT_VECS, X, Y, &
         STATES, YTRANS)
      INTEGER, INTENT(IN) :: N, MDI, MDS, MNS, MDO
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDI, MDS) :: INPUT_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MDS, MNS-1) :: STATE_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MNS-1) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MDO) :: OUTPUT_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(:,:) :: X
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: Y
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:,:) :: STATES
      LOGICAL, INTENT(IN), OPTIONAL :: YTRANS
      ! Local variables to evaluating a single batch.
      INTEGER :: D, L, S1, S2, S3
      LOGICAL :: REUSE_STATES, YT
      IF (PRESENT(YTRANS)) THEN
         YT = YTRANS
      ELSE
         YT = .FALSE.
      END IF
      REUSE_STATES = (SIZE(STATES,3) .LT. MNS)
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
         IF (REUSE_STATES) THEN
            S1 = 1 ; S2 = 2   ; S3 = 1
         ELSE
            S1 = L ; S2 = L+1 ; S3 = L+1
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
      IF (REUSE_STATES) THEN
         S3 = 1
      ELSE
         S3 = MNS
      END IF
      ! Return the final output (default to assuming Y is contiguous
      !   by component unless PRESENT(YTRANS) and YTRANS = .TRUE.
      !   then assume it is contiguous by individual sample).
      IF (YT) THEN
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
    END SUBROUTINE UNPACKED_EVALUATE
  END SUBROUTINE EVALUATE


  ! Given the values at all internal states in the model and an output gradient,
  !  propogate the output gradient through the model and return the gradient
  !  of all paprameters.
  SUBROUTINE BASIS_GRADIENT(CONFIG, MODEL, Y, X, AX, SIZES, &
       M_STATES, A_STATES, AY, GRAD)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT),  DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(INOUT),  DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),  DIMENSION(:)   :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: M_STATES ! SIZE(X, 2), MDS, MNS+1
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_STATES ! SIZE(AX,2), ADS, ANS+1
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY ! SIZE(AX,2), ADO
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: GRAD ! model gradient
    ! Set the dimension of the X gradient that should be calculated.
    INTEGER :: I, J, GS, GE, XDG
    IF (CONFIG%ADI .GT. 0) THEN
       XDG = CONFIG%ADO + CONFIG%MDE
    ELSE
       XDG = CONFIG%MDE
    END IF
    ! Do the backward gradient calculation assuming "Y" contains output gradient.
    CALL UNPACKED_BASIS_GRADIENT( Y(:,:), M_STATES(:,:,:), X(:,:), &
         CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDO, XDG, &
         MODEL(CONFIG%MSIV:CONFIG%MEIV), &
         MODEL(CONFIG%MSIS:CONFIG%MEIS), &
         MODEL(CONFIG%MSSV:CONFIG%MESV), &
         MODEL(CONFIG%MSSS:CONFIG%MESS), &
         MODEL(CONFIG%MSOV:CONFIG%MEOV), &
         GRAD(CONFIG%MSIV:CONFIG%MEIV), &
         GRAD(CONFIG%MSIS:CONFIG%MEIS), &
         GRAD(CONFIG%MSSV:CONFIG%MESV), &
         GRAD(CONFIG%MSSS:CONFIG%MESS), &
         GRAD(CONFIG%MSOV:CONFIG%MEOV))
    ! Propogate the gradient form X into the aggregate outputs.
    IF (CONFIG%ADI .GT. 0) THEN
       XDG = SIZE(X,1) - CONFIG%ADO + 1
       GS = 1
       DO I = 1, SIZE(SIZES)
          GE = GS + SIZES(I) - 1
          DO J = GS, GE
             AY(J,:) = X(XDG:,I) / REAL(SIZES(I),RT)
          END DO
          GS = GE + 1
       END DO
       ! Do the backward gradient calculation assuming "AY" contains output gradient.
       CALL UNPACKED_BASIS_GRADIENT( AY(:,:), A_STATES(:,:,:), AX(:,:), &
            CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADO, CONFIG%ADE, &
            MODEL(CONFIG%ASIV:CONFIG%AEIV), &
            MODEL(CONFIG%ASIS:CONFIG%AEIS), &
            MODEL(CONFIG%ASSV:CONFIG%AESV), &
            MODEL(CONFIG%ASSS:CONFIG%AESS), &
            MODEL(CONFIG%ASOV:CONFIG%AEOV), &
            GRAD(CONFIG%ASIV:CONFIG%AEIV), &
            GRAD(CONFIG%ASIS:CONFIG%AEIS), &
            GRAD(CONFIG%ASSV:CONFIG%AESV), &
            GRAD(CONFIG%ASSS:CONFIG%AESS), &
            GRAD(CONFIG%ASOV:CONFIG%AEOV), YTRANS=.TRUE.)
    END IF

  CONTAINS
    ! Compute the model gradient.
    SUBROUTINE UNPACKED_BASIS_GRADIENT( Y, STATES, X, &
         MDI, MDS, MNS, MDO, MDE, &
         INPUT_VECS, INPUT_SHIFT, &
         STATE_VECS, STATE_SHIFT, OUTPUT_VECS, &
         INPUT_VECS_GRADIENT, INPUT_SHIFT_GRADIENT, &
         STATE_VECS_GRADIENT, STATE_SHIFT_GRADIENT, &
         OUTPUT_VECS_GRADIENT, YTRANS )
      REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: STATES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
      INTEGER, INTENT(IN) :: MDI, MDS, MNS, MDO, MDE
      ! Model variables.
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDI,MDS) :: INPUT_VECS
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS,MDS,MNS-1) :: STATE_VECS
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS,MNS-1) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS,MDO) :: OUTPUT_VECS
      ! Model variable gradients.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI,MDS) :: INPUT_VECS_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS) :: INPUT_SHIFT_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MDS,MNS-1) :: STATE_VECS_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MNS-1) :: STATE_SHIFT_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MDO) :: OUTPUT_VECS_GRADIENT
      LOGICAL, INTENT(IN), OPTIONAL :: YTRANS
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
      IF (PRESENT(YTRANS)) THEN
         IF (YTRANS) THEN
            YT = 'N'
         ELSE
            YT = 'T'
         END IF
      ELSE
         YT = 'T'
      END IF
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
  SUBROUTINE MODEL_GRADIENT(CONFIG, MODEL, Y, X, XI, AX, AXI, SIZES, &
       SUM_SQUARED_GRADIENT, MODEL_GRAD, ERROR_GRADIENT, INFO, SHIFT, THREADS)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN), DIMENSION(:)   :: SIZES
    ! Sum (over all data) squared error (summed over dimensions).
    REAL(KIND=RT), INTENT(INOUT) :: SUM_SQUARED_GRADIENT
    ! Gradient of the model parameters.
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: MODEL_GRAD
    ! Interface to subroutine that computes the error gradient given outputs and targets.
    INTERFACE
       SUBROUTINE ERROR_GRADIENT(TARGETS, OUTPUTS)
         USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
         REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: TARGETS
         REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: OUTPUTS
       END SUBROUTINE ERROR_GRADIENT
    END INTERFACE
    INTEGER, INTENT(INOUT) :: INFO
    LOGICAL, INTENT(IN), OPTIONAL :: SHIFT
    INTEGER, INTENT(IN), OPTIONAL :: THREADS
    ! Local allocations for computing gradient.
    REAL(KIND=RT), DIMENSION(CONFIG%MDI,SIZE(X,2)) :: XXI
    REAL(KIND=RT), DIMENSION(CONFIG%ADI,SIZE(AX,2)) :: AXXI
    REAL(KIND=RT), DIMENSION(CONFIG%MDO,SIZE(Y,2)) :: Y_GRADIENT
    REAL(KIND=RT), DIMENSION(SIZE(X,2),CONFIG%MDS,CONFIG%MNS+1) :: M_STATES
    REAL(KIND=RT), DIMENSION(SIZE(AX,2),CONFIG%ADS,CONFIG%ANS+1) :: A_STATES
    REAL(KIND=RT), DIMENSION(SIZE(AX,2),CONFIG%ADO) :: AY
    ! Embed all integer inputs into real matrix inputs.
    CALL EMBED(CONFIG, MODEL, X, XI, AX, AXI, XXI, AXXI)
    ! Evaluate the model, storing internal states (for gradient calculation).
    CALL EVALUATE(CONFIG, MODEL, Y_GRADIENT, XXI, AXXI, SIZES, &
         M_STATES, A_STATES, AY, INFO, SHIFT=SHIFT, THREADS=THREADS)
    ! Compute the gradient of the model outputs, overwriting "Y_GRADIENT"
    CALL ERROR_GRADIENT(Y, Y_GRADIENT)
    SUM_SQUARED_GRADIENT = SUM_SQUARED_GRADIENT + SUM(Y_GRADIENT(:,:)**2)
    ! Compute the gradient with respect to the model basis functions.
    CALL BASIS_GRADIENT(CONFIG, MODEL, Y_GRADIENT, XXI, AXXI, &
         SIZES, M_STATES, A_STATES, AY, MODEL_GRAD)
    ! Convert the computed input gradients into average gradients for each embedding.
    IF (SIZE(XI,1) .GT. 0) THEN
       CALL EMBEDDING_GRADIENT(CONFIG%MDE, CONFIG%MNE, &
            XI, XXI(CONFIG%ADO+SIZE(X,1)+1:,:), &
            MODEL_GRAD(CONFIG%MSEV:CONFIG%MEEV))
    END IF
    ! Convert the computed input gradients into average gradients for each embedding.
    IF (SIZE(AXI,1) .GT. 0) THEN
       CALL EMBEDDING_GRADIENT(CONFIG%ADE, CONFIG%ANE, &
            AXI, AXXI(SIZE(AX,1)+1:,:), &
            MODEL_GRAD(CONFIG%ASEV:CONFIG%AEEV))
    END IF
  END SUBROUTINE MODEL_GRADIENT


  ! Compute the sum of squared error, store the gradient in the OUTPUTS.
  !   TARGETS - row vectors containing target values
  !   OUTPUTS - column vectors containing model predictions
  SUBROUTINE SQUARED_ERROR_GRADIENT(TARGETS, OUTPUTS)
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: TARGETS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: OUTPUTS
    INTEGER :: D
    OUTPUTS(:,:) = OUTPUTS(:,:) - TARGETS(:,:)
  END SUBROUTINE SQUARED_ERROR_GRADIENT


  ! Produce the true values as the gradient (which will show large
  !  magnitudes for parameters in the model that align with values).
  SUBROUTINE TRUE_VALUE_GRADIENT(TARGETS, OUTPUTS)
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: TARGETS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: OUTPUTS
    INTEGER :: D
    OUTPUTS(:,:) = TARGETS(:,:)
  END SUBROUTINE TRUE_VALUE_GRADIENT


  ! Fit input / output pairs by minimizing mean squared error.
  SUBROUTINE MINIMIZE_MSE(CONFIG, MODEL, Y, X, XI, AX, AXI, SIZES, &
       STEPS, SUM_SQUARED_ERROR, RECORD, INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN), DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN), DIMENSION(:) :: SIZES
    INTEGER,       INTENT(IN) :: STEPS
    REAL(KIND=RT), INTENT(OUT) :: SUM_SQUARED_ERROR
    REAL(KIND=RT), INTENT(OUT), DIMENSION(4,STEPS), OPTIONAL :: RECORD
    INTEGER,       INTENT(OUT) :: INFO
    !  Local variables.
    !    gradient step arrays, 4 copies of model + (num threads - 1)
    REAL(KIND=RT), DIMENSION(CONFIG%NUM_VARS) :: &
         MODEL_GRAD, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, BEST_MODEL
    REAL(KIND=RT), DIMENSION(CONFIG%MDI-CONFIG%MDE-CONFIG%ADO) :: X_SCALE
    REAL(KIND=RT), DIMENSION(CONFIG%ADI-CONFIG%ADE) :: AX_SCALE
    REAL(KIND=RT), DIMENSION(CONFIG%MDO) :: Y_SCALE
    !    batch start and end indices for parallelization
    INTEGER, DIMENSION(2,CONFIG%NUM_THREADS) :: BATCHA, BATCHM
    !    "backspace" character array for printing to the same line repeatedly
    CHARACTER(LEN=*), PARAMETER :: RESET_LINE = REPEAT(CHAR(8),25)
    !    singletons
    LOGICAL :: REVERT_TO_BEST, DID_PRINT
    INTEGER :: BN, I, NB, NS, NY, BATCH, SS, SE
    INTEGER :: UPDATE_INDICES(CONFIG%NUM_VARS), NUM_TO_UPDATE
    REAL(KIND=RT) :: RNY, BATCHES, PREV_MSE, MSE, BEST_MSE
    REAL(KIND=RT) :: STEP_FACTOR, STEP_MEAN_CHANGE, STEP_MEAN_REMAIN, &
         STEP_CURV_CHANGE, STEP_CURV_REMAIN
    REAL :: LAST_PRINT_TIME, CURRENT_TIME, WAIT_TIME
    ! Check for a valid data shape given the model.
    INFO = 0
    CALL CHECK_SHAPE(CONFIG, MODEL, Y, X, XI, AX, AXI, SIZES, INFO)
    IF (INFO .NE. 0) RETURN
    ! Number of points.
    NY = SIZE(Y,2)
    RNY = REAL(NY, RT)
    ! Set the step factor.
    STEP_FACTOR = CONFIG%INITIAL_STEP
    ! Set the initial "number of steps taken since best" counter.
    NS = 0
    ! Set the batch N (BN) and num batches (NB).
    NB = MIN(CONFIG%NUM_THREADS, NY)
    BN = (NY + NB - 1) / NB
    CALL COMPUTE_BATCHES(NB, X, AX, SIZES, BATCHA, BATCHM, INFO)
    IF (INFO .NE. 0) THEN
       Y(:,:) = 0.0_RT
       RETURN
    END IF
    ! Only "revert" to the best model seen if some steps are taken.
    REVERT_TO_BEST = CONFIG%KEEP_BEST .AND. (STEPS .GT. 0)
    ! Store the start time of this routine (to make sure updates can
    !    be shown to the user at a reasonable frequency).
    CALL CPU_TIME(LAST_PRINT_TIME)
    DID_PRINT = .FALSE.
    WAIT_TIME = 2.0 * NB ! 2 seconds (times number of threads)
    ! Set the initial number of variables to update at the whole model.
    NUM_TO_UPDATE = CONFIG%NUM_VARS
    ! Initial rates of change of mean and variance values.
    STEP_MEAN_CHANGE = CONFIG%INITIAL_STEP_MEAN_CHANGE
    STEP_MEAN_REMAIN = 1.0_RT - STEP_MEAN_CHANGE
    STEP_CURV_CHANGE = CONFIG%INITIAL_STEP_CURV_CHANGE
    STEP_CURV_REMAIN = 1.0_RT - STEP_CURV_CHANGE
    ! Initial mean squared error is "max representable value".
    PREV_MSE = HUGE(PREV_MSE)
    BEST_MSE = HUGE(BEST_MSE)
    ! Set the average step sizes.
    MODEL_GRAD_MEAN(:) = 0.0_RT
    ! Set the estiamted curvature in steps.
    MODEL_GRAD_CURV(:) = 0.0_RT
    ! Set the default size start and end indices for when it is absent.
    IF (SIZE(SIZES) .EQ. 0) THEN
       SS = 1
       SE = -1
    END IF
    ! Set shift terms to center all inputs and outputs.
    MODEL(CONFIG%MISS:CONFIG%MISE) = -SUM(X(:,:),2) / RNY
    MODEL(CONFIG%AISS:CONFIG%AISE) = -SUM(AX(:,:),2) / RNY
    MODEL(CONFIG%MOSS:CONFIG%MOSE) =  SUM(Y(:,:),2) / RNY
    IF (SIZE(X,1) .GT. 0) THEN
       DO I = 1, SIZE(X,2)
          X(:,I) = X(:,I) + MODEL(CONFIG%MISS:CONFIG%MISE)
       END DO
    END IF
    IF (SIZE(AX,1) .GT. 0) THEN
       DO I = 1, SIZE(AX,2)
          AX(:,I) = AX(:,I) + MODEL(CONFIG%AISS:CONFIG%AISE)
       END DO
    END IF
    DO I = 1, SIZE(Y,2)
       Y(:,I) = Y(:,I) - MODEL(CONFIG%MOSS:CONFIG%MOSE)
    END DO
    ! Make inputs and outputs componentwise unit deviation.
    !   X
    X_SCALE(:)  = NORM2(X(:,:),2) / SQRT(RNY)
    WHERE ( X_SCALE(:) .LT. EPSILON(1.0_RT))  X_SCALE(:) = 1.0_RT
    IF (SIZE(X,1) .GT. 0) THEN
       DO I = 1, SIZE(X,2)
          X(:,I) = X(:,I) / X_SCALE(:)
       END DO
    END IF
    !   AX
    AX_SCALE(:) = NORM2(AX(:,:),2)
    IF (SIZE(AX,2) .GT. 0) AX_SCALE(:) = AX_SCALE(:) / SQRT(REAL(SIZE(AX,2),RT))
    WHERE (AX_SCALE(:) .LT. EPSILON(1.0_RT)) AX_SCALE(:) = 1.0_RT
    IF (SIZE(AX,1) .GT. 0) THEN
       DO I = 1, SIZE(AX,2)
          AX(:,I) = AX(:,I) / AX_SCALE(:)
       END DO
    END IF
    !   Y
    Y_SCALE(:)  = NORM2(Y(:,:),2) / SQRT(RNY)
    WHERE ( Y_SCALE(:) .LT. EPSILON(1.0_RT))  Y_SCALE(:) = 1.0_RT
    DO I = 1, SIZE(Y,2)
       Y(:,I) = Y(:,I) / Y_SCALE(:)
    END DO
    ! Iterate, taking steps with the average gradient over all data.
    fit_loop : DO I = 1, STEPS
       ! Compute the average gradient over all points.
       SUM_SQUARED_ERROR = 0.0_RT
       ! Set gradients to zero initially.
       MODEL_GRAD(:) = 0.0_RT
       !$OMP PARALLEL DO NUM_THREADS(NB) PRIVATE(BATCH) FIRSTPRIVATE(SS, SE) &
       !$OMP& REDUCTION(+: SUM_SQUARED_ERROR, MODEL_GRAD)
       DO BATCH = 1, NB
          ! Set the size start and end.
          IF (CONFIG%ADI .GT. 0) THEN
             SS = BATCHM(1,BATCH)
             SE = BATCHM(2,BATCH)
          END IF
          ! Sum the gradient over all data batches.
          CALL MODEL_GRADIENT(CONFIG, MODEL(:), &
               Y(:,BATCHM(1,BATCH):BATCHM(2,BATCH)), &
               X(:,BATCHM(1,BATCH):BATCHM(2,BATCH)), &
               XI(:,BATCHM(1,BATCH):BATCHM(2,BATCH)), &
               AX(:,BATCHA(1,BATCH):BATCHA(2,BATCH)), &
               AXI(:,BATCHA(1,BATCH):BATCHA(2,BATCH)), &
               SIZES(SS:SE), SUM_SQUARED_ERROR, MODEL_GRAD(:), &
               SQUARED_ERROR_GRADIENT, INFO, SHIFT=.FALSE., THREADS=1)
       END DO
       !$OMP END PARALLEL DO
       IF (INFO .NE. 0) RETURN
       ! Convert the sum of squared errors into the mean squared error.
       MSE = SUM_SQUARED_ERROR / REAL(SIZE(Y),RT) ! RNY * SIZE(Y,1)
       ! Update the step factor based on model improvement.
       IF (MSE .LE. PREV_MSE) THEN
          STEP_FACTOR = STEP_FACTOR * CONFIG%FASTER_RATE
          STEP_MEAN_CHANGE = STEP_MEAN_CHANGE * CONFIG%SLOWER_RATE
          STEP_MEAN_REMAIN = 1.0_RT - STEP_MEAN_CHANGE
          STEP_CURV_CHANGE = STEP_CURV_CHANGE * CONFIG%SLOWER_RATE
          STEP_CURV_REMAIN = 1.0_RT - STEP_CURV_CHANGE
          NUM_TO_UPDATE = MIN(CONFIG%NUM_VARS, &
               NUM_TO_UPDATE + (CONFIG%NUM_VARS - NUM_TO_UPDATE) / 2)
       ELSE
          STEP_FACTOR = STEP_FACTOR * CONFIG%SLOWER_RATE
          STEP_MEAN_CHANGE = STEP_MEAN_CHANGE * CONFIG%FASTER_RATE
          STEP_MEAN_REMAIN = 1.0_RT - STEP_MEAN_CHANGE
          STEP_CURV_CHANGE = STEP_CURV_CHANGE * CONFIG%FASTER_RATE
          STEP_CURV_REMAIN = 1.0_RT - STEP_CURV_CHANGE
          NUM_TO_UPDATE = MAX(1, NUM_TO_UPDATE / 2)
       END IF
       ! Store the previous error for tracking the best-so-far.
       PREV_MSE = MSE

       ! Record that a step was taken.
       NS = NS + 1
       ! Update the saved "best" model based on error.
       IF (MSE .LT. BEST_MSE) THEN
          NS = 0
          BEST_MSE = MSE
          IF (REVERT_TO_BEST) THEN
             BEST_MODEL(:) = MODEL(1:CONFIG%NUM_VARS)
          END IF
       ! Early stop if we don't expect to see a better solution
       !  by the time the fit operation is complete.
       ELSE IF (CONFIG%EARLY_STOP .AND. (NS .GT. STEPS - I)) THEN
          EXIT fit_loop
       END IF

       ! Convert the summed gradients to average gradients.
       MODEL_GRAD(:) = MODEL_GRAD(:) / REAL(NB,RT)
       MODEL_GRAD_MEAN(:) = STEP_MEAN_REMAIN * MODEL_GRAD_MEAN(:) &
            + STEP_MEAN_CHANGE * MODEL_GRAD(:)
       MODEL_GRAD_CURV(:) = STEP_CURV_REMAIN * MODEL_GRAD_CURV(:) &
            + STEP_CURV_CHANGE * (MODEL_GRAD_MEAN(:) - MODEL_GRAD(:))**2
       MODEL_GRAD_CURV(:) = MAX(MODEL_GRAD_CURV(:), EPSILON(STEP_FACTOR))
       ! Set the step as the mean direction (over the past few steps).
       MODEL_GRAD(:) = MODEL_GRAD_MEAN(:)
       ! Start scaling by step magnitude by curvature once enough data is collected.
       IF (I .GE. CONFIG%MIN_STEPS_TO_STABILITY) THEN
          MODEL_GRAD(:) = MODEL_GRAD(:) / SQRT(MODEL_GRAD_CURV(:))
       END IF
       ! Update as many parameters as it seems safe to update (and still converge).
       IF (NUM_TO_UPDATE .LT. CONFIG%NUM_VARS) THEN
          ! Identify the subset of components that will be updapted this step.
          CALL ARGSELECT(-ABS(MODEL_GRAD(:)), NUM_TO_UPDATE, UPDATE_INDICES(:))
          ! Take the gradient steps (based on the computed "step" above).
          MODEL(UPDATE_INDICES(1:NUM_TO_UPDATE)) = MODEL(UPDATE_INDICES(1:NUM_TO_UPDATE)) &
               - MODEL_GRAD(UPDATE_INDICES(1:NUM_TO_UPDATE)) * STEP_FACTOR
       ELSE
          ! Take the gradient steps (based on the computed "step" above).
          MODEL(1:CONFIG%NUM_VARS) = MODEL(1:CONFIG%NUM_VARS) - MODEL_GRAD(:) * STEP_FACTOR
       END IF

       ! Record the 2-norm of the step that was taken (the GRAD variables were updated).
       IF (PRESENT(RECORD)) THEN
          ! Store the mean squared error at this iteration.
          RECORD(1,I) = MSE
          ! Store the current multiplier on the step.
          RECORD(2,I) = STEP_FACTOR
          ! Store the norm of the step that was taken.
          RECORD(3,I) = SQRT(MAX(EPSILON(0.0_RT), SUM(MODEL_GRAD(:)**2)))
          ! Store the percentage of parameters updated in this step.
          RECORD(4,I) = REAL(NUM_TO_UPDATE,RT) / REAL(CONFIG%NUM_VARS)
       END IF

       ! TODO: Replace with a more general "condition_model" routine that takes
       !       the values and gradients for a model, then updates and replaces
       !       bad basis functions, as well as normalizing them all.
       !       Must orthogonalize outputs of basis functions when ranking,
       !       ensure that highly redundant functions are removed in favor
       !       of single (more information-unique) basis functions.
       !       Use the expected decrease in loss (gradient * step) to
       !       determine when a replacement can be made.
       ! 
       ! Maintain a constant max-norm across the magnitue of input and internal vectors.
       CALL UNIT_MAX_NORM(CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, &
            MODEL(CONFIG%MSIV:CONFIG%MEIV), &
            MODEL(CONFIG%MSSV:CONFIG%MESV))
       IF (CONFIG%ADI .GT. 0) THEN
          CALL UNIT_MAX_NORM(CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASSV:CONFIG%AESV))
       END IF

       ! Write an update about step and convergence to the command line.
       CALL CPU_TIME(CURRENT_TIME)
       IF (CURRENT_TIME - LAST_PRINT_TIME .GT. WAIT_TIME) THEN
          IF (DID_PRINT) WRITE (*,'(A)',ADVANCE='NO') RESET_LINE
          WRITE (*,'(I6,"  (",F6.3,") [",F6.3,"]")', ADVANCE='NO') I, MSE, BEST_MSE
          LAST_PRINT_TIME = CURRENT_TIME
          DID_PRINT = .TRUE.
       END IF

    END DO fit_loop

    ! Restore the best model seen so far (if enough steps were taken).
    IF (REVERT_TO_BEST) THEN
       MSE                      = BEST_MSE
       MODEL(1:CONFIG%NUM_VARS) = BEST_MODEL(:)
    END IF

    ! Apply the data normalizing scaling factors to the weight matrices.
    IF (SIZE(X,1) .GT. 0) &
         CALL SCALE_BASIS(CONFIG%MDI, CONFIG%MDS, &
         MODEL(CONFIG%MSIV:CONFIG%MEIV), 1.0_RT / X_SCALE(:))
    IF (SIZE(AX,1) .GT. 0) &
         CALL SCALE_BASIS(CONFIG%ADI, CONFIG%ADS, &
         MODEL(CONFIG%ASIV:CONFIG%AEIV), 1.0_RT / AX_SCALE(:))
    CALL SCALE_BASIS(CONFIG%MDS, CONFIG%MDO, &
         MODEL(CONFIG%MSOV:CONFIG%MEOV), Y_SCALE(:), TRANS=.TRUE.)

    ! Erase the printed message if one was produced.
    IF (DID_PRINT) WRITE (*,'(A)',ADVANCE='NO') RESET_LINE

  CONTAINS

    ! Set the input vectors and the state vectors to 
    SUBROUTINE UNIT_MAX_NORM(MDI, MDS, MNS, INPUT_VECS, STATE_VECS)
      INTEGER, INTENT(IN) :: MDI, MDS, MNS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI,MDS)       :: INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MDS,MNS-1) :: STATE_VECS
      REAL(KIND=RT) :: SCALAR
      INTEGER :: L
      SCALAR = SQRT(MAXVAL(SUM(INPUT_VECS(:,:)**2, 1)))
      INPUT_VECS(:,:) = INPUT_VECS(:,:) / SCALAR
      !$OMP PARALLEL DO NUM_THREADS(NB) PRIVATE(L, SCALAR)
      DO L = 1, SIZE(STATE_VECS,3)
         SCALAR = SQRT(MAXVAL(SUM(STATE_VECS(:,:,L)**2, 1)))
         STATE_VECS(:,:,L) = STATE_VECS(:,:,L) / SCALAR
      END DO
      !$OMP END PARALLEL DO
    END SUBROUTINE UNIT_MAX_NORM

    ! Scale a set of basis functions by "weights".
    SUBROUTINE SCALE_BASIS(M, N, MATRIX, WEIGHTS, TRANS)
      INTEGER, INTENT(IN) :: M, N
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(M,N) :: MATRIX
      REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: WEIGHTS
      LOGICAL, INTENT(IN), OPTIONAL :: TRANS
      LOGICAL :: T
      INTEGER :: I
      IF (PRESENT(TRANS)) THEN ; T = TRANS
      ELSE                     ; T = .FALSE.
      END IF
      IF (T) THEN
         DO I = 1, SIZE(WEIGHTS)
            MATRIX(:,I) = MATRIX(:,I) * WEIGHTS(I)
         END DO
      ELSE
         DO I = 1, SIZE(WEIGHTS)
            MATRIX(I,:) = MATRIX(I,:) * WEIGHTS(I)
         END DO
      END IF
    END SUBROUTINE SCALE_BASIS

    
    ! ------------------------------------------------------------------
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
      USE ISO_FORTRAN_ENV, ONLY: RT => REAL32
      IMPLICIT NONE
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

    SUBROUTINE SWAP_INT(V1, V2)
      IMPLICIT NONE
      INTEGER, INTENT(INOUT) :: V1, V2
      INTEGER :: TEMP
      TEMP = V1
      V1 = V2
      V2 = TEMP
    END SUBROUTINE SWAP_INT


  END SUBROUTINE MINIMIZE_MSE


END MODULE APOS

