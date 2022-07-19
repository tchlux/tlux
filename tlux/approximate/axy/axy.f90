! TODO:
! 
! - Update FIT code to allow specifying a subset of points to be evaluated
!   at a time (i.e., support batched evaluation).
! - Rotate out the points that have the lowest expected change in error.
! 
! - Add (PAIRWISE_AGGREGATION, MAX_PAIRS) functionality to the aggregator model input,
!   where a nonrepeating random number generator is used in conjunction with a mapping
!   from the integer line to pairs of points. When MAX_PAIRS is less than the total
!   number of pairs, then do greedy rotation of points identically to above.
! 
! - Handle NaN and Infinity in the data normalization process, as well
!   as in model evaluation (controlled through some logical setting).
!   Replace NaN/Inf inputs with means, ignore NaN/Inf outputs in gardient.
! 
! - Update CONDITION_MODEL to:
!    multiply the 2-norm of output weights by values before orthogonalization
!    compress basis weights with linear regression when rank deficiency is detected
!    reinitialize basis functions randomly at first, metric PCA best
!    sum the number of times a component had no rank across threads
!    swap weights for the no-rank components to the back
!    swap no-rank state component values into contiguous memory at back
!    linearly regress the kept-components onto the next-layer dropped difference
!    compute the first no-rank principal components of the gradient, store in droped slots
!    regress previous layer onto the gradient components
!    fill any remaining nodes (if not enough from gradient) with "uncaptured" principal components
!    set new shift terms as the best of 5 well spaced values in [-1,1], or random given no order
! - Run some tests to determine how weights should be updated on conditioning
!   to not change the output of the model *at all*, and similarly update the
!   gradient estimates to reflect those changes as well.
! 
! - Update Python testing code to test all combinations of AX, AXI, AY, X, XI, and Y.
! - Update Python testing code to attempt different edge-case model sizes
!    (linear regression, no aggregator, no model).
! - Start writing a test-case suite that tests all of the basic operations.
!   Start with end-to-end tests that include "bad" input and output scaling.
! - Verify that the *condition model* operation correctly updates the gradient
!   related variables (mean and curvature).
! - Make sure that the print time actually adheres to the 3-second guidance.
!   Or optionally write updates to a designated file instead.
! 
! - Make data normalization use the same work space as the fit procedure
!   (since these are not needed at the same time).
! - Make model conditioning use the same work space as evaluation (where possible).
! - Pull normalization code out and have it be called separately from 'FIT'.
!   Goal is to achieve near-zero inefficiencies for doing a few steps at a time in
!   Python (allowing for easier cancellation, progress updates, checkpoints, ...).
! 
! - Use LAPACK to do linear regression, implement simple SVD + gradient descent method
!   in MATRIX_OPERATIONS, compare speed of both methodologies.
! - Implement and test Fortran native version of matrix multiplication (manual DO loop).
! - Implement and test Fortran native version of SSYRK (manual DO loop).
! 
! ---------------------------------------------------------------------------
! 
! NOTES:
! 
! - When conditioning the model, the multipliers applied to the weight matrices
!   can affect the model gradient in nonlinear (difficult to predict) ways.
! 
! - When taking adaptive optimization steps, the current architecture takes steps
!   and then projects back onto the feasible region (of the optimization space).
!   The projection is not incorporated directly into the steps, so it is possible
!   for these two operations to "fight" each other. This is ignored.
! 

! An aggregator and fixed piecewise linear regression model.
MODULE AXY
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, INT64, INT8
  USE IEEE_ARITHMETIC, ONLY: IS_NAN => IEEE_IS_NAN, IS_FINITE => IEEE_IS_FINITE
  USE RANDOM, ONLY: RANDOM_UNIT_VECTORS
  USE SORT_AND_SELECT, ONLY: ARGSORT, ARGSELECT
  USE MATRIX_OPERATIONS, ONLY: GEMM, ORTHOGONALIZE, RADIALIZE, LEAST_SQUARES

  IMPLICIT NONE

  ! Model configuration, internal sizes and fit parameters.
  TYPE, BIND(C) :: MODEL_CONFIG
     ! Aggregator model configuration.
     INTEGER :: ADN      ! aggregator dimension numeric (input)
     INTEGER :: ADE = 0  ! aggregator dimension of embeddings
     INTEGER :: ANE = 0  ! aggregator number of embeddings
     INTEGER :: ADS = 32 ! aggregator dimension of state
     INTEGER :: ANS = 8  ! aggregator number of states
     INTEGER :: ADO      ! aggregator dimension of output
     INTEGER :: ADI      ! aggregator dimension of input (internal usage only)
     INTEGER :: ADSO     ! aggregator dimension of state output (internal usage only)
     ! (Positional) model configuration.
     INTEGER :: MDN      ! model dimension numeric (input)
     INTEGER :: MDE = 0  ! model dimension of embeddings
     INTEGER :: MNE = 0  ! model number of embeddings
     INTEGER :: MDS = 32 ! model dimension of state
     INTEGER :: MNS = 8  ! model number of states
     INTEGER :: MDO      ! model dimension of output
     INTEGER :: MDI      ! model dimension of input (internal usage only)
     INTEGER :: MDSO     ! model dimension of state output (internal usage only)
     ! Summary numbers that are computed.
     INTEGER :: TOTAL_SIZE
     INTEGER :: NUM_VARS
     ! Index subsets of total size vector naming scheme:
     !   M___ -> model,   A___ -> aggregator model
     !   _S__ -> start,   _E__ -> end
     !   __I_ -> input,   __S_ -> states, __O_ -> output, __E_ -> embedding
     !   ___V -> vectors, ___S -> shifts
     INTEGER :: ASIV, AEIV, ASIS, AEIS ! aggregator input
     INTEGER :: ASSV, AESV, ASSS, AESS ! aggregator states
     INTEGER :: ASOV, AEOV             ! aggregator output
     INTEGER :: ASEV, AEEV             ! aggregator embedding
     INTEGER :: MSIV, MEIV, MSIS, MEIS ! model input
     INTEGER :: MSSV, MESV, MSSS, MESS ! model states
     INTEGER :: MSOV, MEOV             ! model output
     INTEGER :: MSEV, MEEV             ! model embedding
     ! Index subsets for input and output shifts.
     ! M___ -> model,       A___ -> aggregator (/ aggregate) model
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
     REAL(KIND=RT) :: STEP_FACTOR = 0.001_RT     ! Initial multiplier on gradient steps.
     REAL(KIND=RT) :: STEP_MEAN_CHANGE = 0.1_RT  ! Rate of exponential sliding average over gradient steps.
     REAL(KIND=RT) :: STEP_CURV_CHANGE = 0.01_RT ! Rate of exponential sliding average over gradient variation.
     REAL(KIND=RT) :: STEP_AY_CHANGE = 0.05_RT   ! Rate of exponential sliding average over AY (forcing mean to zero).
     REAL(KIND=RT) :: FASTER_RATE = 1.01_RT      ! Rate of increase of optimization factors.
     REAL(KIND=RT) :: SLOWER_RATE = 0.99_RT      ! Rate of decrease of optimization factors.
     REAL(KIND=RT) :: MIN_UPDATE_RATIO = 0.1_RT  ! Minimum ratio of model variables to update in any optimizaiton step.
     INTEGER :: MIN_STEPS_TO_STABILITY = 1 ! Minimum number of steps before allowing model saves and curvature approximation.
     INTEGER :: NUM_THREADS = 1 ! Number of parallel threads to use in fit & evaluation.
     INTEGER :: PRINT_DELAY_SEC = 2 ! Delay between output logging during fit.
     INTEGER :: STEPS_TAKEN = 0 ! Total number of updates made to model variables.
     INTEGER :: LOGGING_STEP_FREQUENCY = 10 ! Frequency with which to log expensive records (model variable 2-norm step size).
     INTEGER :: RANK_CHECK_FREQUENCY = 10 ! Frequency with which to orthogonalize internal basis functions.
     INTEGER :: NUM_TO_UPDATE = HUGE(0) ! Number of model variables to update (initialize to large number).
     LOGICAL(KIND=INT8) :: AX_NORMALIZED = .FALSE. ! False if AX data needs to be normalized.
     LOGICAL(KIND=INT8) :: AXI_NORMALIZED = .FALSE. ! False if AXI embeddings need to be normalized.
     LOGICAL(KIND=INT8) :: AY_NORMALIZED = .FALSE. ! False if aggregator outputs need to  be normalized.
     LOGICAL(KIND=INT8) :: X_NORMALIZED = .FALSE. ! False if X data needs to be normalized.
     LOGICAL(KIND=INT8) :: XI_NORMALIZED = .FALSE. ! False if XI embeddings need to be normalized.
     LOGICAL(KIND=INT8) :: Y_NORMALIZED = .FALSE. ! False if Y data needs to be normalized.
     LOGICAL(KIND=INT8) :: EQUALIZE_Y = .FALSE. ! Rescale all Y components to be equally weighted.
     LOGICAL(KIND=INT8) :: ENCODE_NORMALIZATION = .TRUE. ! True if input and output weight matrices shuld embed normalization.
     LOGICAL(KIND=INT8) :: APPLY_SHIFT = .TRUE. ! True if shifts should be applied to inputs before processing.
     LOGICAL(KIND=INT8) :: KEEP_BEST = .TRUE. ! True if best observed model should be greedily kept at end of optimization.
     LOGICAL(KIND=INT8) :: EARLY_STOP = .TRUE. ! True if optimization should end when num-steps since best model is greater than the num-steps remaining.
     LOGICAL(KIND=INT8) :: BASIS_REPLACEMENT = .FALSE. ! True if redundant basis functions should be replaced during optimization rank checks.
     ! Descriptions of the number of points that can be in one batch.
     INTEGER(KIND=INT64) :: RWORK_SIZE = 0
     INTEGER(KIND=INT64) :: IWORK_SIZE = 0
     INTEGER(KIND=INT64) :: NA = 0
     INTEGER(KIND=INT64) :: NM = 0
     ! Optimization real work space.
     INTEGER(KIND=INT64) :: SMG, EMG ! MODEL_GRAD(NUM_VARS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMGM, EMGM ! MODEL_GRAD_MEAN(NUM_VARS)
     INTEGER(KIND=INT64) :: SMGC, EMGC ! MODEL_GRAD_CURV(NUM_VARS)
     INTEGER(KIND=INT64) :: SBM, EBM ! BEST_MODEL(NUM_VARS)
     INTEGER(KIND=INT64) :: SAXS, EAXS ! A_STATES(NA,ADS,ANS+1)
     INTEGER(KIND=INT64) :: SAXG, EAXG ! A_GRADS(NA,ADS,ANS+1)
     INTEGER(KIND=INT64) :: SAY, EAY ! AY(NA,ADO)
     INTEGER(KIND=INT64) :: SAYG, EAYG ! AY_GRADIENT(NA,ADO)
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
     FUNCTION OMP_GET_THREAD_NUM()
       INTEGER :: OMP_GET_THREAD_NUM
     END FUNCTION OMP_GET_THREAD_NUM
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
     ELSE IF (CONFIG%ANS .EQ. 0) THEN
        CONFIG%ADO = MIN(32, CONFIG%ADI)
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
     ! Compute indices related to the variable locations for this model.
     CONFIG%TOTAL_SIZE = 0
     ! ---------------------------------------------------------------
     !   aggregator input vecs
     CONFIG%ASIV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEIV = CONFIG%ASIV-1  +  CONFIG%ADI * CONFIG%ADS
     CONFIG%TOTAL_SIZE = CONFIG%AEIV
     !   aggregator input shift
     CONFIG%ASIS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEIS = CONFIG%ASIS-1  +  CONFIG%ADS
     CONFIG%TOTAL_SIZE = CONFIG%AEIS
     !   aggregator state vecs
     CONFIG%ASSV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AESV = CONFIG%ASSV-1  +  CONFIG%ADS * CONFIG%ADS * MAX(0,CONFIG%ANS-1)
     CONFIG%TOTAL_SIZE = CONFIG%AESV
     !   aggregator state shift
     CONFIG%ASSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AESS = CONFIG%ASSS-1  +  CONFIG%ADS * MAX(0,CONFIG%ANS-1)
     CONFIG%TOTAL_SIZE = CONFIG%AESS
     !   aggregator output vecs
     CONFIG%ASOV = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AEOV = CONFIG%ASOV-1  +  CONFIG%ADSO * CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AEOV
     !   aggregator embedding vecs
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
     ! THIS IS SPECIAL, IT IS PART OF MODEL AND CHANGES DURING OPTIMIZATION
     !   aggregator post-output shift
     CONFIG%AOSS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AOSE = CONFIG%AOSS-1 + CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AOSE
     ! ---------------------------------------------------------------
     !   number of variables
     CONFIG%NUM_VARS = CONFIG%TOTAL_SIZE
     ! ---------------------------------------------------------------
     !   aggregator pre-input shift
     CONFIG%AISS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%AISE = CONFIG%AISS-1 + CONFIG%ADN
     CONFIG%TOTAL_SIZE = CONFIG%AISE
     !   model pre-input shift
     CONFIG%MISS = 1 + CONFIG%TOTAL_SIZE
     CONFIG%MISE = CONFIG%MISS-1 + CONFIG%MDN
     CONFIG%TOTAL_SIZE = CONFIG%MISE
     !   model post-output shift
     CONFIG%MOSS = 1 + CONFIG%TOTAL_SIZE
     IF (CONFIG%MDO .GT. 0) THEN
        CONFIG%MOSE = CONFIG%MOSS-1 + CONFIG%MDO
     ELSE
        CONFIG%MOSE = CONFIG%MOSS-1 + CONFIG%ADO
     END IF
     CONFIG%TOTAL_SIZE = CONFIG%MOSE
  END SUBROUTINE NEW_MODEL_CONFIG

  ! Given a number of X points "NM", and a number of aggregator X points
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
    ! aggregator states
    CONFIG%SAXS = 1 + CONFIG%RWORK_SIZE
    CONFIG%EAXS = CONFIG%SAXS-1 + CONFIG%NA * CONFIG%ADS * (CONFIG%ANS+1)
    CONFIG%RWORK_SIZE = CONFIG%EAXS
    ! aggregator gradients at states
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
    ! aggregator batch starts
    CONFIG%SBAS = 1 + CONFIG%IWORK_SIZE
    CONFIG%EBAS = CONFIG%SBAS-1 + CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EBAS
    ! aggregator batch ends
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
    !  Storage for seeding the random number generator (for repeatability). LOCAL ALLOCATION
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
    ! Initialize the aggregator model.
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
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI, MDS) :: INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS, MDS, MAX(0,MNS-1)) :: STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS, MAX(0,MNS-1)) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDSO, MDO) :: OUTPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDE, MNE) :: EMBEDDINGS
      ! Local holder for "origin" at each layer.
      REAL(KIND=RT), DIMENSION(MDS) :: ORIGIN ! LOCAL ALLOCATION
      INTEGER,       DIMENSION(MDS) :: ORDER  ! LOCAL ALLOCATION
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
      ! Generate deterministic equally spaced shifts for inputs and internal layers, 
      !  zero shift for the output layer (first two will be rescaled).
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
  !  the aggregator and positional inputs. Store in (2,_) arrays.
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
       PRINT *, 'ERROR (COMPUTE_BATCHES): Number of batches is not positive.', NUM_BATCHES
       INFO = -4
       RETURN
    END IF
    ! Construct batches for data sets with aggregator inputs.
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
    ! LOCAL ALLOCATION
    INTEGER, DIMENSION(:), ALLOCATABLE :: BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS
    ! If there are no points to evaluate, then immediately return.
    IF (SIZE(Y,2,KIND=INT64) .EQ. 0) RETURN
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
       ! Aggregator model forward pass.
       IF (CONFIG%ADI .GT. 0) THEN
          BS = BATCHA_STARTS(BATCH)
          BE = BATCHA_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .LE. 0) CYCLE batch_evaluation
          ! Apply shift terms to aggregator inputs.
          IF ((CONFIG%APPLY_SHIFT) .AND. (CONFIG%ADN .GT. 0)) THEN
             DO I = BS, BE
                AX(:CONFIG%ADN,I) = AX(:CONFIG%ADN,I) + MODEL(CONFIG%AISS:CONFIG%AISE)
             END DO
             ! Remove any NaN or Inf values from the data.
             WHERE (IS_NAN(AX(:,:)) .OR. (.NOT. IS_FINITE(AX(:,:))))
                AX(:,:) = 0.0_RT
             END WHERE
          END IF
          ! Evaluate the aggregator model.
          CALL UNPACKED_EVALUATE(BT, &
               CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADSO, CONFIG%ADO, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASIS:CONFIG%AEIS), &
               MODEL(CONFIG%ASSV:CONFIG%AESV), &
               MODEL(CONFIG%ASSS:CONFIG%AESS), &
               MODEL(CONFIG%ASOV:CONFIG%AEOV), &
               AX(:,BS:BE), AY(BS:BE,:), A_STATES(BS:BE,:,:), YTRANS=.TRUE.)
          ! Unapply shift terms to aggregator inputs.
          IF ((CONFIG%APPLY_SHIFT) .AND. (CONFIG%ADN .GT. 0)) THEN
             DO I = BS, BE
                AX(:CONFIG%ADN,I) = AX(:CONFIG%ADN,I) - MODEL(CONFIG%AISS:CONFIG%AISE)
             END DO
          END IF
          GS = BS ! First group start is the batch start.
          ! Take the mean of all outputs from the aggregator model, store
          !   as input to the model that proceeds this aggregation.
          IF (CONFIG%MDO .GT. 0) THEN
             ! Apply zero-mean shift terms to aggregator model outputs.
             DO I = 1, CONFIG%ADO
                AY(BS:BE,I) = AY(BS:BE,I) + MODEL(CONFIG%AOSS + I-1)
             END DO
             E = CONFIG%MDN+CONFIG%MDE+1 ! <- start of aggregator output
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
          IF (BT .LE. 0) CYCLE batch_evaluation
          ! Apply shift terms to numeric model inputs.
          IF ((CONFIG%APPLY_SHIFT) .AND. (CONFIG%MDN .GT. 0)) THEN
             DO I = BS, BE
                X(:CONFIG%MDN,I) = X(:CONFIG%MDN,I) + MODEL(CONFIG%MISS:CONFIG%MISE)
             END DO
             ! Remove any NaN or Inf values from the data.
             WHERE (IS_NAN(X(:,:)) .OR. (.NOT. IS_FINITE(X(:,:))))
                X(:,:) = 0.0_RT
             END WHERE
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
    ! Local variables. LOCAL ALLOCATION
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
  SUBROUTINE MODEL_GRADIENT(CONFIG, MODEL, AX, AXI, AY, SIZES, X, XI, Y, YW, &
       SUM_SQUARED_GRADIENT, MODEL_GRAD, INFO, &
       AY_GRADIENT, Y_GRADIENT, A_GRADS, M_GRADS, A_STATES, M_STATES)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: AXI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY
    INTEGER,       INTENT(IN),    DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: YW
    ! Sum (over all data) squared error (summed over dimensions).
    REAL(KIND=RT), INTENT(INOUT) :: SUM_SQUARED_GRADIENT
    ! Gradient of the model variables.
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:) :: MODEL_GRAD
    ! Output and optional inputs.
    INTEGER, INTENT(INOUT) :: INFO
    ! Work space.
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: M_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:), OPTIONAL :: A_STATES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:), OPTIONAL :: M_STATES
    INTEGER :: L, D
    ! Exit early if there is no data.
    IF (SIZE(Y,2,KIND=INT64) .EQ. 0) RETURN
    ! Embed all integer inputs into real vector inputs.
    CALL EMBED(CONFIG, MODEL, AXI, XI, AX, X)
    ! Evaluate the model, storing internal states (for gradient calculation).
    IF (PRESENT(A_STATES) .AND. PRESENT(M_STATES)) THEN
       CALL EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, Y_GRADIENT, A_STATES, M_STATES, INFO)
       ! Copy the state values into holders for the gradients.
       A_GRADS(:,:,:) = A_STATES(:,:,:)
       AY_GRADIENT(:,:) = AY(:,:)
       M_GRADS(:,:,:) = M_STATES(:,:,:)
    ELSE
       CALL EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, Y_GRADIENT, A_GRADS, M_GRADS, INFO)
    END IF
    ! Compute the gradient of the model outputs, overwriting "Y_GRADIENT"
    Y_GRADIENT(:,:) = Y_GRADIENT(:,:) - Y(:,:) ! squared error gradient
    ! Apply weights to the computed gradients (if they were provided.
    IF (SIZE(YW,1) .EQ. SIZE(Y,1)) THEN
       Y_GRADIENT(:,:) = Y_GRADIENT(:,:) * YW(:,:)
    ELSE IF (SIZE(YW,1) .EQ. 1) THEN
       DO D = 1, SIZE(Y,1)
          Y_GRADIENT(D,:) = Y_GRADIENT(D,:) * YW(1,:)
       END DO
    END IF
    ! Compute the total squared gradient.
    SUM_SQUARED_GRADIENT = SUM_SQUARED_GRADIENT + SUM(Y_GRADIENT(:,:)**2)
    ! Compute the gradient with respect to the model basis functions.
    CALL BASIS_GRADIENT(CONFIG, MODEL, Y_GRADIENT, X, AX, &
         SIZES, M_GRADS, A_GRADS, AY_GRADIENT, MODEL_GRAD)
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
  SUBROUTINE NORMALIZE_DATA(CONFIG, MODEL, AX, AXI, AY, SIZES, X, XI, Y, YW, &
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
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW
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
    !$OMP PARALLEL NUM_THREADS(6)
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
            Y_RESCALE(:,:), INVERT_RESULT=.TRUE., FLATTEN=LOGICAL(CONFIG%EQUALIZE_Y))
       CONFIG%Y_NORMALIZED = .TRUE.
    ELSE
       MODEL(CONFIG%MOSS:CONFIG%MOSE) = 0.0_RT
       Y_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:SIZE(Y,1)) Y_RESCALE(D,D) = 1.0_RT
    END IF
    !$OMP SECTION
    IF (SIZE(YW) .GT. 0) THEN
       YW(:,:) = YW(:,:) / (SUM(YW(:,:)) / REAL(SIZE(YW),RT))
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
  SUBROUTINE CONDITION_MODEL(CONFIG, MODEL, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, &
       AX, AXI, AY, AY_GRADIENT, X, XI, Y, Y_GRADIENT, &
       NUM_THREADS, FIT_STEP, &
       A_STATES, M_STATES, A_GRADS, M_GRADS, &
       A_LENGTHS, M_LENGTHS, A_STATE_TEMP, M_STATE_TEMP, A_ORDER, M_ORDER, &
       TOTAL_EVAL_RANK, TOTAL_GRAD_RANK)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL_GRAD_MEAN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL_GRAD_CURV
    ! Data.
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: AXI
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: AY
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: AY_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y_GRADIENT
    ! Configuration.
    INTEGER, INTENT(IN) :: NUM_THREADS, FIT_STEP
    ! States, gradients, lengths, temporary storage, and order (of ranks).
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_STATES, M_STATES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_GRADS, M_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_LENGTHS, M_LENGTHS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_STATE_TEMP, M_STATE_TEMP
    INTEGER,       INTENT(INOUT), DIMENSION(:,:) :: A_ORDER, M_ORDER
    INTEGER :: TOTAL_EVAL_RANK, TOTAL_GRAD_RANK
    ! 
    ! Maintain a constant max-norm across the magnitue of input and internal vectors.
    ! 
    CALL UNIT_MAX_NORM(CONFIG, NUM_THREADS, &
         MODEL(CONFIG%ASIV:CONFIG%AEIV), & ! A input vecs
         MODEL(CONFIG%ASIS:CONFIG%AEIS), & ! A input shift
         MODEL(CONFIG%ASSV:CONFIG%AESV), & ! A state vecs
         MODEL(CONFIG%ASSS:CONFIG%AESS), & ! A state shift
         MODEL(CONFIG%ASOV:CONFIG%AEOV), & ! A out vecs
         MODEL(CONFIG%AOSS:CONFIG%AOSE), & ! AY shift
         MODEL(CONFIG%MSIV:CONFIG%MEIV), & ! M input vecs
         MODEL(CONFIG%MSIS:CONFIG%MEIS), & ! M input shift
         MODEL(CONFIG%MSSV:CONFIG%MESV), & ! M state vecs
         MODEL(CONFIG%MSSS:CONFIG%MESS), & ! M state shift
         MODEL(CONFIG%MSOV:CONFIG%MEOV)) ! M output vecs
    ! 
    ! Check rank and optionally replace redundant basis functions.
    ! 
    IF ((CONFIG%RANK_CHECK_FREQUENCY .GT. 0) .AND. &
         (MOD(FIT_STEP-1,CONFIG%RANK_CHECK_FREQUENCY) .EQ. 0)) THEN
       ! Embed all integer inputs into real vector inputs.
       CALL EMBED(CONFIG, MODEL, AXI, XI, AX, X)
       ! Compute total rank for values at all internal layers.
       TOTAL_EVAL_RANK = 0
       TOTAL_GRAD_RANK = 0
       ! Update for the aggregator model.
       CALL CHECK_MODEL_RANK( &
            CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADSO, CONFIG%ADO, NUM_THREADS, &
            AX(:,:), AY_GRADIENT(:,:), &
            MODEL(CONFIG%ASIV:CONFIG%AEIV), & ! A input vecs
            MODEL(CONFIG%ASIS:CONFIG%AEIS), & ! A input shift
            MODEL(CONFIG%ASSV:CONFIG%AESV), & ! A state vecs
            MODEL(CONFIG%ASSS:CONFIG%AESS), & ! A state shift
            MODEL(CONFIG%ASOV:CONFIG%AEOV), & ! A out vecs
            MODEL_GRAD_MEAN(CONFIG%ASIV:CONFIG%AEIV), & ! A input vecs
            MODEL_GRAD_MEAN(CONFIG%ASIS:CONFIG%AEIS), & ! A input shift
            MODEL_GRAD_MEAN(CONFIG%ASSV:CONFIG%AESV), & ! A state vecs
            MODEL_GRAD_MEAN(CONFIG%ASSS:CONFIG%AESS), & ! A state shift
            MODEL_GRAD_MEAN(CONFIG%ASOV:CONFIG%AEOV), & ! A out vecs
            MODEL_GRAD_CURV(CONFIG%ASIV:CONFIG%AEIV), & ! A input vecs
            MODEL_GRAD_CURV(CONFIG%ASIS:CONFIG%AEIS), & ! A input shift
            MODEL_GRAD_CURV(CONFIG%ASSV:CONFIG%AESV), & ! A state vecs
            MODEL_GRAD_CURV(CONFIG%ASSS:CONFIG%AESS), & ! A state shift
            MODEL_GRAD_CURV(CONFIG%ASOV:CONFIG%AEOV), & ! A out vecs
            A_STATE_TEMP(:,:), A_STATES(:,:,:), A_LENGTHS(:,:), A_ORDER(:,:), A_GRADS(:,:,:))
       ! Update for the model.
       CALL CHECK_MODEL_RANK( &
            CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDSO, CONFIG%MDO, NUM_THREADS, &
            X(:,:), TRANSPOSE(Y_GRADIENT(:,:)), &
            MODEL(CONFIG%MSIV:CONFIG%MEIV), & ! M input vecs
            MODEL(CONFIG%MSIS:CONFIG%MEIS), & ! M input shift
            MODEL(CONFIG%MSSV:CONFIG%MESV), & ! M state vecs
            MODEL(CONFIG%MSSS:CONFIG%MESS), & ! M state shift
            MODEL(CONFIG%MSOV:CONFIG%MEOV), & ! M out vecs
            MODEL_GRAD_MEAN(CONFIG%MSIV:CONFIG%MEIV), & ! M input vecs
            MODEL_GRAD_MEAN(CONFIG%MSIS:CONFIG%MEIS), & ! M input shift
            MODEL_GRAD_MEAN(CONFIG%MSSV:CONFIG%MESV), & ! M state vecs
            MODEL_GRAD_MEAN(CONFIG%MSSS:CONFIG%MESS), & ! M state shift
            MODEL_GRAD_MEAN(CONFIG%MSOV:CONFIG%MEOV), & ! M output vecs
            MODEL_GRAD_CURV(CONFIG%MSIV:CONFIG%MEIV), & ! M input vecs
            MODEL_GRAD_CURV(CONFIG%MSIS:CONFIG%MEIS), & ! M input shift
            MODEL_GRAD_CURV(CONFIG%MSSV:CONFIG%MESV), & ! M state vecs
            MODEL_GRAD_CURV(CONFIG%MSSS:CONFIG%MESS), & ! M state shift
            MODEL_GRAD_CURV(CONFIG%MSOV:CONFIG%MEOV), & ! M output vecs
            M_STATE_TEMP(:,:), M_STATES(:,:,:), M_LENGTHS(:,:), M_ORDER(:,:), M_GRADS(:,:,:))
    END IF

  CONTAINS

    ! Make max length vector in each weight matrix have unit length.
    SUBROUTINE UNIT_MAX_NORM(CONFIG, NUM_THREADS, &
         A_INPUT_VECS, A_INPUT_SHIFT, A_STATE_VECS, A_STATE_SHIFT, A_OUTPUT_VECS, AY_SHIFT, &
         M_INPUT_VECS, M_INPUT_SHIFT, M_STATE_VECS, M_STATE_SHIFT, M_OUTPUT_VECS)
      TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
      INTEGER, INTENT(IN) :: NUM_THREADS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADI, CONFIG%ADS) :: A_INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS) :: A_INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS, CONFIG%ADS, MAX(0,CONFIG%ANS-1)) :: A_STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS, MAX(0,CONFIG%ANS-1)) :: A_STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADSO, CONFIG%ADO) :: A_OUTPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADO) :: AY_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDI, CONFIG%MDS) :: M_INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDS) :: M_INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDS, CONFIG%MDS, MAX(0,CONFIG%MNS-1)) :: M_STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDS, MAX(0,CONFIG%MNS-1)) :: M_STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDSO, CONFIG%MDO) :: M_OUTPUT_VECS
      ! Local variables.
      INTEGER :: L
      REAL(KIND=RT) :: SCALAR
      ! 
      !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) PRIVATE(SCALAR)
      DO L = 1, CONFIG%MNS+CONFIG%ANS+1
         ! [1,ANS-1] -> A_STATE_VECS
         IF (L .LT. CONFIG%ANS) THEN
            SCALAR = SQRT(MAXVAL(SUM(A_STATE_VECS(:,:,L)**2, 1)))
            A_STATE_VECS(:,:,L) = A_STATE_VECS(:,:,L) / SCALAR
            A_STATE_SHIFT(:,L) = A_STATE_SHIFT(:,L) / SCALAR
            ! [ANS] -> A_INPUT_VECS
         ELSE IF (L .EQ. CONFIG%ANS) THEN
            SCALAR = SQRT(MAXVAL(SUM(A_INPUT_VECS(:,:)**2, 1)))
            A_INPUT_VECS(:,:) = A_INPUT_VECS(:,:) / SCALAR
            A_INPUT_SHIFT(:) = A_INPUT_SHIFT(:) / SCALAR
            ! [ANS+1, ANS+MNS-1] -> M_STATE_VECS
         ELSE IF (L-CONFIG%ANS .LT. CONFIG%MNS) THEN
            SCALAR = SQRT(MAXVAL(SUM(M_STATE_VECS(:,:,L-CONFIG%ANS)**2, 1)))
            M_STATE_VECS(:,:,L-CONFIG%ANS) = M_STATE_VECS(:,:,L-CONFIG%ANS) / SCALAR
            M_STATE_SHIFT(:,L-CONFIG%ANS) = M_STATE_SHIFT(:,L-CONFIG%ANS) / SCALAR
            ! [ANS+MNS] -> M_INPUT_VECS
         ELSE IF (L .EQ. CONFIG%ANS+CONFIG%MNS) THEN
            SCALAR = SQRT(MAXVAL(SUM(M_INPUT_VECS(:,:)**2, 1)))
            M_INPUT_VECS(:,:) = M_INPUT_VECS(:,:) / SCALAR
            M_INPUT_SHIFT(:) = M_INPUT_SHIFT(:) / SCALAR
            ! [ANS+MNS+1] -> AY
         ELSE
            ! Update the aggregator model output shift to produce componentwise mean-zero
            !  values (prevent divergence), but only when there is a model afterwards. 
            IF ((CONFIG%MDO .GT. 0) .AND. (CONFIG%ADO .GT. 0)) THEN
               MODEL(CONFIG%AOSS:CONFIG%AOSE) = &
                    (1.0_RT - CONFIG%STEP_AY_CHANGE) *  MODEL(CONFIG%AOSS:CONFIG%AOSE) &
                    - CONFIG%STEP_AY_CHANGE  * (SUM(AY(:,:),1) / REAL(SIZE(AY,1),RT))
            END IF
         END IF
      END DO
      !$OMP END PARALLEL DO
    END SUBROUTINE UNIT_MAX_NORM

    ! Check the rank of all internal states.
    SUBROUTINE CHECK_MODEL_RANK(DI, DS, NS, DSO, DO, NUM_THREADS, X, Y_GRADIENT, &
         INPUT_VECS, INPUT_SHIFT, STATE_VECS, STATE_SHIFT, OUTPUT_VECS, &
         INPUT_VECS_GRAD_MEAN, INPUT_SHIFT_GRAD_MEAN, STATE_VECS_GRAD_MEAN, STATE_SHIFT_GRAD_MEAN, OUTPUT_VECS_GRAD_MEAN, &
         INPUT_VECS_GRAD_CURV, INPUT_SHIFT_GRAD_CURV, STATE_VECS_GRAD_CURV, STATE_SHIFT_GRAD_CURV, OUTPUT_VECS_GRAD_CURV, &
         STATE_TEMP, STATES, LENGTHS, ORDER, GRADS)
      INTEGER, INTENT(IN) :: DI, DS, NS, DSO, DO, NUM_THREADS
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X, Y_GRADIENT
      ! Model variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DI, DS) :: INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, DS, MAX(0,NS-1)) :: STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, MAX(0,NS-1)) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DSO, DO) :: OUTPUT_VECS
      ! Gradient means for all variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DI, DS) :: INPUT_VECS_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS) :: INPUT_SHIFT_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, DS, MAX(0,NS-1)) :: STATE_VECS_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, MAX(0,NS-1)) :: STATE_SHIFT_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DSO, DO) :: OUTPUT_VECS_GRAD_MEAN
      ! Gradient curvatures for all variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DI, DS) :: INPUT_VECS_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS) :: INPUT_SHIFT_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, DS, MAX(0,NS-1)) :: STATE_VECS_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, MAX(0,NS-1)) :: STATE_SHIFT_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DSO, DO) :: OUTPUT_VECS_GRAD_CURV
      ! Temporary variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: STATE_TEMP
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: STATES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: LENGTHS
      INTEGER, INTENT(INOUT), DIMENSION(:,:) :: ORDER
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: GRADS
      INTEGER :: BATCH, BS, BE, BN, I, N, NT, TER, TGR
      ! TODO: This allocation should occur at workspace initialization.
      INTEGER, DIMENSION(DS, NUM_THREADS) :: STATE_USAGE ! LOCAL ALLOCATION
      ! 
      ! Batch computation formula.
       IF (NS .GT. 0) THEN
          N = SIZE(STATE_TEMP,1)
          NT = MIN(NUM_THREADS, MAX(1, N / DS)) ! number of threads (as not to artificially reduce rank)
          BN = (N + NT - 1) / NT ! = CEIL(N / NT)
       END IF
       DO I = 1, NS
          STATE_USAGE(:,:) = 0
          TER = 0; TGR = 0;
          !$OMP PARALLEL DO PRIVATE(BS,BE) NUM_THREADS(NT) &
          !$OMP& REDUCTION(MAX: TER, TGR)
          DO BATCH = 1, NT
             BS = BN*(BATCH-1) + 1
             BE = MIN(N, BN*BATCH)
             ! Compute model state rank.
             STATE_TEMP(BS:BE,:) = STATES(BS:BE,:,I)
             ! ! TODO: multiply column values by 2-norm magnitude of output weights
             ! IF (I .LT. NS) THEN
             !    DO J = 1, DS
             !       STATE_TEMP(BS:BE,J) = STATE_TEMP(BS:BE,J) * NORM2(STATE_VECS(J,:,I))
             !    END DO
             ! ELSE
             !    DO J = 1, DS
             !       STATE_TEMP(BS:BE,J) = STATE_TEMP(BS:BE,J) * NORM2(OUTPUT_VECS(J,:))
             !    END DO
             ! END IF
             CALL ORTHOGONALIZE(STATE_TEMP(BS:BE,:), LENGTHS(:,BATCH), TER, ORDER(:,BATCH))
             STATE_USAGE(ORDER(:TER,BATCH),BATCH) = 1
             ! Compute grad state rank.
             STATE_TEMP(BS:BE,:) = GRADS(BS:BE,:,I)
             CALL ORTHOGONALIZE(STATE_TEMP(BS:BE,:), LENGTHS(:,BATCH), TGR, ORDER(:,BATCH))
          END DO
          !$OMP END PARALLEL DO
          TOTAL_EVAL_RANK = TOTAL_EVAL_RANK + TER
          TOTAL_GRAD_RANK = TOTAL_GRAD_RANK + TGR
          ! --------------------------------------------------------------------------------
          ! If basis replacement is enabled..
          IF (CONFIG%BASIS_REPLACEMENT) THEN
             ! Sum the "usage" of internal nodes to see which are entirely unuseful.
             STATE_USAGE(:,1) = SUM(STATE_USAGE(:,:), 2)
             ! Replace the basis functions with a policy that ensures convergence.
             IF (I .EQ. 1) THEN
                IF (NS .GT. 1) THEN
                   CALL REPLACE_BASIS_FUNCTIONS( &
                        STATE_USAGE(:,1), &
                        X(:,:), &
                        STATES(:,:,I), &
                        GRADS(:,:,I+1), &
                        INPUT_VECS(:,:),   INPUT_VECS_GRAD_MEAN(:,:),   INPUT_VECS_GRAD_CURV(:,:), &
                        INPUT_SHIFT(:),    INPUT_SHIFT_GRAD_MEAN(:),    INPUT_SHIFT_GRAD_CURV(:),  &
                        STATE_VECS(:,:,I), STATE_VECS_GRAD_MEAN(:,:,I), STATE_VECS_GRAD_CURV(:,:,I))
                ELSE
                   CALL REPLACE_BASIS_FUNCTIONS( &
                        STATE_USAGE(:,1), &
                        X(:,:), &
                        STATES(:,:,I), &
                        Y_GRADIENT(:,:), &
                        INPUT_VECS(:,:),  INPUT_VECS_GRAD_MEAN(:,:),  INPUT_VECS_GRAD_CURV(:,:), &
                        INPUT_SHIFT(:),   INPUT_SHIFT_GRAD_MEAN(:),   INPUT_SHIFT_GRAD_CURV(:),  &
                        OUTPUT_VECS(:,:), OUTPUT_VECS_GRAD_MEAN(:,:), OUTPUT_VECS_GRAD_CURV(:,:))
                END IF
             ELSE IF (I .EQ. NS) THEN
                CALL REPLACE_BASIS_FUNCTIONS( &
                     STATE_USAGE(:,1), &
                     STATES(:,:,I-1), &
                     STATES(:,:,I), &
                     Y_GRADIENT(:,:), &
                     STATE_VECS(:,:,I-1), STATE_VECS_GRAD_MEAN(:,:,I-1), STATE_VECS_GRAD_CURV(:,:,I-1), &
                     STATE_SHIFT(:,I-1),  STATE_SHIFT_GRAD_MEAN(:,I-1),  STATE_SHIFT_GRAD_CURV(:,I-1),  &
                     OUTPUT_VECS(:,:),    OUTPUT_VECS_GRAD_MEAN(:,:),    OUTPUT_VECS_GRAD_CURV(:,:))
             ELSE
                CALL REPLACE_BASIS_FUNCTIONS( &
                     STATE_USAGE(:,1), &
                     STATES(:,:,I-1), &
                     STATES(:,:,I), &
                     GRADS(:,:,I+1), &
                     STATE_VECS(:,:,I-1), STATE_VECS_GRAD_MEAN(:,:,I-1), STATE_VECS_GRAD_CURV(:,:,I-1), &
                     STATE_SHIFT(:,I-1),  STATE_SHIFT_GRAD_MEAN(:,I-1),  STATE_SHIFT_GRAD_CURV(:,I-1),  &
                     STATE_VECS(:,:,I),   STATE_VECS_GRAD_MEAN(:,:,I),   STATE_VECS_GRAD_CURV(:,:,I))
             END IF
          END IF ! END basis replacement
          ! --------------------------------------------------------------------------------
       END DO
    END SUBROUTINE CHECK_MODEL_RANK

    ! Create new basis functions when the total rank of the current
    ! state is not full with the following priorities:
    !   Pick directions that align with the gradient at next state.
    !   Pick directions that are not already captured in this state.
    !   Pick directions that are different from those already captured.
    SUBROUTINE REPLACE_BASIS_FUNCTIONS(USAGE, &
         PREV_STATE, CURR_STATE, NEXT_GRADS, &
         IN_VECS, IN_VECS_GRAD_MEAN, IN_VECS_GRAD_CURV, &
         SHIFTS, SHIFTS_GRAD_MEAN, SHIFTS_GRAD_CURV, &
         OUT_VECS, OUT_VECS_GRAD_MEAN, OUT_VECS_GRAD_CURV)
      INTEGER, INTENT(IN), DIMENSION(:) :: USAGE
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: PREV_STATE
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: CURR_STATE
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: NEXT_GRADS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: IN_VECS, IN_VECS_GRAD_MEAN, IN_VECS_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: SHIFTS, SHIFTS_GRAD_MEAN, SHIFTS_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: OUT_VECS, OUT_VECS_GRAD_MEAN, OUT_VECS_GRAD_CURV
      ! Local variables.
      ! REAL(KIND=RT), DIMENSION(SIZE(USAGE)) :: VALUES ! LOCAL ALLOCATION
      ! REAL(KIND=RT), DIMENSION(SIZE(PREV_STATE,1), SIZE(PREV_STATE,2)) :: PREV_TEMP ! LOCAL ALLOCATION
      ! REAL(KIND=RT), DIMENSION(SIZE(CURR_STATE,1), SIZE(CURR_STATE,2)) :: CURR_TEMP ! LOCAL ALLOCATION
      ! REAL(KIND=RT), DIMENSION(SIZE(NEXT_GRADS,1), SIZE(NEXT_GRADS,2)) :: GRAD_TEMP ! LOCAL ALLOCATION
      ! REAL(KIND=RT), DIMENSION(SIZE(IN_VECS,2), SIZE(IN_VECS,1)) :: VECS_TEMP ! LOCAL ALLOCATION
      ! REAL(KIND=RT), DIMENSION(SIZE(CURR_STATE,2)) :: VECS_TEMP ! LOCAL ALLOCATION
      ! INTEGER, DIMENSION(SIZE(USAGE)) :: ORDER ! LOCAL ALLOCATION
      INTEGER :: RANK, I, GRAD_RANK, MISS_RANK
      ! TODO:
      !  - Multiply value columns by the 2-norm of all outgoing
      !    weights before doing the orthogonalization and ranking.
      ! 
      !  - Set new shift term such that the sum of the gradient is maximized.
      ! 
      !  - Create a function that does LEAST_SQUARES with a truncation factor
      !    that uses the SVD to truncate the number of vectors generated.
      ! 
      ! - When measuring alignment of two vectors come up with way to
      !   quickly find the "most aligned" shift term (the shift that
      !   maximizes the dot product of the vectors assuming rectification).
      ! 
      ! - Update REPLACE_BASIS_FUNCTIONS to:
      !    sum the number of times a component had no rank across threads
      !    (not necessary) swap weights for the no-rank components to the back
      !    (not necessary) swap no-rank state component values into contiguous memory at back
      !    linearly regress the kept-components onto the next-layer dropped difference
      !    compute the first no-rank principal components of the gradient, store in droped slots
      !    regress previous layer onto the gradient components
      !    fill any remaining nodes (if not enough from gradient) with "uncaptured" principal components
      !    set new shift terms as the best of 5 well spaced values in [-1,1], or random given no order

      ! ! Find the first zero-valued (unused) basis function (after orthogonalization).
      ! FORALL (RANK = 1 :SIZE(ORDER(:))) ORDER(RANK) = RANK
      ! VALUES(:) = -REAL(USAGE,RT)
      ! CALL ARGSORT(VALUES(:), ORDER(:))
      ! DO RANK = 1, SIZE(ORDER(:))
      !    IF (USAGE(ORDER(RANK)) .EQ. 0) EXIT
      ! END DO
      ! IF (RANK .GT. SIZE(ORDER)) RETURN

      ! Pack the ORDER(:RANK) nodes into the front of the weights:
      !   - update input weights, mean gradient, gradient curvature
      !   - update input shifts, mean gradient, gradient curvature
      ! Perform a least squares fit of the ORDER(:RANK) nodes to the output values.
      ! If the residual is low, then replace the output weights of the ORDER(:RANK)
      !  nodes and set the other values to zeros.
      ! Reset all gradients to zero and curvatures to zero for the directly affected weights.
      ! 

    END SUBROUTINE REPLACE_BASIS_FUNCTIONS

  END SUBROUTINE CONDITION_MODEL


  ! Fit input / output pairs by minimizing mean squared error.
  SUBROUTINE MINIMIZE_MSE(CONFIG, MODEL, RWORK, IWORK, &
       AX, AXI, SIZES, X, XI, Y, YW, &
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
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW
    INTEGER,       INTENT(IN) :: STEPS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(6,STEPS), OPTIONAL :: RECORD
    REAL(KIND=RT), INTENT(OUT) :: SUM_SQUARED_ERROR
    INTEGER,       INTENT(OUT) :: INFO
    ! Local variables.
    ! ------------------------------------------------------------------
    !                DEVELOPING BATCHED EVALUATION CODE
    !    measured gradient contribution of all input points
    REAL(KIND=RT), DIMENSION(SIZE(Y,2)) :: Y_ERROR
    REAL(KIND=RT), DIMENSION(SIZE(Y,2)) :: Y_REDUCTION
    INTEGER, DIMENSION(SIZE(Y,2)) :: Y_CONSECUTIVE_STEPS
    INTEGER, DIMENSION(SIZE(Y,2)) :: Y_UNUSED_STEPS
    !    count of how many steps have been taken since last usage
    INTEGER, DIMENSION(SIZE(AX,2)) :: AX_UNUSED_STEPS
    INTEGER, DIMENSION(SIZE(X,2)) :: X_UNUSED_STEPS
    !    indices (used for sorting and selecting points for gradient computation)
    INTEGER, DIMENSION(SIZE(AX,2)) :: AX_INDICES
    INTEGER, DIMENSION(SIZE(X,2)) :: X_INDICES
    ! ------------------------------------------------------------------
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
    IF (SIZE(RWORK,KIND=INT64) .LT. CONFIG%RWORK_SIZE) THEN
       INFO = 13
    ELSE IF (SIZE(IWORK,KIND=INT64) .LT. CONFIG%IWORK_SIZE) THEN
       INFO = 14
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (CONFIG%NA .LT. 1)) THEN
       INFO = 15
    ELSE IF ((CONFIG%MDI .GT. 0) .AND. (CONFIG%NM .LT. 1)) THEN
       INFO = 16
    END IF
    ! Do shape checks on the YW (weights for Y's) provided.
    IF (SIZE(YW,2) .NE. SIZE(Y,2)) THEN
       INFO = 17 ! Bad YW number of points.
    ELSE IF ((SIZE(YW,1) .NE. 0) & ! no weights provided
         .AND. (SIZE(YW,1) .NE. 1) & ! one weight per point
         .AND. (SIZE(YW,1) .NE. SIZE(Y,1))) THEN ! one weight per output component
       INFO = 18 ! Bad YW dimension.
    ELSE IF (MINVAL(YW(:,:)) .LE. 0.0_RT) THEN
       INFO = 19 ! Bad YW values.
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
         RWORK(CONFIG%SAYG : CONFIG%EAYG), & ! AY_GRADIENT(NA,ADO)
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
         MODEL(CONFIG%ASIV : CONFIG%AEIV), & ! AGGREGATOR_INPUT_VECS
         MODEL(CONFIG%MSIV : CONFIG%MEIV), & ! MODEL_INPUT_VECS
         MODEL(CONFIG%ASOV : CONFIG%AEOV), & ! AGGREGATOR_OUTPUT_VECS
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
         AY, AY_GRADIENT, X_RESCALE, XI_SHIFT, XI_RESCALE, &
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
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADO) :: AY, AY_GRADIENT
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
      ! Store the start time of this routine (to make sure updates can
      !  be shown to the user at a reasonable frequency).
      CALL SYSTEM_CLOCK(LAST_PRINT_TIME, CLOCK_RATE, CLOCK_MAX)
      WAIT_TIME = CLOCK_RATE * CONFIG%PRINT_DELAY_SEC
      DID_PRINT = .FALSE.
      ! Cap the "number [of variables] to update" at the model size.
      CONFIG%NUM_TO_UPDATE = MAX(1,MIN(CONFIG%NUM_TO_UPDATE, CONFIG%NUM_VARS))
      ! Set the "total rank", the number of internal state components.
      TOTAL_RANK = CONFIG%MDS*CONFIG%MNS + CONFIG%ADS*CONFIG%ANS
      ! Compute the minimum number of model variables to update.
      MIN_TO_UPDATE = MAX(1,INT(CONFIG%MIN_UPDATE_RATIO * REAL(CONFIG%NUM_VARS,RT)))
      ! Set the initial "number of steps taken since best" counter.
      NS = 0
      ! Initial rates of change of mean and variance values.
      STEP_MEAN_REMAIN = 1.0_RT - CONFIG%STEP_MEAN_CHANGE
      STEP_CURV_REMAIN = 1.0_RT - CONFIG%STEP_CURV_CHANGE
      ! Initial mean squared error is "max representable value".
      PREV_MSE = HUGE(PREV_MSE)
      BEST_MSE = HUGE(BEST_MSE)
      ! Disable the application of SHIFT (since data is / will be normalized).
      APPLY_SHIFT = CONFIG%APPLY_SHIFT
      CONFIG%APPLY_SHIFT = .FALSE.
      ! Normalize the data (with parallelization enabled for evaluating AY).
      CALL NORMALIZE_DATA(CONFIG, MODEL, AX, AXI, AY, SIZES, X, XI, Y, YW, &
           AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_RESCALE, X_RESCALE, &
           XI_SHIFT, XI_RESCALE, Y_RESCALE, A_STATES, &
           MODEL(CONFIG%ASEV:CONFIG%AEEV), &
           MODEL(CONFIG%MSEV:CONFIG%MEEV), &
           MODEL(CONFIG%ASOV:CONFIG%AEOV), INFO)
      IF (INFO .NE. 0) RETURN
      ! Set the num batches (NB).
      NB = MIN(CONFIG%NUM_THREADS, SIZE(Y,2))
      CALL COMPUTE_BATCHES(NB, CONFIG%NA, CONFIG%NM, SIZES, &
           BATCHA_STARTS, BATCHA_ENDS, BATCHM_STARTS, BATCHM_ENDS, INFO)
      IF (INFO .NE. 0) THEN
         Y(:,:) = 0.0_RT
         RETURN
      END IF
      ! Set the default size start and end indices for when it is absent.
      IF (SIZE(SIZES) .EQ. 0) THEN
         SS = 1
         SE = -1
      END IF
      ! Set the configured number of threads to 1 to prevent nested parallelization.
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
            ! Sum the gradient over all data batches. If a rank check will be
            !  performed then store the states separate from the gradients.
            !  Otherwise, only compute the gradients and reuse that memory space.
            IF ((CONFIG%RANK_CHECK_FREQUENCY .GT. 0) .AND. &
                 (MOD(STEP-1,CONFIG%RANK_CHECK_FREQUENCY) .EQ. 0)) THEN
               CALL MODEL_GRADIENT(CONFIG, MODEL(:), &
                    AX(:,BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH)), &
                    AXI(:,BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH)), &
                    AY(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:), &
                    SIZES(SS:SE), &
                    X(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    XI(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    Y(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    YW(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    SUM_SQUARED_ERROR, MODEL_GRAD(:,BATCH), INFO, &
                    AY_GRADIENT(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:), &
                    Y_GRADIENT(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    A_GRADS(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:,:), &
                    M_GRADS(BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH),:,:), &
                    A_STATES(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:,:), &
                    M_STATES(BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH),:,:))
            ELSE
               CALL MODEL_GRADIENT(CONFIG, MODEL(:), &
                    AX(:,BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH)), &
                    AXI(:,BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH)), &
                    AY(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:), &
                    SIZES(SS:SE), &
                    X(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    XI(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    Y(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    YW(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    SUM_SQUARED_ERROR, MODEL_GRAD(:,BATCH), INFO, &
                    AY_GRADIENT(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:), &
                    Y_GRADIENT(:,BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH)), &
                    A_GRADS(BATCHA_STARTS(BATCH):BATCHA_ENDS(BATCH),:,:), &
                    M_GRADS(BATCHM_STARTS(BATCH):BATCHM_ENDS(BATCH),:,:))
            END IF
         END DO
         !$OMP END PARALLEL DO
         IF (INFO .NE. 0) RETURN
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
         !              Modify the model variables (take step).
         ! 
         ! Aggregate over computed batches and compute average gradient.
         MODEL_GRAD(:,1) = SUM(MODEL_GRAD(:,:),2) / REAL(NB,RT)
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
         ! Update as many variables as it seems safe to update (and still converge).
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
         !  Project the model parameters back into a safely constrained region.
         ! 
         ! Rescale internal vectors to have a maximum 2-norm of 1.
         ! Center the outputs of the aggregator model about the origin.
         ! Measure the "total rank" of all internal state representations of data.
         CALL CONDITION_MODEL(CONFIG, &
              MODEL(:), MODEL_GRAD_MEAN(:), MODEL_GRAD_CURV(:), & ! Model and gradient.
              AX(:,:), AXI(:,:), AY(:,:), AY_GRADIENT(:,:), & ! Data.
              X(:,:), XI(:,:), Y(:,:), Y_GRADIENT(:,:), &
              NUM_THREADS, CONFIG%STEPS_TAKEN, & ! Configuration for conditioning.
              A_STATES(:,:,:), M_STATES(:,:,:), & ! State values at basis functions.
              A_GRADS(:,:,:), M_GRADS(:,:,:), & ! Gradient values at basis functions.
              A_LENGTHS(:,:), M_LENGTHS(:,:), & ! Work space for orthogonalization.
              A_STATE_TEMP(:,:), M_STATE_TEMP(:,:), & ! Work space for state values.
              A_ORDER(:,:), M_ORDER(:,:), &
              TOTAL_EVAL_RANK, TOTAL_GRAD_RANK)
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
            ! Store the percentage of variables updated in this step.
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


END MODULE AXY

