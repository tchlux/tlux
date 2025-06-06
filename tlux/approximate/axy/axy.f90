! 
! To start linter for Emacs:
!   M-x lsp
!   M-x lsp-diagnostics-mode
!     for some reason this needs to be done for error highlighting
!   C-c ! n
!     next error or warning
!   C-c ! p
!     previous error or warning
!   M-x magit-status
!     look at all changes in current commit
! 
! TODO:
!
! - Add "chords" that enables parallel additive XY models.
!
! - "Dynamic data normalization" default to 4 when batch size is smaller than all data,
!   includes updates to zero mean, unit variance, pca-for-covariance included in the
!   gradient calculation exactly as is done for the AY values.
! 
! - Add incremental checks (and updates?) to the normalization matrices generated from a batch
!   of data, since the normalizations determined might be "bad". Could use a very slow sliding
!   average so that value will only update in the face of large data shifts. Could compute the
!   gradient towards the true PCA by doing one power iteration, as well as the gradient towards
!   componentwise zero mean and unit variance.
! 
! - Track aggregate output state (either in X or Y) for model conditioning steps so that
!   can be use to correctly normalize the values coming out of the aggregate model for
!   componentwise zero mean and unit variance.
! 
! - Add a parameter PARTIAL_STEPS that determines how many steps are included in
!   the partial aggregation process. This can be used to short-circuit doing all
!   possible subset sizes, instead only doing subsets of well-spaced sizes.
!   Specifically, those steps should traverse including all comparisons between
!   increasingly more elements (so progressing as ~x**2).
!
! - Pull normalization code out and have it be called separately from 'FIT'.
!   Goal is to achieve near-zero inefficiencies for doing a few steps at a time in
!   Python (allowing for easier cancellation, progress updates, checkpoints, ...).
!
! - Rotate out the points that have the lowest expected change in error when batching.
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
!
! - Check if OMP TARGET actually sends code to a different device.
! - Experiment with 'OMP TARGET TEAMS DISTRIBUTE PARALLEL' to see if it uses GPU correctly.
!
! - Run some tests to determine how weights should be updated on conditioning
!   to not change the output of the model *at all*, and similarly update the
!   gradient estimates to reflect those changes as well (might have to reset gradient).
!
! - Verify that the *condition model* operation correctly updates the gradient
!   related variables (mean and curvature). (resets back to initialization)
!
! - Make model conditioning use the same work space as evaluation (where possible).
! - Implement and test Fortran native version of GEMM (manual DO loop).
! - Implement and test Fortran native version of SYRK (manual DO loop).
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
  USE ISO_C_BINDING, ONLY: C_BOOL, RT => C_FLOAT, INT32 => C_INT, INT64 => C_LONG
  ! USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, INT64, INT32, INT8
  USE IEEE_ARITHMETIC, ONLY: IS_NAN => IEEE_IS_NAN, IS_FINITE => IEEE_IS_FINITE
  USE RANDOM, ONLY: SEED_RANDOM, RANDOM_REAL, RANDOM_UNIT_VECTORS, &
       INITIALIZE_ITERATOR, INDEX_TO_PAIR, PAIR_TO_INDEX, GET_NEXT_INDEX
  USE SORT_AND_SELECT, ONLY: ARGSORT, ARGSELECT
  USE MATRIX_OPERATIONS, ONLY: GEMM, ORTHONORMALIZE, RADIALIZE

  IMPLICIT NONE

  INTEGER(KIND=INT64) :: CLOCK_RATE, CLOCK_MAX
  INTEGER(KIND=INT64), PARAMETER :: ZERO = 0_INT64
  INTEGER(KIND=INT64), PARAMETER :: ONE = 1_INT64
  INTEGER(KIND=INT64), PARAMETER :: TWO = 2_INT64

  ! Model configuration, internal sizes and fit parameters.
  TYPE, BIND(C) :: MODEL_CONFIG
     ! Aggregator model configuration.
     INTEGER(KIND=INT32) :: ADN     ! aggregator dimension numeric (input)
     INTEGER(KIND=INT32) :: ADE = 0 ! aggregator dimension of embeddings
     INTEGER(KIND=INT32) :: ANE = 0 ! aggregator number of embeddings
     INTEGER(KIND=INT32) :: ADS = 0 ! aggregator dimension of state
     INTEGER(KIND=INT32) :: ANS = 0 ! aggregator number of states
     INTEGER(KIND=INT32) :: ANC = 0 ! aggregator number of chords
     INTEGER(KIND=INT32) :: ADO     ! aggregator dimension of output
     INTEGER(KIND=INT32) :: ADI     ! aggregator dimension of input (internal usage only)
     INTEGER(KIND=INT32) :: ADSO    ! aggregator dimension of state output (internal usage only)
     ! Fixed model configuration.
     INTEGER(KIND=INT32) :: MDN     ! model dimension numeric (input)
     INTEGER(KIND=INT32) :: MDE = 0 ! model dimension of embeddings
     INTEGER(KIND=INT32) :: MNE = 0 ! model number of embeddings
     INTEGER(KIND=INT32) :: MDS = 0 ! model dimension of state
     INTEGER(KIND=INT32) :: MNS = 0 ! model number of states
     INTEGER(KIND=INT32) :: MNC = 0 ! model number of chords
     INTEGER(KIND=INT32) :: MDO     ! model dimension of output
     INTEGER(KIND=INT32) :: MDI     ! model dimension of input (internal usage only)
     INTEGER(KIND=INT32) :: MDSO    ! model dimension of state output (internal usage only)
     ! Output configuration (number of embeddings if there are any).
     INTEGER(KIND=INT32) :: DOE     ! dimension of output embeddings
     INTEGER(KIND=INT32) :: NOE     ! number of output embeddings
     ! Summary numbers that are computed.
     INTEGER(KIND=INT32) :: DON     ! dimension of output numeric values
     INTEGER(KIND=INT32) :: DO      ! dimension output (either MDO or ADO, plus DOE). 
     ! Model descriptors.
     INTEGER(KIND=INT64) :: TOTAL_SIZE
     INTEGER(KIND=INT64) :: NUM_VARS
     ! Index subsets of total size vector naming scheme:
     !   M___ -> model,   A___ -> aggregator
     !   _S__ -> start,   _E__ -> end
     !   __I_ -> input,   __S_ -> states, __O_ -> output, __E_ -> embedding
     !   ___V -> vectors, ___S -> shifts
     INTEGER(KIND=INT64) :: ASEV, AEEV             ! aggregator embedding
     INTEGER(KIND=INT64) :: ASIV, AEIV, ASIS, AEIS ! aggregator input (vec & shift)
     INTEGER(KIND=INT64) :: ASSV, AESV, ASSS, AESS ! aggregator states (vec & shift)
     INTEGER(KIND=INT64) :: ASOV, AEOV             ! aggregator output
     INTEGER(KIND=INT64) :: MSEV, MEEV             ! model embedding
     INTEGER(KIND=INT64) :: MSIV, MEIV, MSIS, MEIS ! model input (vec & shift)
     INTEGER(KIND=INT64) :: MSSV, MESV, MSSS, MESS ! model states (vec & shift)
     INTEGER(KIND=INT64) :: MSOV, MEOV             ! model output
     INTEGER(KIND=INT64) :: OSEV, OEEV             ! output embedding
     ! Index subsets for data normalization.
     !   M___ -> model,  A___ -> aggregator,  O___ -> output
     !   _I__ -> input,  _O__ -> output,      _E__ -> embedding
     !   __S_ -> shift,  __M_ -> multiplier,  __C_ -> center
     !   ___S -> start,  ___E -> end
     INTEGER(KIND=INT64) :: AISS, AISE, AOSS, AOSE, AOMS, AOME
     INTEGER(KIND=INT64) :: AIMS, AIME
     INTEGER(KIND=INT64) :: AECS, AECE
     INTEGER(KIND=INT64) :: MISS, MISE, MOSS, MOSE
     INTEGER(KIND=INT64) :: MIMS, MIME, MOMS, MOME
     INTEGER(KIND=INT64) :: MECS, MECE
     INTEGER(KIND=INT64) :: OECS, OECE
     ! Function parameter.
     REAL(KIND=RT) :: DISCONTINUITY = 0.0_RT
     REAL(KIND=RT) :: CATEGORY_GAP = 0.1_RT
     REAL(KIND=RT) :: MIN_AGG_WEIGHT = SQRT(SQRT(EPSILON(1.0_RT))) ! Minimum weight assigned to aggregate inputs (made convex).
     ! Optimization related parameters.
     REAL(KIND=RT) :: MIN_STEP_FACTOR = 0.0005_RT ! Minimum multiplier on gradient steps.
     REAL(KIND=RT) :: STEP_FACTOR = 0.001_RT ! Initial multiplier on gradient steps.
     REAL(KIND=RT) :: MAX_STEP_FACTOR = 0.01_RT ! Maximum multiplier on gradient steps.
     REAL(KIND=RT) :: MIN_CURV_COMPONENT = EPSILON(1.0_RT) ! Minimum value of any component of the model curvature estimate (SQRT of this number appears in denominator).
     REAL(KIND=RT) :: MAX_CURV_COMPONENT = HUGE(1.0_RT) ! Maximum value of any component of the model curvature estimate (SQRT of this number appears in denominator).
     REAL(KIND=RT) :: MAX_STEP_COMPONENT = SQRT(SQRT(SQRT(SQRT(HUGE(1.0_RT))))) ! Maximum value of any component of the model update step.
     REAL(KIND=RT) :: FASTER_RATE = 1.001_RT ! Rate of increase of optimization factors.
     REAL(KIND=RT) :: SLOWER_RATE = 0.999_RT ! Rate of decrease of optimization factors.
     REAL(KIND=RT) :: MIN_UPDATE_RATIO = 0.5_RT ! Minimum ratio of model variables to update in any optimizaiton step.
     REAL(KIND=RT) :: UPDATE_RATIO_STEP = 0.025_RT ! The step change in ratio of parameters updated when error is decreased.
     REAL(KIND=RT) :: STEP_MEAN_CHANGE = 0.1_RT ! Rate of exponential sliding average over gradient mean estimate.
     REAL(KIND=RT) :: STEP_CURV_CHANGE = 0.01_RT ! Rate of exponential sliding average over gradient curvature estimate.
     REAL(KIND=RT) :: STEP_AY_CHANGE = 0.05_RT ! Rate of exponential sliding average over AY mean & variance (forcing to zero & one).
     REAL(KIND=RT) :: STEP_EMB_CHANGE = 0.05_RT ! Rate of exponential sliding average over A/M embedding mean & variance (forcing to zero & one).
     REAL(KIND=RT) :: INITIAL_CURV_ESTIMATE = 0.0_RT ! Initial estimate used for the curvature term ("magnifies" the first few steps when close to zero).
     REAL(KIND=RT) :: MSE_UPPER_LIMIT = 20.0_RT ! If an MSE greater than this value occurs, a reversion to the "best model" will happen.
     INTEGER(KIND=INT64) :: STEPS_TAKEN = 0 ! Total number of updates already made to model variables.
     INTEGER(KIND=INT64) :: MIN_STEPS_TO_STABILITY = 1 ! Minimum number of steps before allowing model saves and curvature approximation.
     INTEGER(KIND=INT64) :: MAX_BATCH = 10000 ! Max number of points in one batch matrix multiplication.
     INTEGER(KIND=INT64) :: NUM_THREADS = 1 ! Number of parallel threads to use in fit & evaluation.
     INTEGER(KIND=INT64) :: DATA_CONDITION_FREQUENCY = 1 ! Frequency with which to update the conditioning transformations of data (while fitting).
     INTEGER(KIND=INT64) :: MODEL_CONDITION_FREQUENCY = 1 ! Frequency with which to perform model conditioning operations (while fitting).
     INTEGER(KIND=INT64) :: LOG_GRAD_NORM_FREQUENCY = 1 ! Frequency with which to log expensive records (model variable 2-norm step size).
     INTEGER(KIND=INT64) :: RANK_CHECK_FREQUENCY = 0 ! Frequency with which to orthogonalize internal basis functions (0 for "never").
     INTEGER(KIND=INT64) :: NUM_TO_UPDATE = HUGE(ONE) ! Number of model variables to update (initialize to large number).
     INTEGER(KIND=INT64) :: INTERRUPT_DELAY_SEC = 2 ! Delay between interrupts during fit.
     LOGICAL(KIND=C_BOOL) :: BASIS_REPLACEMENT = .FALSE. ! True if linearly dependent basis functions should be replaced during optimization rank checks.
     LOGICAL(KIND=C_BOOL) :: KEEP_BEST = .TRUE. ! True if best observed model should be greedily kept at end of optimization (by MSE if no error checking, otherwise by error check).
     LOGICAL(KIND=C_BOOL) :: EARLY_STOP = .TRUE. ! True if optimization should end when steps since best model is greater than the steps remaining.
     LOGICAL(KIND=C_BOOL) :: RESHUFFLE = .TRUE. ! True if the linear random generator for optimization should be randomized after cycling over all input data.
     ! Aggregation controls.
     LOGICAL(KIND=C_BOOL) :: PAIRWISE_AGGREGATION = .FALSE. ! True if all pairs of aggregate inputs should be considered in evaluation.
     LOGICAL(KIND=C_BOOL) :: PARTIAL_AGGREGATION = .FALSE. ! True if intermediate values (in sum of aggregates) should be passed through fixed model.
     LOGICAL(KIND=C_BOOL) :: ORDERED_AGGREGATION = .FALSE. ! True if the elements of the aggregate set are meaningfully ordered (only relevant for partial aggregation).
     ! Normalization and data handling during FIT_MODEL.
     LOGICAL(KIND=C_BOOL) :: AX_NORMALIZED = .FALSE. ! False if AX data needs to be normalized.
     LOGICAL(KIND=C_BOOL) :: RESCALE_AX = .TRUE. ! Rescale all AX components to be equally weighted.
     LOGICAL(KIND=C_BOOL) :: AXI_NORMALIZED = .FALSE. ! False if AXI embeddings need to be normalized.
     LOGICAL(KIND=C_BOOL) :: AY_NORMALIZED = .FALSE. ! False if aggregator outputs need to be normalized.
     LOGICAL(KIND=C_BOOL) :: X_NORMALIZED = .FALSE. ! False if X data needs to be normalized.
     LOGICAL(KIND=C_BOOL) :: RESCALE_X = .TRUE. ! Rescale all X components to be equally weighted.
     LOGICAL(KIND=C_BOOL) :: XI_NORMALIZED = .FALSE. ! False if XI embeddings need to be normalized.
     LOGICAL(KIND=C_BOOL) :: Y_NORMALIZED = .FALSE. ! False if Y data needs to be normalized.
     LOGICAL(KIND=C_BOOL) :: RESCALE_Y = .TRUE. ! Rescale all Y components to be equally weighted.
     LOGICAL(KIND=C_BOOL) :: YI_NORMALIZED = .FALSE. ! False if YI embeddings need to be normalized.
     LOGICAL(KIND=C_BOOL) :: ENCODE_SCALING = .FALSE. ! True if input and output weight matrices should embed normalization scaling (streamlines, but makes further fitting difficult).
     ! Normalization and data handling during EVALUATE (will be temporarily set to FALSE within FIT_MODEL).
     LOGICAL(KIND=C_BOOL) :: NORMALIZE = .TRUE. ! True if shifting, cleaning, and scaling need to be done to inputs & outputs.
     LOGICAL(KIND=C_BOOL) :: NEEDS_SHIFTING = .TRUE. ! True if shifts need to be applied to inputs.
     LOGICAL(KIND=C_BOOL) :: NEEDS_CLEANING = .TRUE. ! True if NaN and Inf values should be removed.
     LOGICAL(KIND=C_BOOL) :: NEEDS_SCALING = .TRUE. ! True if input and output weight matrices are NOT already rescaled.
     ! Descriptions of the number of points that can be in one batch during the fit.
     INTEGER(KIND=INT64) :: RWORK_SIZE = 0
     INTEGER(KIND=INT64) :: IWORK_SIZE = 0
     INTEGER(KIND=INT64) :: LWORK_SIZE = 0
     INTEGER(KIND=INT64) :: NA = 0 ! Number of aggregate inputs to process at once.
     INTEGER(KIND=INT64) :: NAT = 0 ! TOTAL number of aggregate inputs for model fit.
     INTEGER(KIND=INT64) :: NM = 0 ! Number of fixed model inputs to proces at once.
     INTEGER(KIND=INT64) :: NMS = 0 ! Number of slots for fixed model inputs, used for PARTIAL_AGGREGATION.
     INTEGER(KIND=INT64) :: NMT = 0 ! TOTAL number of fixed model inputs for model fit.
     ! Default linear iterator over data for FIT_MODEL.
     INTEGER(KIND=INT64) :: I_NEXT = 0 
     INTEGER(KIND=INT64) :: I_STEP = 1
     INTEGER(KIND=INT64) :: I_MULT = 1
     INTEGER(KIND=INT64) :: I_MOD = HUGE(1)
     INTEGER(KIND=INT64) :: I_ITER = 0
     ! Real work space (for model optimization).
     INTEGER(KIND=INT64) :: SMG, EMG ! MODEL_GRAD(NUM_VARS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMGM, EMGM ! MODEL_GRAD_MEAN(NUM_VARS)
     INTEGER(KIND=INT64) :: SMGC, EMGC ! MODEL_GRAD_CURV(NUM_VARS)
     INTEGER(KIND=INT64) :: SBM, EBM ! BEST_MODEL(NUM_VARS)
     INTEGER(KIND=INT64) :: SAXB, EAXB ! AX(ADI,NA)
     INTEGER(KIND=INT64) :: SAY, EAY ! AY(NA,ADO+1)
     INTEGER(KIND=INT64) :: SMXB, EMXB ! X(MDI,NMS)
     INTEGER(KIND=INT64) :: SMYB, EMYB ! Y(DON,NMS)
     INTEGER(KIND=INT64) :: SAG, EAG ! AX_GRADIENT(ADI,NA)
     INTEGER(KIND=INT64) :: SAET, EAET ! A_EMB_TEMP(ADE,ANE,NUM_THREADS)
     INTEGER(KIND=INT64) :: SAXS, EAXS ! A_STATES(NA,ADS,ANS+1,ANC)
     INTEGER(KIND=INT64) :: SAXG, EAXG ! A_GRADS(NA,ADS,ANS+1,ANC)
     INTEGER(KIND=INT64) :: SAYG, EAYG ! AY_GRADIENT(NA,ADO+1)
     INTEGER(KIND=INT64) :: SXG, EXG ! X_GRADIENT(MDI,NMS)
     INTEGER(KIND=INT64) :: SMET, EMET ! M_EMB_TEMP(MDE,MNE,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMXS, EMXS ! M_STATES(NMS,MDS,MNS+1,MNC)
     INTEGER(KIND=INT64) :: SMXG, EMXG ! M_GRADS(NMS,MDS,MNS+1,MNC)
     INTEGER(KIND=INT64) :: SOET, EOET ! O_EMB_TEMP(DOE,NOE,NUM_THREADS)
     INTEGER(KIND=INT64) :: SEOS, EEOS ! EMB_OUTS(NOE,NMS)
     INTEGER(KIND=INT64) :: SEOG, EEOG ! EMB_GRADS(NOE,NMS)
     INTEGER(KIND=INT64) :: SYG, EYG ! Y_GRADIENT(DO,NMS)
     INTEGER(KIND=INT64) :: SAXIS, EAXIS ! AXI_SHIFT(ADE)
     INTEGER(KIND=INT64) :: SAXIR, EAXIR ! AXI_RESCALE(ADE,ADE)
     INTEGER(KIND=INT64) :: SMXIS, EMXIS ! XI_SHIFT(MDE)
     INTEGER(KIND=INT64) :: SMXIR, EMXIR ! XI_RESCALE(MDE,MDE)
     INTEGER(KIND=INT64) :: SOXIS, EOXIS ! YI_SHIFT(DOE)
     INTEGER(KIND=INT64) :: SOXIR, EOXIR ! YI_RESCALE(DOE,DOE)
     INTEGER(KIND=INT64) :: SAL, EAL ! A_LENGTHS(ADS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SML, EML ! M_LENGTHS(MDS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SAST, EAST ! A_STATE_TEMP(NA,ADS)
     INTEGER(KIND=INT64) :: SMST, EMST ! M_STATE_TEMP(NMS,MDS)
     ! Integer workspace (for model optimization).
     INTEGER(KIND=INT64) :: SAXI, EAXI ! AXI (aggregate batch indices)
     INTEGER(KIND=INT64) :: SMXI, EMXI ! XI (model batch indices)
     INTEGER(KIND=INT64) :: SOXI, EOXI ! YI (model batch indices)
     INTEGER(KIND=INT64) :: SSB, ESB ! SIZES (sizes for batch)
     INTEGER(KIND=INT64) :: SAO, EAO ! A_ORDER(ADS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMO, EMO ! M_ORDER(MDS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SUI, EUI ! UPDATE_INDICES(1:NUM_VARS)
     ! Integers for counting timers.
     INTEGER(KIND=INT64) :: WINT, CINT ! initialize
     INTEGER(KIND=INT64) :: WFIT, CFIT ! fit (for all minimization steps below)
     INTEGER(KIND=INT64) :: WNRM, CNRM ! normalize
     INTEGER(KIND=INT64) :: WGEN, CGEN ! generate (fetch) data
     INTEGER(KIND=INT64) :: WEMB, CEMB ! embed
     INTEGER(KIND=INT64) :: WEVL, CEVL ! evaluate
     INTEGER(KIND=INT64) :: WGRD, CGRD ! gradient
     INTEGER(KIND=INT64) :: WRAT, CRAT ! rate 
     INTEGER(KIND=INT64) :: WOPT, COPT ! optimize
     INTEGER(KIND=INT64) :: WCON, CCON ! condition
     INTEGER(KIND=INT64) :: WREC, CREC ! record
     INTEGER(KIND=INT64) :: WENC, CENC ! encode
     ! Variables used in the fit process.
     REAL(KIND=RT) :: FIT_MSE
     REAL(KIND=RT) :: FIT_PREV_MSE
     REAL(KIND=RT) :: FIT_BEST_MSE
     REAL(KIND=RT) :: FIT_STEP_MEAN_REMAIN
     REAL(KIND=RT) :: FIT_STEP_CURV_REMAIN
     INTEGER(KIND=INT32) :: FIT_TOTAL_EVAL_RANK
     INTEGER(KIND=INT32) :: FIT_TOTAL_GRAD_RANK
     LOGICAL(KIND=C_BOOL) :: FIT_NORMALIZE
     INTEGER(KIND=INT64) :: FIT_STEP
     INTEGER(KIND=INT64) :: FIT_MIN_TO_UPDATE
     INTEGER(KIND=INT64) :: FIT_LAST_INTERRUPT_TIME
     INTEGER(KIND=INT64) :: FIT_WAIT_TIME
     INTEGER(KIND=INT64) :: FIT_TOTAL_RANK
     INTEGER(KIND=INT64) :: FIT_NS
     INTEGER(KIND=INT64) :: FIT_NT
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

  ! Uncomment the following as a manual replacement in the absence of OpenMP.
  ! FUNCTION OMP_GET_MAX_THREADS()
  !   INTEGER :: OMP_GET_MAX_THREADS
  !   OMP_GET_MAX_THREADS = 1
  ! END FUNCTION OMP_GET_MAX_THREADS
  ! FUNCTION OMP_GET_THREAD_NUM()
  !   INTEGER :: OMP_GET_THREAD_NUM
  !   OMP_GET_THREAD_NUM = 0
  ! END FUNCTION OMP_GET_THREAD_NUM


  ! Wrapper for retrieving a profile entry by name.
  FUNCTION PROFILE(SUBROUTINE_NAME)
    USE PROFILER, ONLY: GET_PROFILE
    USE ISO_C_BINDING, ONLY: REAL64 => C_DOUBLE, INT64 => C_LONG
    ! USE ISO_FORTRAN_ENV, ONLY: REAL64, INT64
    ! Redefine the type for a profile entry here so that fmodpy knows how to return it.
    TYPE, BIND(C) :: PROFILE_ENTRY
       REAL(KIND=REAL64) :: WALL_TIME
       REAL(KIND=REAL64) :: CPU_TIME
       INTEGER(KIND=INT64) :: CALL_COUNT
       REAL(KIND=REAL64) :: START_WALL_TIME
       REAL(KIND=REAL64) :: START_CPU_TIME
    END TYPE PROFILE_ENTRY
    CHARACTER(LEN=*), INTENT(IN) :: SUBROUTINE_NAME
    TYPE(PROFILE_ENTRY) :: PROFILE
    PROFILE = GET_PROFILE(SUBROUTINE_NAME)
  END FUNCTION PROFILE


  ! Generate a model configuration given state parameters for the model.
  SUBROUTINE NEW_MODEL_CONFIG(ADN, ADE, ANE, ADS, ANS, ANC, ADO, &
       MDN, MDE, MNE, MDS, MNS, MNC, MDO, DOE, NOE, NUM_THREADS, CONFIG)
     ! Size related parameters.
     INTEGER(KIND=INT32), INTENT(IN) :: ADN, MDN
     INTEGER(KIND=INT32), INTENT(IN) :: MDO, NOE
     INTEGER(KIND=INT32), OPTIONAL, INTENT(IN) :: ADO, DOE
     INTEGER(KIND=INT32), OPTIONAL, INTENT(IN) :: ADS, MDS
     INTEGER(KIND=INT32), OPTIONAL, INTENT(IN) :: ANS, MNS
     INTEGER(KIND=INT32), OPTIONAL, INTENT(IN) :: ANC, MNC
     INTEGER(KIND=INT32), OPTIONAL, INTENT(IN) :: ANE, MNE
     INTEGER(KIND=INT32), OPTIONAL, INTENT(IN) :: ADE, MDE
     INTEGER(KIND=INT32), OPTIONAL, INTENT(IN) :: NUM_THREADS
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
     ! ANC
     IF (CONFIG%ADI .EQ. 0) THEN
        CONFIG%ANC = 0 ! No chords if there are no inputs.
     ELSE IF (CONFIG%ANS .EQ. 0) THEN
        CONFIG%ANC = 1 ! Only one chord if this is a linear model.
     ELSE IF (PRESENT(ANC)) THEN
        CONFIG%ANC = ANC ! Otherwise set provided value.
     ELSE
        CONFIG%ANC = 1 ! None provided suggest simplest form, one chord.
     END IF
     ! ADS
     IF (PRESENT(ADS)) THEN
        CONFIG%ADS = ADS
     ELSE IF (CONFIG%ANS .GT. 0) THEN
        CONFIG%ADS = 32
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
        CONFIG%ADSO = 0
        CONFIG%ADO = 0
     ELSE IF (MIN(CONFIG%ANS,CONFIG%ADS) .EQ. 0) THEN
        CONFIG%ADO = MIN(16, CONFIG%ADI)
     ELSE
        CONFIG%ADO = MIN(16, CONFIG%ADS)
     END IF
     ! No aggregate model.
     IF (CONFIG%ADI .EQ. 0) THEN
        CONFIG%ADN = 0
        CONFIG%ADE = 0
        CONFIG%ANE = 0
        CONFIG%ADS = 0
        CONFIG%ANS = 0
        CONFIG%ANC = 0
        CONFIG%ADO = 0
        CONFIG%ADSO = 0
     END IF
     ! ---------------------------------------------------------------
     ! MNE
     IF (PRESENT(MNE)) CONFIG%MNE = MNE
     ! MDE
     IF (PRESENT(MDE)) THEN
        CONFIG%MDE = MDE
     ELSE IF (CONFIG%MNE .GT. 0) THEN
        ! Compute a reasonable default dimension (tied to volume of space).
        CONFIG%MDE = MAX(1, 2 * CEILING(LOG(REAL(CONFIG%MNE,RT)) / LOG(2.0_RT)))
        IF (CONFIG%MNE .LE. 3) CONFIG%MDE = CONFIG%MDE - 1
     END IF
     ! MDN, MDI
     CONFIG%MDN = MDN
     CONFIG%MDI = CONFIG%MDN + CONFIG%MDE + CONFIG%ADO
     ! MNS
     IF (PRESENT(MNS)) CONFIG%MNS = MNS
     ! MNC
     IF (CONFIG%MNS .EQ. 0) THEN
        CONFIG%MNC = 1 ! Only one chord if this is a linear model.
     ELSE IF (PRESENT(MNC)) THEN
        CONFIG%MNC = MNC ! Use provided number of chords.
     ELSE
        CONFIG%MNC = 1 ! Default to simplest model with one chord.
     END IF
     ! MDS
     IF (PRESENT(MDS)) THEN
        CONFIG%MDS = MDS
     ELSE IF (CONFIG%MNS .GT. 0) THEN
        CONFIG%MDS = 32
     END IF
     CONFIG%MDSO = CONFIG%MDS
     IF (CONFIG%MNS .EQ. 0) THEN
        CONFIG%MDS = 0
        CONFIG%MDSO = CONFIG%MDI
     END IF
     ! DOE, NOE (output embeddings)
     CONFIG%NOE = NOE
     IF (PRESENT(DOE)) THEN
        CONFIG%DOE = DOE
     ELSE IF (CONFIG%NOE .GT. 0) THEN
        ! Compute a reasonable default dimension (tied to volume of space).
        CONFIG%DOE = MAX(1, 2 * CEILING(LOG(REAL(CONFIG%NOE,RT)) / LOG(2.0_RT)))
        IF (CONFIG%NOE .LE. 3) CONFIG%DOE = CONFIG%DOE - 1
     ELSE
        ! There is no output embedding.
        CONFIG%DOE = ZERO
     END IF
     ! MDO  (set to '-1' to disable model, 
     CONFIG%MDO = MDO
     ! No fixed model.
     IF (CONFIG%MDO .LT. 0) THEN
        CONFIG%MDI = 0
        CONFIG%MDE = 0
        CONFIG%MNE = 0
        CONFIG%MDS = 0
        CONFIG%MNS = 0
        CONFIG%MNC = 0
        CONFIG%MDSO = 0
     END IF
     ! DO
     IF (CONFIG%MDO .GE. ZERO) THEN
        CONFIG%MDO = CONFIG%MDO + CONFIG%DOE
        CONFIG%DO = CONFIG%MDO
     ELSE
        CONFIG%ADO = CONFIG%ADO + CONFIG%DOE
        CONFIG%DO = CONFIG%ADO
     END IF
     CONFIG%DON = CONFIG%DO - CONFIG%DOE
     ! ---------------------------------------------------------------
     ! NUM_THREADS
     IF (PRESENT(NUM_THREADS)) THEN
        CONFIG%NUM_THREADS = MIN(NUM_THREADS, OMP_GET_MAX_THREADS())
     ELSE
        CONFIG%NUM_THREADS = OMP_GET_MAX_THREADS()
     END IF
     ! Compute indices related to the variable locations for this model.
     CONFIG%TOTAL_SIZE = 0
     ! ---------------------------------------------------------------
     !   aggregator embedding vecs [ADE by ANE]
     CONFIG%ASEV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AEEV = CONFIG%ASEV-ONE +  CONFIG%ADE * CONFIG%ANE
     CONFIG%TOTAL_SIZE = CONFIG%AEEV
     !   aggregator input vecs [ADI by ADS by ANC]
     CONFIG%ASIV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AEIV = CONFIG%ASIV-ONE +  CONFIG%ADI * CONFIG%ADS * CONFIG%ANC
     CONFIG%TOTAL_SIZE = CONFIG%AEIV
     !   aggregator input shift [ADS by ANC]
     CONFIG%ASIS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AEIS = CONFIG%ASIS-ONE +  CONFIG%ADS * CONFIG%ANC
     CONFIG%TOTAL_SIZE = CONFIG%AEIS
     !   aggregator state vecs [ADS by ADS by MAX(0,ANS-1) by ANC]
     CONFIG%ASSV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AESV = CONFIG%ASSV-ONE +  CONFIG%ADS * CONFIG%ADS * MAX(ZERO,CONFIG%ANS-ONE) * CONFIG%ANC
     CONFIG%TOTAL_SIZE = CONFIG%AESV
     !   aggregator state shift [ADS by MAX(0,ANS-1) by ANC]
     CONFIG%ASSS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AESS = CONFIG%ASSS-ONE +  CONFIG%ADS * MAX(ZERO,CONFIG%ANS-ONE) * CONFIG%ANC
     CONFIG%TOTAL_SIZE = CONFIG%AESS
     !   aggregator output vecs [ADSO by ADO+1 by ANC]
     CONFIG%ASOV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AEOV = CONFIG%ASOV-ONE +  CONFIG%ADSO * (CONFIG%ADO+ONE) * CONFIG%ANC
     CONFIG%TOTAL_SIZE = CONFIG%AEOV
     ! ---------------------------------------------------------------
     !   model embedding vecs [MDE by MNE]
     CONFIG%MSEV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MEEV = CONFIG%MSEV-ONE +  CONFIG%MDE * CONFIG%MNE
     CONFIG%TOTAL_SIZE = CONFIG%MEEV
     !   model input vecs [MDI by MDS by MNC]
     CONFIG%MSIV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MEIV = CONFIG%MSIV-ONE +  CONFIG%MDI * CONFIG%MDS * CONFIG%MNC
     CONFIG%TOTAL_SIZE = CONFIG%MEIV
     !   model input shift [MDS by MNC]
     CONFIG%MSIS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MEIS = CONFIG%MSIS-ONE +  CONFIG%MDS * CONFIG%MNC
     CONFIG%TOTAL_SIZE = CONFIG%MEIS
     !   model state vecs [MDS by MDS by MAX(0,MNS-1) by MNC]
     CONFIG%MSSV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MESV = CONFIG%MSSV-ONE +  CONFIG%MDS * CONFIG%MDS * MAX(ZERO,CONFIG%MNS-ONE) * CONFIG%MNC
     CONFIG%TOTAL_SIZE = CONFIG%MESV
     !   model state shift [MDS by MAX(0,MNS-1) by MNC]
     CONFIG%MSSS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MESS = CONFIG%MSSS-ONE +  CONFIG%MDS * MAX(ZERO,CONFIG%MNS-ONE) * CONFIG%MNC
     CONFIG%TOTAL_SIZE = CONFIG%MESS
     !   model output vecs [MDSO by MDO by MNC]
     CONFIG%MSOV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MEOV = CONFIG%MSOV-ONE +  CONFIG%MDSO * CONFIG%MDO * CONFIG%MNC
     CONFIG%TOTAL_SIZE = CONFIG%MEOV
     ! ---------------------------------------------------------------
     !   output embedding vecs [DOE by NOE]
     CONFIG%OSEV = ONE + CONFIG%TOTAL_SIZE
     CONFIG%OEEV = CONFIG%OSEV-ONE +  CONFIG%DOE * CONFIG%NOE
     CONFIG%TOTAL_SIZE = CONFIG%OEEV
     ! ---------------------------------------------------------------
     !   number of variables
     CONFIG%NUM_VARS = CONFIG%TOTAL_SIZE
     ! ---------------------------------------------------------------
     !   aggregator post-output shift [ADO]
     CONFIG%AOSS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AOSE = CONFIG%AOSS-ONE + CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AOSE
     !   aggregator post-output multiplier [ADO]
     CONFIG%AOMS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AOME = CONFIG%AOMS-ONE + CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AOME
     !   aggregator embedding center [ADE]
     CONFIG%AECS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AECE = CONFIG%AECS-ONE + CONFIG%ADE
     CONFIG%TOTAL_SIZE = CONFIG%AECE
     !   model embedding center [MDE]
     CONFIG%MECS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MECE = CONFIG%MECS-ONE + CONFIG%MDE
     CONFIG%TOTAL_SIZE = CONFIG%MECE
     !   output embedding center [MDE]
     CONFIG%OECS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%OECE = CONFIG%OECS-ONE + CONFIG%DOE
     CONFIG%TOTAL_SIZE = CONFIG%OECE
     ! ---------------------------------------------------------------
     !   aggregator pre-input shift
     CONFIG%AISS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AISE = CONFIG%AISS-ONE + CONFIG%ADN
     CONFIG%TOTAL_SIZE = CONFIG%AISE
     !   aggregator pre-input multiplier
     CONFIG%AIMS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%AIME = CONFIG%AIMS-ONE + CONFIG%ADN * CONFIG%ADN
     CONFIG%TOTAL_SIZE = CONFIG%AIME
     !   model pre-input shift
     CONFIG%MISS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MISE = CONFIG%MISS-ONE + CONFIG%MDN
     CONFIG%TOTAL_SIZE = CONFIG%MISE
     !   model pre-input multiplier
     CONFIG%MIMS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MIME = CONFIG%MIMS-ONE + CONFIG%MDN * CONFIG%MDN
     CONFIG%TOTAL_SIZE = CONFIG%MIME
     !   model post-output shift
     CONFIG%MOSS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MOSE = CONFIG%MOSS-ONE + CONFIG%DON
     CONFIG%TOTAL_SIZE = CONFIG%MOSE
     !   model post-output multiplier
     CONFIG%MOMS = ONE + CONFIG%TOTAL_SIZE
     CONFIG%MOME = CONFIG%MOMS-ONE + CONFIG%DON ** TWO
     CONFIG%TOTAL_SIZE = CONFIG%MOME
     ! ---------------------------------------------------------------
     !   set all time counters to zero
     CONFIG%WFIT = ZERO
     CONFIG%CFIT = ZERO
     CONFIG%WNRM = ZERO
     CONFIG%CNRM = ZERO
     CONFIG%WGEN = ZERO
     CONFIG%CGEN = ZERO
     CONFIG%WEMB = ZERO
     CONFIG%CEMB = ZERO
     CONFIG%WEVL = ZERO
     CONFIG%CEVL = ZERO
     CONFIG%WGRD = ZERO
     CONFIG%CGRD = ZERO
     CONFIG%WRAT = ZERO
     CONFIG%CRAT = ZERO
     CONFIG%WOPT = ZERO
     CONFIG%COPT = ZERO
     CONFIG%WCON = ZERO
     CONFIG%CCON = ZERO
     CONFIG%WREC = ZERO
     CONFIG%CREC = ZERO
     CONFIG%WENC = ZERO
     CONFIG%CENC = ZERO
  END SUBROUTINE NEW_MODEL_CONFIG

  ! Given a number of X points "NM", and a number of aggregator X points
  ! "NA", update the "RWORK_SIZE" and "IWORK_SIZE" attributes in "CONFIG"
  ! as well as all related work indices for that size data. Optionally
  ! also provide "NAT" and "NMT" the 'total' number of aggregate and
  ! fixed points respectively.
  SUBROUTINE NEW_FIT_CONFIG(NM, NA, NMT, NAT, ADI, MDI, ODI, SEED, CONFIG)
    INTEGER(KIND=INT64), INTENT(IN) :: NM
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: NA, NMT, NAT, ADI, MDI, ODI
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: SEED
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    INTEGER(KIND=INT64) :: AXI_COLS, XI_COLS, YI_COLS
    ! NM and NMT (total)
    CONFIG%NM = NM
    IF (PRESENT(NMT)) THEN
       CONFIG%NMT = NMT
    ELSE
       CONFIG%NMT = NM
    END IF
    CONFIG%NM = MIN(CONFIG%NMT, CONFIG%NM)
    ! NA
    IF (PRESENT(NA) .AND. (CONFIG%ADO .GT. 0)) THEN
       CONFIG%NA = NA
    ELSE
       CONFIG%NA = 0
    END IF
    ! NAT (total)
    IF (PRESENT(NAT) .AND. (CONFIG%ADO .GT. 0)) THEN
       CONFIG%NAT = MIN(NAT, CONFIG%NA)
    ELSE
       CONFIG%NAT = CONFIG%NA
    END IF
    ! For partial aggregation, the number of slots for batching must be larger to include partial sums.
    IF (CONFIG%PARTIAL_AGGREGATION) THEN
       CONFIG%NMS = MAX(CONFIG%NM, CONFIG%NA + CONFIG%NM-ONE)  ! Maxed out when all aggregates belong to one input, the rest have 0.
    ELSE
       CONFIG%NMS = CONFIG%NM
    END IF
    ! ADI (not the literal value, but the number of categorical columns)
    IF (PRESENT(ADI)) THEN
       AXI_COLS = ADI
    ELSE
       AXI_COLS = ZERO
    END IF
    ! MDI (not the literal value, but the number of categorical columns)
    IF (PRESENT(MDI)) THEN
       XI_COLS = MDI
    ELSE
       XI_COLS = ZERO
    END IF
    ! ODI (not the literal value, but the number of categorical columns)
    IF (PRESENT(ODI)) THEN
       YI_COLS = ODI
    ELSE
       YI_COLS = ZERO
    END IF
    ! ------------------------------------------------------------
    ! Set up the indexing for batch iteration.
    CALL INITIALIZE_ITERATOR( &
         CONFIG%NMT, CONFIG%I_NEXT, CONFIG%I_MULT, &
         CONFIG%I_STEP, CONFIG%I_MOD, CONFIG%I_ITER, SEED=SEED)
    ! ------------------------------------------------------------
    ! Set up the real valued work array.
    CONFIG%RWORK_SIZE = 0
    ! model gradient
    CONFIG%SMG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMG = CONFIG%SMG-ONE + CONFIG%NUM_VARS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EMG
    ! model gradient mean
    CONFIG%SMGM = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMGM = CONFIG%SMGM-ONE + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EMGM
    ! model gradient curvature
    CONFIG%SMGC = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMGC = CONFIG%SMGC-ONE + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EMGC
    ! best model
    CONFIG%SBM = ONE + CONFIG%RWORK_SIZE
    CONFIG%EBM = CONFIG%SBM-ONE + CONFIG%TOTAL_SIZE
    CONFIG%RWORK_SIZE = CONFIG%EBM
    ! ---------------------------------------------------------------
    ! AX
    CONFIG%SAXB = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAXB = CONFIG%SAXB-ONE + CONFIG%ADI * CONFIG%NA
    CONFIG%RWORK_SIZE = CONFIG%EAXB
    ! AX gradient
    CONFIG%SAG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAG = CONFIG%SAG-ONE + CONFIG%ADI * CONFIG%NA
    CONFIG%RWORK_SIZE = CONFIG%EAG
    ! A embedding temp holder
    CONFIG%SAET = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAET = CONFIG%SAET-ONE + CONFIG%ADE * CONFIG%ANE * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EAET
    ! aggregator states
    CONFIG%SAXS = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAXS = CONFIG%SAXS-ONE + CONFIG%NA * CONFIG%ADS * (CONFIG%ANS+ONE) * CONFIG%ANC
    CONFIG%RWORK_SIZE = CONFIG%EAXS
    ! aggregator gradients at states
    CONFIG%SAXG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAXG = CONFIG%SAXG-ONE + CONFIG%NA * CONFIG%ADS * (CONFIG%ANS+ONE) * CONFIG%ANC
    CONFIG%RWORK_SIZE = CONFIG%EAXG
    ! AY
    CONFIG%SAY = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAY = CONFIG%SAY-ONE + CONFIG%NA * (CONFIG%ADO+ONE)
    CONFIG%RWORK_SIZE = CONFIG%EAY
    ! AY gradient
    CONFIG%SAYG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAYG = CONFIG%SAYG-ONE + CONFIG%NA * (CONFIG%ADO+ONE)
    CONFIG%RWORK_SIZE = CONFIG%EAYG
    ! X
    CONFIG%SMXB = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMXB = CONFIG%SMXB-ONE + CONFIG%MDI * CONFIG%NMS
    CONFIG%RWORK_SIZE = CONFIG%EMXB
    ! X gradient
    CONFIG%SXG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EXG = CONFIG%SXG-ONE + CONFIG%MDI * CONFIG%NMS
    CONFIG%RWORK_SIZE = CONFIG%EXG
    ! M embedding temp holder
    CONFIG%SMET = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMET = CONFIG%SMET-ONE + CONFIG%MDE * CONFIG%MNE * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EMET
    ! model states
    CONFIG%SMXS = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMXS = CONFIG%SMXS-ONE + CONFIG%NMS * CONFIG%MDS * (CONFIG%MNS+ONE) * CONFIG%MNC
    CONFIG%RWORK_SIZE = CONFIG%EMXS
    ! model gradients at states
    CONFIG%SMXG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMXG = CONFIG%SMXG-ONE + CONFIG%NMS * CONFIG%MDS * (CONFIG%MNS+ONE) * CONFIG%MNC
    CONFIG%RWORK_SIZE = CONFIG%EMXG
    ! O embedding temp holder
    CONFIG%SOET = ONE + CONFIG%RWORK_SIZE
    CONFIG%EOET = CONFIG%SOET-ONE + CONFIG%DOE * CONFIG%NOE * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EOET
    ! Embedding outputs holder
    CONFIG%SEOS = ONE + CONFIG%RWORK_SIZE
    CONFIG%EEOS = CONFIG%SEOS-ONE + CONFIG%NOE * CONFIG%NMS
    CONFIG%RWORK_SIZE = CONFIG%EEOS
    ! Embedding gradients holder
    CONFIG%SEOG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EEOG = CONFIG%SEOG-ONE + CONFIG%NOE * CONFIG%NMS
    CONFIG%RWORK_SIZE = CONFIG%EEOG
    ! Y
    CONFIG%SMYB = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMYB = CONFIG%SMYB-ONE + CONFIG%DO * CONFIG%NMS
    CONFIG%RWORK_SIZE = CONFIG%EMYB
    ! Y gradient
    CONFIG%SYG = ONE + CONFIG%RWORK_SIZE
    CONFIG%EYG = CONFIG%SYG-ONE + CONFIG%DO * CONFIG%NMS
    CONFIG%RWORK_SIZE = CONFIG%EYG
    ! AXI shift
    CONFIG%SAXIS = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAXIS = CONFIG%SAXIS-ONE + CONFIG%ADE
    CONFIG%RWORK_SIZE = CONFIG%EAXIS
    ! AXI rescale
    CONFIG%SAXIR = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAXIR = CONFIG%SAXIR-ONE + CONFIG%ADE * CONFIG%ADE
    CONFIG%RWORK_SIZE = CONFIG%EAXIR
    ! XI shift
    CONFIG%SMXIS = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMXIS = CONFIG%SMXIS-ONE + CONFIG%MDE
    CONFIG%RWORK_SIZE = CONFIG%EMXIS
    ! XI rescale
    CONFIG%SMXIR = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMXIR = CONFIG%SMXIR-ONE + CONFIG%MDE * CONFIG%MDE
    CONFIG%RWORK_SIZE = CONFIG%EMXIR
    ! YI shift
    CONFIG%SOXIS = ONE + CONFIG%RWORK_SIZE
    CONFIG%EOXIS = CONFIG%SOXIS-ONE + CONFIG%DOE
    CONFIG%RWORK_SIZE = CONFIG%EOXIS
    ! YI rescale
    CONFIG%SOXIR = ONE + CONFIG%RWORK_SIZE
    CONFIG%EOXIR = CONFIG%SOXIR-ONE + CONFIG%DOE * CONFIG%DOE
    CONFIG%RWORK_SIZE = CONFIG%EOXIR
    ! A lengths (lengths of state values after orthogonalization)
    CONFIG%SAL = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAL = CONFIG%SAL-ONE + CONFIG%ADS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EAL
    ! M lengths (lengths of state values after orthogonalization)
    CONFIG%SML = ONE + CONFIG%RWORK_SIZE
    CONFIG%EML = CONFIG%SML-ONE + CONFIG%MDS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EML
    ! A state temp holder (for orthogonality computation)
    CONFIG%SAST = ONE + CONFIG%RWORK_SIZE
    CONFIG%EAST = CONFIG%SAST-ONE + CONFIG%NA * CONFIG%ADS
    CONFIG%RWORK_SIZE = CONFIG%EAST
    ! M state temp holder (for orthogonality computation)
    CONFIG%SMST = ONE + CONFIG%RWORK_SIZE
    CONFIG%EMST = CONFIG%SMST-ONE + CONFIG%NMS * CONFIG%MDS
    CONFIG%RWORK_SIZE = CONFIG%EMST
    ! ------------------------------------------------------------
    ! Set up the integer valued work array.
    CONFIG%IWORK_SIZE = ZERO
    ! A order (for orthogonalization)
    CONFIG%SAO = ONE + CONFIG%IWORK_SIZE
    CONFIG%EAO = CONFIG%SAO-ONE + CONFIG%ADS * CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EAO
    ! M order (for orthogonalization)
    CONFIG%SMO = ONE + CONFIG%IWORK_SIZE
    CONFIG%EMO = CONFIG%SMO-ONE + CONFIG%MDS * CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EMO
    ! Set up long-integer valued work array.
    CONFIG%LWORK_SIZE = ZERO
    ! AXI
    CONFIG%SAXI = ONE + CONFIG%LWORK_SIZE
    CONFIG%EAXI = CONFIG%SAXI-ONE + AXI_COLS * CONFIG%NA
    CONFIG%LWORK_SIZE = CONFIG%EAXI
    ! XI
    CONFIG%SMXI = ONE + CONFIG%LWORK_SIZE
    CONFIG%EMXI = CONFIG%SMXI-ONE + XI_COLS * CONFIG%NMS
    CONFIG%LWORK_SIZE = CONFIG%EMXI
    ! YI
    CONFIG%SOXI = ONE + CONFIG%LWORK_SIZE
    CONFIG%EOXI = CONFIG%SOXI-ONE + YI_COLS * CONFIG%NMS
    CONFIG%LWORK_SIZE = CONFIG%EOXI
    ! SIZES
    CONFIG%SSB = ONE + CONFIG%LWORK_SIZE
    IF (CONFIG%ADI .GT. 0) THEN
       CONFIG%ESB = CONFIG%SSB-ONE + CONFIG%NM
    ELSE
       CONFIG%ESB = CONFIG%SSB-ONE + ZERO
    END IF
    CONFIG%LWORK_SIZE = CONFIG%ESB
    ! UPDATE_INDICES
    CONFIG%SUI = ONE + CONFIG%LWORK_SIZE
    CONFIG%EUI = CONFIG%SUI + CONFIG%NUM_VARS-ONE
    CONFIG%LWORK_SIZE = CONFIG%EUI
  END SUBROUTINE NEW_FIT_CONFIG

  ! Initialize the weights for a model, optionally provide a random seed.
  SUBROUTINE INIT_MODEL(CONFIG, MODEL, SEED, INITIAL_SHIFT_RANGE, INITIAL_OUTPUT_SCALE)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: SEED
    REAL(KIND=RT), INTENT(IN), OPTIONAL :: INITIAL_SHIFT_RANGE, INITIAL_OUTPUT_SCALE
    REAL(KIND=RT) :: SHIFT_RANGE, OUTPUT_SCALE
    ! Local iterator.
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Set optional arguments (if provided, else set default values).
    IF (PRESENT(INITIAL_SHIFT_RANGE)) THEN
       SHIFT_RANGE = INITIAL_SHIFT_RANGE
    ELSE
       SHIFT_RANGE = 1.0_RT
    END IF
    ! Set a random seed, if one was provided (otherwise leave default).
    IF (PRESENT(SEED)) THEN
       CALL SEED_RANDOM(SEED)
    END IF
    ! Initialize the aggregator model.
    !   If there is no model afterwards, use output scaling, otherwise use 1.
    IF (CONFIG%MDO .LT. 0) THEN
       IF (PRESENT(INITIAL_OUTPUT_SCALE)) THEN
          OUTPUT_SCALE = INITIAL_OUTPUT_SCALE
       ELSE
          OUTPUT_SCALE = 0.1_RT
       END IF
    ELSE
       OUTPUT_SCALE = 1.0_RT
    END IF
    ! Initialize the aggregate model embeddings.
    CALL INIT_EMBEDDINGS(CONFIG%ADE, CONFIG%ANE, &
         MODEL(CONFIG%ASEV:CONFIG%AEEV), &
         MODEL(CONFIG%AECS:CONFIG%AECE), &
         RANDOM_LENGTHS=.TRUE._C_BOOL)
    ! Initialize the aggregator model.
    CALL INIT_SUBMODEL(&
         CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ANC, &
         CONFIG%ADSO, CONFIG%ADO+1, &
         MODEL(CONFIG%ASIV:CONFIG%AEIV), &
         MODEL(CONFIG%ASIS:CONFIG%AEIS), &
         MODEL(CONFIG%ASSV:CONFIG%AESV), &
         MODEL(CONFIG%ASSS:CONFIG%AESS), &
         MODEL(CONFIG%ASOV:CONFIG%AEOV))
    !   Set up the output scaling for the fixed model (won't matter if it doesn't exist).
    IF (PRESENT(INITIAL_OUTPUT_SCALE)) THEN
       OUTPUT_SCALE = INITIAL_OUTPUT_SCALE
    ELSE
       OUTPUT_SCALE = 0.1_RT
    END IF
    ! Initialize the fixed model embeddings.
    CALL INIT_EMBEDDINGS(CONFIG%MDE, CONFIG%MNE, &
         MODEL(CONFIG%MSEV:CONFIG%MEEV), &
         MODEL(CONFIG%MECS:CONFIG%MECE), &
         RANDOM_LENGTHS=.TRUE._C_BOOL)
    ! Initialize the fixed model.
    CALL INIT_SUBMODEL(&
         CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MNC, &
         CONFIG%MDSO, CONFIG%MDO, &
         MODEL(CONFIG%MSIV:CONFIG%MEIV), &
         MODEL(CONFIG%MSIS:CONFIG%MEIS), &
         MODEL(CONFIG%MSSV:CONFIG%MESV), &
         MODEL(CONFIG%MSSS:CONFIG%MESS), &
         MODEL(CONFIG%MSOV:CONFIG%MEOV))
    ! Initialize the output embeddings.
    CALL INIT_EMBEDDINGS(CONFIG%DOE, CONFIG%NOE, &
         MODEL(CONFIG%OSEV:CONFIG%OEEV), &
         MODEL(CONFIG%OECS:CONFIG%OECE), &
         RANDOM_LENGTHS=.FALSE._C_BOOL)
    ! ---------------------------------------------------------------------
    ! Set the normalization shifts to zero and multipliers to the identity.
    !   aggregator input shift,
    MODEL(CONFIG%AISS:CONFIG%AISE) = 0.0_RT    
    !   aggregator input multiplier,
    MODEL(CONFIG%AIMS:CONFIG%AIME) = 0.0_RT
    MODEL(CONFIG%AIMS:CONFIG%AIME:CONFIG%ADN+1) = 1.0_RT
    !   fixed input shift,
    MODEL(CONFIG%MISS:CONFIG%MISE) = 0.0_RT    
    !   fixed input multiplier,
    MODEL(CONFIG%MIMS:CONFIG%MIME) = 0.0_RT
    MODEL(CONFIG%MIMS:CONFIG%MIME:CONFIG%MDN+1) = 1.0_RT
    !   output shift, and
    MODEL(CONFIG%MOSS:CONFIG%MOSE) = 0.0_RT
    !   output multiplier.
    MODEL(CONFIG%MOMS:CONFIG%MOME) = 0.0_RT
    MODEL(CONFIG%MOMS:CONFIG%MOME:CONFIG%DO+1) = 1.0_RT
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WINT = CONFIG%WINT + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CINT = CONFIG%CINT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    ! Unseed the random number generator if it was seeded.
    IF (PRESENT(SEED)) THEN
       CALL SEED_RANDOM()
    END IF


  CONTAINS
    ! Initialize embeddings.
    SUBROUTINE INIT_EMBEDDINGS(DE, NE, EMBEDDINGS, EMBEDDINGS_MEAN, RANDOM_LENGTHS)
      INTEGER(KIND=INT32), INTENT(IN) :: DE, NE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DE, NE) :: EMBEDDINGS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DE) :: EMBEDDINGS_MEAN
      LOGICAL(KIND=C_BOOL), INTENT(IN), OPTIONAL :: RANDOM_LENGTHS
      REAL(KIND=RT) :: D, R
      INTEGER(KIND=INT64) :: I
      CALL RANDOM_UNIT_VECTORS(EMBEDDINGS(:,:))
      EMBEDDINGS_MEAN(:) = 0.0_RT
      ! If random length was provided and they are desired..
      IF (PRESENT(RANDOM_LENGTHS)) THEN
         IF (RANDOM_LENGTHS) THEN
            ! Multiply the embeddings by random lengths to make them better spaced,
            !  specifically, try to make them uniformly distributed in the ball.
            D = 1.0_RT / REAL(DE, RT)
            DO I = 1, NE
               CALL RANDOM_REAL(V=R)
               EMBEDDINGS(:,I) = EMBEDDINGS(:,I) * R**D
            END DO
         END IF
      END IF
    END SUBROUTINE INIT_EMBEDDINGS

    ! Initialize the model after unpacking it into its constituent parts.
    SUBROUTINE INIT_SUBMODEL(MDI, MDS, MNS, MNC, MDSO, MDO, &
         INPUT_VECS, INPUT_SHIFT, STATE_VECS, STATE_SHIFT, &
         OUTPUT_VECS)
      INTEGER(KIND=INT32), INTENT(IN) :: MDI, MDS, MNS, MNC, MDSO, MDO
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI, MDS, MNC) :: INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS, MNC) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS, MDS, MAX(ZERO,MNS-ONE), MNC) :: STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS, MAX(ZERO,MNS-ONE), MNC) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDSO, MDO, MNC) :: OUTPUT_VECS
      ! Local holder for "origin" at each layer.
      REAL(KIND=RT), ALLOCATABLE, DIMENSION(:) :: ORIGIN ! LOCAL ALLOCATION
      INTEGER(KIND=INT64), ALLOCATABLE, DIMENSION(:) :: ORDER  ! LOCAL ALLOCATION
      INTEGER(KIND=INT64) :: I, J, C
      ! Allocate local variables.
      ALLOCATE(ORIGIN(1:MDS), ORDER(1:MDS))
      ! Iterate over chords (individual models that are summed).
      DO C = 1, MNC
         ! Generate well spaced random unit-length vectors (no scaling biases)
         ! for all initial variables in the input, internal, and output.
         CALL RANDOM_UNIT_VECTORS(INPUT_VECS(:,:,C))
         DO I = 1, MNS-1
            CALL RANDOM_UNIT_VECTORS(STATE_VECS(:,:,I,C))
         END DO
         CALL RANDOM_UNIT_VECTORS(OUTPUT_VECS(:,:,C))
         ! Make the output vectors have desired initial magnitude (e.g., small).
         OUTPUT_VECS(:,:,C) = OUTPUT_VECS(:,:,C) * OUTPUT_SCALE
         ! Generate deterministic equally spaced shifts for inputs and internal layers, 
         !  zero shift for the output layer (first two will be rescaled).
         DO I = 1, MDS
            INPUT_SHIFT(I,C) = 2.0_RT * SHIFT_RANGE * &             ! 2 * shift *
                 (REAL(I-1,RT) / MAX(1.0_RT, REAL(MDS-1, RT))) &  ! range [0, 1]
                 - SHIFT_RANGE                                    ! - shift
         END DO
         ! Set the state shifts based on translation of the origin, always try
         !  to apply translations to bring the origin back closer to center
         !  (to prevent terrible conditioning of models with many layers).
         ORIGIN(:) = INPUT_SHIFT(:,C)
         DO J = 1, MNS-1
            ORIGIN(:) = MATMUL(ORIGIN(:), STATE_VECS(:,:,J,C))
            ! Argsort the values of origin, adding the most to the minimum values.
            CALL ARGSORT(ORIGIN(:), ORDER(:))
            DO I = 1, MDS
               STATE_SHIFT(ORDER(MDS-I+1),J,C) = INPUT_SHIFT(I,C) ! range [-shift, shift]
            END DO
            ORIGIN(:) = ORIGIN(:) + STATE_SHIFT(:,J,C)
         END DO
      END DO
      ! Deallocate local variables.
      DEALLOCATE(ORIGIN, ORDER)
    END SUBROUTINE INIT_SUBMODEL
  END SUBROUTINE INIT_MODEL


  ! Return nonzero INFO if any shapes or values do not match expectations.
  SUBROUTINE CHECK_SHAPE(CONFIG, MODEL, AX, AXI, SIZES, X, XI, Y, YI, INFO, FOR_EVALUATION)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AX
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: YI
    LOGICAL(KIND=C_BOOL), INTENT(IN), OPTIONAL :: FOR_EVALUATION
    INTEGER(KIND=INT32), INTENT(OUT) :: INFO
    ! Local variable for tracking the additional (not present in Y) output dimensions.
    INTEGER(KIND=INT32) :: ADDITIONAL_OUTPUT
    IF (PRESENT(FOR_EVALUATION)) THEN
       IF (FOR_EVALUATION) THEN
          ADDITIONAL_OUTPUT = 0
       ELSE
          ADDITIONAL_OUTPUT = CONFIG%DOE
       END IF
    ELSE
       ADDITIONAL_OUTPUT = CONFIG%DOE
    END IF
    ! Default code is 0.
    INFO = 0
    ! Compute whether the shape matches the CONFIG.
    IF (SIZE(MODEL,KIND=INT64) .NE. CONFIG%TOTAL_SIZE) THEN
       INFO = 1 ! Model size does not match model configuration.
    ELSE IF (SIZE(X,2,INT64) .NE. SIZE(Y,2,INT64)) THEN
       INFO = 2 ! Input arrays do not match in size.
    ELSE IF (SIZE(X,1,INT64) .NE. CONFIG%MDN) THEN
       INFO = 3 ! X input dimension is bad.
    ELSE IF ((CONFIG%MDO .GT. 0) .AND. (SIZE(Y,1,INT64)+ADDITIONAL_OUTPUT .NE. CONFIG%MDO)) THEN
       INFO = 4 ! Model output dimension is bad, does not match Y.
    ELSE IF ((CONFIG%MDO .EQ. 0) .AND. (SIZE(Y,1,INT64)+ADDITIONAL_OUTPUT .NE. CONFIG%ADO)) THEN
       INFO = 5 ! Aggregator output dimension is bad, does not match Y.
    ELSE IF ((CONFIG%NOE .GT. 0) .AND. (SIZE(YI,2,INT64) .NE. SIZE(Y,2,INT64))) THEN
       INFO = 6 ! Input integer YI size does not match Y.
    ELSE IF ((MINVAL(YI) .LE. 0) .OR. (MAXVAL(YI) .GT. CONFIG%NOE)) THEN
       INFO = 7 ! Input integer YI out of range [1, number of output embeddings].
    ELSE IF ((CONFIG%MNE .GT. 0) .AND. (SIZE(XI,2,INT64) .NE. SIZE(X,2,INT64))) THEN
       INFO = 8 ! Input integer XI size does not match X.
    ELSE IF ((MINVAL(XI) .LT. 0) .OR. (MAXVAL(XI) .GT. CONFIG%MNE)) THEN
       INFO = 9 ! Input integer XI out of range.
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (.NOT. CONFIG%PARTIAL_AGGREGATION) &
         .AND. (SIZE(SIZES) .NE. SIZE(Y,2,INT64))) THEN
       INFO = 10 ! SIZES has wrong size.
    ELSE IF (SIZE(AX,2,INT64) .NE. SUM(SIZES)) THEN
       INFO = 11 ! AX and SUM(SIZES) do not match.
    ELSE IF (SIZE(AX,1,INT64) .NE. CONFIG%ADN) THEN
       INFO = 12 ! AX input dimension is bad.
    ELSE IF (SIZE(AXI,2,INT64) .NE. SIZE(AX,2,INT64)) THEN
       INFO = 13 ! Input integer AXI size does not match AX.
    ELSE IF ((MINVAL(AXI) .LT. 0) .OR. (MAXVAL(AXI) .GT. CONFIG%ANE)) THEN
       INFO = 14 ! Input integer AXI out of range.
    END IF
  END SUBROUTINE CHECK_SHAPE

 
  ! Given a number of batches, compute the batch start and ends for
  !  the aggregator and fixed inputs. Store in (2,_) arrays.
  SUBROUTINE COMPUTE_BATCHES(CONFIG, NA, NM, SIZES, BATCHA_STARTS, &
       BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, JOINT, INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    INTEGER(KIND=INT64), INTENT(IN) :: NA, NM
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
    INTEGER(KIND=INT64), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: BATCHA_STARTS, BATCHA_ENDS
    INTEGER(KIND=INT64), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: AGG_STARTS, FIX_STARTS
    INTEGER(KIND=INT64), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: BATCHM_STARTS, BATCHM_ENDS
    LOGICAL(KIND=C_BOOL), INTENT(IN) :: JOINT
    INTEGER(KIND=INT32), INTENT(INOUT) :: INFO
    ! Local variables.
    INTEGER(KIND=INT64) :: BATCH, BN, BE, BS, GE, GS, I, MB, NB, NUM_A_BATCHES, NUM_M_BATCHES
    ! Check for errors.
    IF (CONFIG%NUM_THREADS .LT. 1) THEN
       WRITE (*,*) 'ERROR (COMPUTE_BATCHES): Number of threads (NUM_THREADS) is not positive.', CONFIG%NUM_THREADS
       INFO = 15 ! Number of threads (NUM_THREADS) is not positive (is 0 or negative).
       RETURN
    ELSE IF (CONFIG%MAX_BATCH .LT. 1) THEN
       WRITE (*,*) 'ERROR (COMPUTE_BATCHES): Batch size (MAX_BATCH) is not positive.', CONFIG%MAX_BATCH
       INFO = 16 ! Number of points per batch (MAX_BATCH) is not positive (is 0 or negative).
       RETURN
    ELSE IF (NM .EQ. 0) THEN
       WRITE (*,*) 'ERROR (COMPUTE_BATCHES): Number of points not positive.', NM
       INFO = 17 ! Number of points is not positive (is 0 or negative).
       RETURN
    END IF
    ! Compute batch sizes and allocate space.
    IF ((.NOT. JOINT) .OR. (NA .EQ. 0)) THEN
       ! Batches are computed independently.
       NUM_A_BATCHES = MAX(MIN(NA, CONFIG%NUM_THREADS), &
            NA / CONFIG%MAX_BATCH)
       NUM_M_BATCHES = MAX(MIN(NM, CONFIG%NUM_THREADS), &
            NM / CONFIG%MAX_BATCH)
    ELSE
       ! Batches are computed jointly, determining batch sizes by
       !  iterating over data (in order) and ending a batch when
       !  either NA or NM limits are hit.
       NUM_M_BATCHES = ONE
       BS = ONE
       BE = ZERO
       ! Compute a max batch size that will be more amenable to the number of threads, if possible.
       MB = MIN(CONFIG%MAX_BATCH, SUM(SIZES) / MAX(ONE,CONFIG%NUM_THREADS-ONE))
       DO I = ONE, SIZE(SIZES)
          ! Get the size of this set.
          NB = SIZES(I)
          IF (CONFIG%PARTIAL_AGGREGATION .AND. (NB .EQ. ZERO)) THEN
             NB = NB + ONE
          END IF
          ! Add it to the end.
          BE = BE + NB
          ! Transition batches based on size of next iterate.
          IF (I .LT. SIZE(SIZES)) THEN
             ! Get the size of the next set.
             NB = SIZES(I+ONE)
             IF (CONFIG%PARTIAL_AGGREGATION .AND. (NB .EQ. ZERO)) THEN
                NB = NB + ONE
             END IF
             ! If the (set size + current size > max batch size) ...
             IF (NB + BE - BS + ONE .GT. MB) THEN
                BS = BE + ONE
                NUM_M_BATCHES = NUM_M_BATCHES + ONE
             END IF
          END IF
       END DO
       NUM_A_BATCHES = NUM_M_BATCHES
       ! ^ joint batching is needed for higher level parallelization
       !   in the fit loop, which will allow for increased throughput.
    END IF
    ! Deallocate arrays if they were already allocated (should not be).
    IF (ALLOCATED(BATCHA_STARTS)) DEALLOCATE(BATCHA_STARTS)
    IF (ALLOCATED(BATCHA_ENDS)) DEALLOCATE(BATCHA_ENDS)
    IF (ALLOCATED(AGG_STARTS)) DEALLOCATE(AGG_STARTS)
    IF (ALLOCATED(FIX_STARTS)) DEALLOCATE(FIX_STARTS)
    IF (ALLOCATED(BATCHM_STARTS)) DEALLOCATE(BATCHM_STARTS)
    IF (ALLOCATED(BATCHM_ENDS)) DEALLOCATE(BATCHM_ENDS)
    ! Allocate new arrays for batching.
    ALLOCATE( &
         BATCHM_STARTS(1:NUM_M_BATCHES), &
         BATCHM_ENDS(1:NUM_M_BATCHES), &
         BATCHA_STARTS(1:NUM_A_BATCHES), &
         BATCHA_ENDS(1:NUM_A_BATCHES), &
         AGG_STARTS(1:MIN(NM,SIZE(SIZES,KIND=INT64))), &
         FIX_STARTS(1:MIN(NM,SIZE(SIZES,KIND=INT64))) &
    )
    ! Construct batches for data sets with aggregator inputs.
    IF (NA .GT. ZERO) THEN
       ! Compute the location of the first index in each aggregate set.
       AGG_STARTS(1) = ONE
       FIX_STARTS(1) = ONE
       DO I = TWO, SIZE(SIZES,KIND=INT64)
          AGG_STARTS(I) = AGG_STARTS(I-ONE) + SIZES(I-ONE)
          FIX_STARTS(I) = FIX_STARTS(I-ONE) + MAX(ONE, SIZES(I-ONE))
       END DO
       ! Handle number of batches.
       IF (NUM_A_BATCHES .EQ. ONE) THEN
          BATCHA_STARTS(1) = ONE
          BATCHA_ENDS(1) = NA
          ! Construct fixed batches.
          BN = (NM + NUM_M_BATCHES - ONE) / NUM_M_BATCHES ! = CEIL(NM / NUM_BATCHES)
          DO BATCH = ONE, NUM_M_BATCHES
             BATCHM_STARTS(BATCH) = BN*(BATCH-ONE) + ONE
             BATCHM_ENDS(BATCH) = MIN(NM, BN*BATCH)
          END DO
       ELSE
          IF (.NOT. JOINT) THEN
             ! Construct aggregate batches.
             BN = (NA + NUM_A_BATCHES - ONE) / NUM_A_BATCHES ! = CEIL(NA / NUM_BATCHES)
             DO BATCH = ONE, NUM_A_BATCHES
                BATCHA_STARTS(BATCH) = MIN(NA+ONE, BN*(BATCH-ONE) + ONE)
                BATCHA_ENDS(BATCH) = MIN(NA, BN*BATCH)
             END DO
             ! Construct fixed batches.
             BN = (NM + NUM_M_BATCHES - ONE) / NUM_M_BATCHES ! = CEIL(NM / NUM_BATCHES)
             DO BATCH = ONE, NUM_M_BATCHES
                BATCHM_STARTS(BATCH) = BN*(BATCH-ONE) + ONE
                BATCHM_ENDS(BATCH) = MIN(NM, BN*BATCH)
             END DO
          ! Handle partial aggregation (which has differnt fixed batches).
          ELSE IF (CONFIG%PARTIAL_AGGREGATION) THEN
             ! Compute the joint batches over the data, with end-to-end parallelization
             !  and more jobs instead of granular parallelization (with more barriers).
             BATCH = ONE
             ! GS, GE -> Fixed group start, fixed group end.
             GS = ONE
             GE = ZERO
             ! BS, BE -> Aggregate batch start, aggregate batch end.
             BS = ONE
             BE = ZERO
             DO I = ONE, SIZE(SIZES)-ONE
                BE = BE + SIZES(I)
                GE = GE + MAX(ONE, SIZES(I))
                ! Transition batches based on size of next iterate without partial aggregation.
                IF ((MAX(ONE,SIZES(I+ONE)) + (GE - GS + ONE) .GT. CONFIG%MAX_BATCH) .OR. &
                     ((SUM(MAX(ONE,SIZES(I+ONE:))) .LE. CONFIG%MAX_BATCH) .AND. &
                     (BATCH .LT. NUM_M_BATCHES))) THEN
                   BATCHA_STARTS(BATCH) = BS
                   BATCHA_ENDS(BATCH) = BE
                   BS = BE + ONE
                   BATCHM_STARTS(BATCH) = GS
                   BATCHM_ENDS(BATCH) = GE
                   GS = GE + ONE
                   BATCH = BATCH + ONE
                END IF
             END DO
             ! Perform steps for last batch.
             NB = SIZES(SIZE(SIZES,KIND=INT64))
             BE = BE + NB
             BATCHA_STARTS(BATCH) = BS
             BATCHA_ENDS(BATCH) = BE
             GE = GE + MAX(ONE,NB)
             BATCHM_STARTS(BATCH) = GS
             BATCHM_ENDS(BATCH) = GE
             NUM_A_BATCHES = NUM_M_BATCHES
          ! Handle standard aggregation.
          ELSE
             ! Compute the joint batches over the data, with end-to-end parallelization
             !  and more jobs instead of granular parallelization (with more barriers).
             BATCH = ONE
             BATCHM_STARTS(BATCH) = ONE
             BS = ONE
             BE = ZERO
             DO I = ONE, SIZE(SIZES)-ONE
                BE = BE + SIZES(I)
                ! Transition batches based on size of next iterate without partial aggregation.
                IF ((SIZES(I+ONE) + (BE - BS + ONE) .GT. CONFIG%MAX_BATCH) .OR. &
                     ((SUM(SIZES(I+ONE:)) .LE. CONFIG%MAX_BATCH) .AND. &
                     (BATCH .LT. NUM_M_BATCHES))) THEN
                   BATCHA_STARTS(BATCH) = BS
                   BATCHA_ENDS(BATCH) = BE
                   BS = BE + ONE
                   BATCHM_ENDS(BATCH) = I
                   BATCH = BATCH + ONE
                   BATCHM_STARTS(BATCH) = I+ONE
                END IF
             END DO
             ! Perform steps for last batch.
             BE = BE + SIZES(SIZE(SIZES))
             BATCHA_STARTS(BATCH) = BS
             BATCHA_ENDS(BATCH) = BE
             BATCHM_ENDS(BATCH) = SIZE(SIZES)
             NUM_A_BATCHES = NUM_M_BATCHES
          END IF
       END IF
    ELSE ! NA = 0
       BATCHA_STARTS(:) = ONE
       BATCHA_ENDS(:) = ZERO
       AGG_STARTS(:) = ONE
       DO BATCH = ONE, SIZE(FIX_STARTS,KIND=INT64)
          FIX_STARTS(BATCH) = BATCH
       END DO
       IF (NUM_M_BATCHES .EQ. ONE) THEN
          BATCHM_STARTS(1) = ONE
          BATCHM_ENDS(1) = NM
       ELSE
          ! Construct fixed batches.
          BN = (NM + NUM_M_BATCHES - ONE) / NUM_M_BATCHES ! = CEIL(NM / NUM_BATCHES)
          DO BATCH = 1, NUM_M_BATCHES
             BATCHM_STARTS(BATCH) = BN*(BATCH-ONE) + ONE
             BATCHM_ENDS(BATCH) = MIN(NM, BN*BATCH)
          END DO
       END IF
    END IF
  END SUBROUTINE COMPUTE_BATCHES


  ! Give the raw input data, fetch a new set of data that fits in memory.
  SUBROUTINE FETCH_DATA(CONFIG, AGG_ITERATORS_IN, &
       AX_IN, AX, AXI_IN, AXI, SIZES_IN, SIZES, &
       X_IN, X, XI_IN, XI, Y_IN, Y, YI_IN, YI, YW_IN, YW, NA, NM)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: AGG_ITERATORS_IN
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AX_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI_IN
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: AXI
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES_IN
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI_IN
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: YI_IN
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: YI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: YW_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW
    INTEGER(KIND=INT64), INTENT(OUT) :: NA, NM
    ! Local variables.
    INTEGER(KIND=INT64) :: I, J, K, L, P1, P2, MAX_AGG, NEXTRA, AS, GENDEX
    REAL(KIND=RT) :: NREMAIN, CURRENT_TOTAL
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:) :: RSIZES
    INTEGER(KIND=INT64), ALLOCATABLE, DIMENSION(:) :: SORTED_ORDER, GENDEXES
    INTEGER(KIND=INT64), ALLOCATABLE, DIMENSION(:) :: AGG_STARTS_IN, AGG_STARTS
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Allocate storage space if it will be used.
    IF (SIZE(SIZES_IN) .GT. 0) THEN
       ALLOCATE(RSIZES(1:CONFIG%NM), SORTED_ORDER(1:CONFIG%NM), GENDEXES(CONFIG%NM))  ! LOCAL ALLOCATION
    END IF
    ! Pack the regular inputs into the working space, storing sizes.
    DO I = 1, MIN(CONFIG%NM, SIZE(Y_IN,2,KIND=INT64))
       ! Choose iteration strategy (linear is faster when all points are
       !   needed, otherwise the random generator is used to pick points).
       IF (CONFIG%NM .GE. SIZE(Y_IN,2,KIND=INT64)) THEN
          GENDEX = I
       ELSE
          GENDEX = GET_NEXT_INDEX(CONFIG%NMT, CONFIG%I_NEXT, CONFIG%I_MULT, &
               CONFIG%I_STEP, CONFIG%I_MOD, CONFIG%I_ITER, &
               RESHUFFLE=CONFIG%RESHUFFLE)
       END IF
       ! Store the size.
       IF (SIZE(SIZES_IN) .GT. 0) THEN
          ! Store this gendex for later (not able to reproduce if reinitialized).
          GENDEXES(I) = GENDEX
          ! Store the size of this element (depending on pairwise).
          RSIZES(I) = REAL(SIZES_IN(GENDEX), KIND=RT)
          IF (CONFIG%PAIRWISE_AGGREGATION) THEN
             RSIZES(I) = RSIZES(I)**2
          END IF
       END IF
       X(:CONFIG%MDN,I) = X_IN(:,GENDEX)
       XI(:,I) = XI_IN(:,GENDEX)
       ! For evaluation, no Y will be provided.
       IF (SIZE(Y,1) .GT. ZERO) THEN
          Y(:CONFIG%DON,I) = Y_IN(:,GENDEX)
          YI(:,I) = YI_IN(:,GENDEX)
          YW(:,I) = YW_IN(:,GENDEX)
       END IF
    END DO
    ! If there are aggregate inputs ...
    IF (SIZE(SIZES_IN) .GT. 0) THEN
       ! Determine a maximum allowed size that will permit the most
       !  points to have the most possible aggregate inputs.
       CALL ARGSORT(RSIZES(:), SORTED_ORDER(:))
       NA = MIN(CONFIG%NA, SIZE(AX,2,KIND=INT64))
       NREMAIN = REAL(NA,RT)
       compute_max_agg : DO I = 1, CONFIG%NM
          ! Assuming all elements had the size of the current element, would they fit?
          CURRENT_TOTAL = RSIZES(SORTED_ORDER(I)) * REAL(CONFIG%NM - I + ONE, RT)
          ! If they do not fit ...
          IF (CURRENT_TOTAL .GT. NREMAIN) THEN
             ! Determine the maximum size that WOULD fit for the current (and all remaining) elements.
             MAX_AGG = INT(NREMAIN, INT64) / (CONFIG%NM - I + ONE)
             ! Count the remainder, the number of aggregate sets that can have 1 extra.
             NEXTRA = INT(NREMAIN, INT64) - MAX_AGG * (CONFIG%NM - I + ONE)
             ! Set the size for those inputs that get the MAX + 1.
             DO J = I, I+NEXTRA-ONE
                SIZES(SORTED_ORDER(J)) = MAX_AGG + ONE
             END DO
             ! Set the size for those inputs that get the MAX permissible aggregate elements.
             DO J = I+NEXTRA, CONFIG%NM
                SIZES(SORTED_ORDER(J)) = MAX_AGG
             END DO
             NREMAIN = 0_RT
             EXIT compute_max_agg
          END IF
          ! Otherwise, this does fit, so copy over the size into the output.
          SIZES(SORTED_ORDER(I)) = INT(RSIZES(SORTED_ORDER(I)), INT64)
          ! Deduct the number of avilable slots remaining.
          NREMAIN = NREMAIN - RSIZES(SORTED_ORDER(I))
       END DO compute_max_agg
       ! Deallocate memory that is no longer needed.
       DEALLOCATE(RSIZES, SORTED_ORDER)
       ! Compute the start indices for the different aggregate sets (in the input).
       ALLOCATE(AGG_STARTS_IN(SIZE(SIZES_IN)), AGG_STARTS(SIZE(SIZES)))
       AGG_STARTS_IN(1) = ONE
       DO I = ONE, SIZE(SIZES_IN)-ONE
          AGG_STARTS_IN(I+ONE) = AGG_STARTS_IN(I) + SIZES_IN(I)
       END DO
       AGG_STARTS(1) = ONE
       DO I = ONE, SIZE(SIZES)-ONE
          AGG_STARTS(I+ONE) = AGG_STARTS(I) + SIZES(I)
       END DO
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS) PRIVATE(I, GENDEX, AS, J, K, L, P1, P2)
       DO I = 1, MIN(CONFIG%NM, SIZE(SIZES_IN,KIND=INT64))
          ! Retrieve the same GENDEX used for the I'th fixed element.
          GENDEX = GENDEXES(I)
          ! Randomly initialize the iterator for this aggregate set (each time we enter).
          IF ((CONFIG%RESHUFFLE) .AND. (.NOT. CONFIG%ORDERED_AGGREGATION)) THEN
             CALL INITIALIZE_ITERATOR( &
                  I_LIMIT=AGG_ITERATORS_IN(1,GENDEX), &
                  I_NEXT=AGG_ITERATORS_IN(2,GENDEX), &
                  I_MULT=AGG_ITERATORS_IN(3,GENDEX), &
                  I_STEP=AGG_ITERATORS_IN(4,GENDEX), &
                  I_MOD=AGG_ITERATORS_IN(5,GENDEX), &
                  I_ITER=AGG_ITERATORS_IN(6,GENDEX) &
             )
          ! If this set is ordered, only iterate over the last SIZE elements.
          ELSE IF (CONFIG%ORDERED_AGGREGATION) THEN
             AGG_ITERATORS_IN(1,GENDEX) = AGG_ITERATORS_IN(1,GENDEX) ! I_LIMIT
             AGG_ITERATORS_IN(2,GENDEX) = AGG_ITERATORS_IN(1,GENDEX) - SIZES(I) ! I_NEXT
             AGG_ITERATORS_IN(3,GENDEX) = ONE ! I_MULT
             AGG_ITERATORS_IN(4,GENDEX) = ONE ! I_STEP
             AGG_ITERATORS_IN(5,GENDEX) = AGG_ITERATORS_IN(5,GENDEX) ! I_MOD
             AGG_ITERATORS_IN(6,GENDEX) = ZERO ! I_ITER
          END IF
          AS = AGG_STARTS(I)
          ! Pack in those inputs. Note that SIZES(I) is derived from SIZES_IN(GENDEX).
          ! We are using the same generator to recreate the GENDEX from earlier.
          DO J = 1, SIZES(I)
             ! Get the random AGG index (might be a pair).
             K = GET_NEXT_INDEX( &
                  AGG_ITERATORS_IN(1,GENDEX), &
                  AGG_ITERATORS_IN(2,GENDEX), &
                  AGG_ITERATORS_IN(3,GENDEX), &
                  AGG_ITERATORS_IN(4,GENDEX), &
                  AGG_ITERATORS_IN(5,GENDEX), &
                  AGG_ITERATORS_IN(6,GENDEX), &
                  RESHUFFLE=.FALSE._C_BOOL &
             )
             IF (CONFIG%PAIRWISE_AGGREGATION) THEN
                ! Get a unique pair 
                CALL INDEX_TO_PAIR(SIZES_IN(GENDEX), K, P1, P2)
                P1 = AGG_STARTS_IN(GENDEX) + P1 - ONE
                P2 = AGG_STARTS_IN(GENDEX) + P2 - ONE
                ! Handle the case of a lone index (it paired with itself).
                IF (P1 .EQ. P2) THEN
                   ! Store AGG value in AX.
                   AX(:CONFIG%ADN,AS) = AX_IN(:,P1)
                   ! Store AGG value in AXI.
                   AXI(:,AS) = AXI_IN(:,P1)
                ! Handle the case of two unique indices (that are being compared).
                ELSE
                   ! Compute AX pair difference and store.
                   AX(:CONFIG%ADN,AS) = AX_IN(:,P1) - AX_IN(:,P2)
                   ! Translate AXI embedding differences into appropriate integer values.
                   DO L = 1, SIZE(AXI_IN,1)
                      ! Get the single integer representing this pair.
                      CALL PAIR_TO_INDEX(INT(CONFIG%ANE,INT64)+ONE, AXI_IN(L,P1)+ONE, AXI_IN(L,P2)+ONE, K)
                      AXI(L,AS) = K + INT(CONFIG%ANE,INT64)
                      ! ^ Integer embeddings over the total number are considered to be pairs of embeddings.
                   END DO
                END IF
             ELSE  ! .NOT. CONFIG%PAIRWISE_AGGREGATION
                ! Convert the index in THIS aggregate set (K) to the offset input index (P1).
                P1 = AGG_STARTS_IN(GENDEX) + K - ONE
                ! Store AGG value in AX.
                AX(:CONFIG%ADN,AS) = AX_IN(:,P1)
                ! Store AGG value in AXI.
                AXI(:,AS) = AXI_IN(:,P1)
             END IF ! PAIRWISE_AGGREGATION
             ! Update A Start.
             AS = AS + ONE
          END DO
       END DO
       ! Done use the gendexes.
       DEALLOCATE(GENDEXES)
       ! Set the total number of aggregate inputs that were added.
       NA = AGG_STARTS(SIZE(AGG_STARTS)) + SIZES(SIZE(SIZES)) - ONE
       ! For PARTIAL_AGGREGATION, 'I' needs to step proportional to how many
       !   aggregate inputs a given point has, repeating the value in each spot.
       IF (CONFIG%PARTIAL_AGGREGATION) THEN
          ! Overwrite AGG_STARTS_IN to hold the starts of the batch inputs.
          DO I = 1, MIN(CONFIG%NM, SIZE(SIZES,KIND=INT64))-1
             AGG_STARTS_IN(I+1) = AGG_STARTS_IN(I) + MAX(ONE,SIZES(I))
          END DO
          ! Cycle backwards through the batches, making appropriate copies of fixed values
          !  that were originally placed in indices (1:N). Starting at the back removes
          !  the need for memory scratch space for performing the copy.
          DO I = MIN(CONFIG%NM, SIZE(SIZES,KIND=INT64)), 1, -1
             DO J = AGG_STARTS_IN(I), AGG_STARTS_IN(I) + MAX(ONE,SIZES(I)) - ONE
                X(:CONFIG%MDN,J) = X(:CONFIG%MDN,I)
                XI(:,J) = XI(:,I)
                ! For evaluation, no Y will be provided.
                IF (SIZE(Y,1) .GT. ZERO) THEN
                   Y(:CONFIG%DON,J) = Y(:CONFIG%DON,I)
                   YI(:,J) = YI(:,I)
                   YW(:,J) = YW(:,I)
                END IF
             END DO
          END DO
          ! Set the total NM.
          I = MIN(CONFIG%NM, SIZE(SIZES,KIND=INT64))
          NM = AGG_STARTS_IN(I) + MAX(ONE,SIZES(I)) - ONE
       ELSE
          NM = CONFIG%NM
       END IF
       ! Deallocate memory for identifying start of aggregate sets.
       DEALLOCATE(AGG_STARTS_IN, AGG_STARTS)
    ELSE
       NA = ZERO
       NM = CONFIG%NM
    END IF
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    !$OMP CRITICAL
    CONFIG%WGEN = CONFIG%WGEN + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CGEN = CONFIG%CGEN + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    !$OMP END CRITICAL
  END SUBROUTINE FETCH_DATA


  ! Given a model and mixed real and integer inputs, embed the integer
  !  inputs into their appropriate real-value-only formats.
  ! 
  ! TODO: Should this routine check for usage errors since it is expected
  !       to be called by external users when evaluating a model?
  SUBROUTINE EMBED(CONFIG, MODEL, AXI, XI, AX, X)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX ! ADI, SIZE(AX,2)
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X  ! MDI, SIZE(X,2)
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! If there is AXInteger input, unpack it into X.
    IF (CONFIG%ADE .GT. 0) THEN
       CALL UNPACK_EMBEDDINGS(CONFIG%ADE, CONFIG%ANE, &
            MODEL(CONFIG%ASEV:CONFIG%AEEV), &
            AXI(:,:), AX(CONFIG%ADN+1:SIZE(AX,1),:))
    END IF
    ! If there is XInteger input, unpack it into end of X.
    IF (CONFIG%MDE .GT. 0) THEN
       CALL UNPACK_EMBEDDINGS(CONFIG%MDE, CONFIG%MNE, &
            MODEL(CONFIG%MSEV:CONFIG%MEEV), &
            XI(:,:), X(CONFIG%MDN+ONE:CONFIG%MDN+CONFIG%MDE,:))
    END IF
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    !$OMP CRITICAL
    CONFIG%WEMB = CONFIG%WEMB + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CEMB = CONFIG%CEMB + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    !$OMP END CRITICAL

  CONTAINS
    ! Given integer inputs and embedding vectors, put embeddings in
    !  place of integer inputs inside of a real matrix.
    SUBROUTINE UNPACK_EMBEDDINGS(MDE, MNE, EMBEDDINGS, INT_INPUTS, EMBEDDED)
      INTEGER(KIND=INT32), INTENT(IN) :: MDE, MNE
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDE, MNE) :: EMBEDDINGS
      INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: INT_INPUTS
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: EMBEDDED
      INTEGER(KIND=INT64) :: N, D, E, E1, E2
      REAL(KIND=RT) :: RD
      RD = REAL(SIZE(INT_INPUTS,1,INT64),RT)
      ! Add together appropriate embedding vectors based on integer inputs.
      EMBEDDED(:,:) = 0.0_RT
      !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS) PRIVATE(N, D, E, E1, E2)
      DO N = ONE, SIZE(INT_INPUTS,2,KIND=INT64)
         DO D = ONE, SIZE(INT_INPUTS,1,KIND=INT64)
            E = INT_INPUTS(D,N)
            ! Add embeddings referenced directly by 1-index.
            IF ((E .GT. 0) .AND. (E .LE. MNE)) THEN
               EMBEDDED(:,N) = EMBEDDED(:,N) + EMBEDDINGS(:,E)
            ! Larger integers are reserved for pairs of embeddings.
            ELSE IF (E .GT. MNE) THEN
               CALL INDEX_TO_PAIR(INT(MNE,INT64)+ONE, E-MNE, E1, E2)
               ! Skip "0" embeddings, since they are the zero vector.
               IF (E1 .GT. ONE) THEN
                  EMBEDDED(:,N) = EMBEDDED(:,N) + EMBEDDINGS(:,E1-ONE)
               END IF
               IF (E2 .GT. ONE) THEN
                  EMBEDDED(:,N) = EMBEDDED(:,N) - EMBEDDINGS(:,E2-ONE)
               END IF
            ! 0-valued integers are not embedded.
            END IF
         END DO
         ! If multiple embeddings are provided (in columns) they are averaged.
         IF (SIZE(INT_INPUTS,1,KIND=INT64) > ONE) THEN
            EMBEDDED(:,N) = EMBEDDED(:,N) / RD
         END IF
      END DO
    END SUBROUTINE UNPACK_EMBEDDINGS
  END SUBROUTINE EMBED


  ! Normalize numeric input values (for prediction time).
  SUBROUTINE NORMALIZE_INPUTS(CONFIG, MODEL, AX, SIZES, X, INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), TARGET, INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER(KIND=INT32), INTENT(OUT) :: INFO
    ! Local variables.
    REAL(KIND=RT), POINTER, DIMENSION(:) :: X_SHIFT, AX_SHIFT
    REAL(KIND=RT), POINTER, DIMENSION(:,:) :: X_RESCALE, AX_RESCALE
    ! LOCAL ALLOCATION
    INTEGER(KIND=INT64), DIMENSION(:), ALLOCATABLE :: &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS
    INTEGER(KIND=INT64) :: I, BATCH, BS, BE, BT, NT
    ! 
    INFO = 0
    ! Set the pointers to the internal normalization values.
    IF (CONFIG%NEEDS_SHIFTING) THEN
       AX_SHIFT(1:CONFIG%ADN) => MODEL(CONFIG%AISS:CONFIG%AISE)
       X_SHIFT(1:CONFIG%MDN) => MODEL(CONFIG%MISS:CONFIG%MISE)
    END IF
    IF (CONFIG%NEEDS_SCALING) THEN
       AX_RESCALE(1:CONFIG%ADN,1:CONFIG%ADN) => MODEL(CONFIG%AIMS:CONFIG%AIME)
       X_RESCALE(1:CONFIG%MDN,1:CONFIG%MDN) => MODEL(CONFIG%MIMS:CONFIG%MIME)
    END IF
    ! Compute the batch start and end indices.
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.FALSE._C_BOOL, INFO=INFO)
    IF (INFO .NE. 0) RETURN
    ! Normalize the aggregate inputs.
    IF (CONFIG%NORMALIZE .AND. (CONFIG%ADO .GT. ZERO)) THEN
       ! Set the number of threads.
       NT = MIN(SIZE(AX,2,KIND=INT64), CONFIG%NUM_THREADS)
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       aggregate_normalization : DO BATCH = 1, SIZE(BATCHA_STARTS, KIND=INT64)
          BS = BATCHA_STARTS(BATCH)
          BE = BATCHA_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .LE. 0) CYCLE aggregate_normalization
          ! Normalize the data.
          IF (CONFIG%ADN .GT. 0) THEN
             ! Apply shift terms to aggregator inputs.
             IF (CONFIG%NEEDS_SHIFTING) THEN
                DO I = BS, BE
                   AX(:CONFIG%ADN,I) = AX(:CONFIG%ADN,I) + AX_SHIFT(:)
                END DO
             END IF
             ! Remove any NaN or Inf values from the data.
             IF (CONFIG%NEEDS_CLEANING) THEN
                WHERE (IS_NAN(AX(:CONFIG%ADN,BS:BE)) .OR. (.NOT. IS_FINITE(AX(:CONFIG%ADN,BS:BE))))
                   AX(:CONFIG%ADN,BS:BE) = 0.0_RT
                END WHERE
             END IF
             ! Apply multiplier.
             IF (CONFIG%NEEDS_SCALING) THEN
                AX(:CONFIG%ADN,BS:BE) = MATMUL(TRANSPOSE(AX_RESCALE(:,:)), AX(:CONFIG%ADN,BS:BE))
             END IF
          END IF
       END DO aggregate_normalization
    END IF
    ! Normalize the fixed inputs.
    IF (CONFIG%NORMALIZE .AND. (CONFIG%MDO .GT. ZERO)) THEN
       ! Set the number of threads.
       NT = MIN(SIZE(X,2,KIND=INT64), CONFIG%NUM_THREADS)
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       fixed_normalization : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          ! Update "BS", "BE", and "BT" to coincide with the model.
          BS = BATCHM_STARTS(BATCH)
          BE = BATCHM_ENDS(BATCH)
          BT = BE-BS+1
          IF ((BT .LE. 0) .OR. (MIN(BS,BE) .LE. 0)) CYCLE fixed_normalization
          IF (CONFIG%MDN .GT. 0) THEN
             ! Apply shift terms to numeric model inputs.
             IF (CONFIG%NEEDS_SHIFTING) THEN
                DO I = BS, BE
                   X(:CONFIG%MDN,I) = X(:CONFIG%MDN,I) + X_SHIFT(:)
                END DO
             END IF
             ! Remove any NaN or Inf values from the data.
             IF (CONFIG%NEEDS_CLEANING) THEN
                WHERE (IS_NAN(X(:CONFIG%MDN,BS:BE)) .OR. (.NOT. IS_FINITE(X(:CONFIG%MDN,BS:BE))))
                   X(:CONFIG%MDN,BS:BE) = 0.0_RT
                END WHERE
             END IF
             ! Apply multiplier.
             IF (CONFIG%NEEDS_SCALING) THEN
                X(:CONFIG%MDN,BS:BE) = MATMUL(TRANSPOSE(X_RESCALE(:,:)), X(:CONFIG%MDN,BS:BE))
             END IF
          END IF
       END DO fixed_normalization
    END IF
    ! Deallocate batch sizes.
    DEALLOCATE(BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS)
  END SUBROUTINE NORMALIZE_INPUTS


  ! Evaluate the piecewise linear regression model, assume already-embedded inputs.
  SUBROUTINE EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, Y, A_STATES, M_STATES, INFO)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), TARGET, INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: AY
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:,:,:) :: A_STATES ! SIZE(AX,2), ADS, (ANS|2), ANC
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:,:,:) :: M_STATES ! SIZE(X, 2), MDS, (MNS|2), MNC
    INTEGER(KIND=INT32), INTENT(INOUT) :: INFO
    ! Internal values.
    INTEGER(KIND=INT64) :: I, BATCH, BS, BE, BT, NT, E
    REAL(KIND=RT), POINTER, DIMENSION(:) :: AY_SHIFT, AY_SCALE, Y_SHIFT
    REAL(KIND=RT), POINTER, DIMENSION(:,:) :: Y_RESCALE
    REAL(KIND=RT) :: CW
    ! LOCAL ALLOCATION
    INTEGER(KIND=INT64), DIMENSION(:), ALLOCATABLE :: &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS
    ! Timing.
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! If there are no points to evaluate, then immediately return.
    IF (SIZE(Y,2,KIND=INT64) .EQ. 0) RETURN
    AY_SHIFT(1:CONFIG%ADO) => MODEL(CONFIG%AOSS:CONFIG%AOSE)
    AY_SCALE(1:CONFIG%ADO) => MODEL(CONFIG%AOMS:CONFIG%AOME)
    ! Compute the batch start and end indices.
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.FALSE._C_BOOL, INFO=INFO)
    IF (INFO .NE. 0) RETURN
    ! Compute the number of threads based on number of points.
    NT = MIN(SIZE(Y,2,KIND=INT64), CONFIG%NUM_THREADS)
    ! 
    ! Aggregator (set) model evaluation.
    ! 
    IF (CONFIG%ADO .GT. 0) THEN
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       aggregator_evaluation : DO BATCH = 1, SIZE(BATCHA_STARTS, KIND=INT64)
          BS = BATCHA_STARTS(BATCH)
          BE = BATCHA_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .LE. 0) CYCLE aggregator_evaluation
          ! Evaluate the aggregator model.
          CALL UNPACKED_EVALUATE(INT(BT), &
               CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ANC, CONFIG%ADSO, CONFIG%ADO+1, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASIS:CONFIG%AEIS), &
               MODEL(CONFIG%ASSV:CONFIG%AESV), &
               MODEL(CONFIG%ASSS:CONFIG%AESS), &
               MODEL(CONFIG%ASOV:CONFIG%AEOV), &
               AX(:,BS:BE), AY(BS:BE,:), A_STATES(BS:BE,:,:,:), YTRANS=.TRUE._C_BOOL)
          ! It's possible that some AY error estimation terms are zero or negative,
          !  and that is currently handled with a MAX(epsilon, ...) in aggregation.
       END DO aggregator_evaluation
       ! 
       ! Aggregate the output of the set model.
       ! 
       ! Compute the final output.
       IF (CONFIG%MDO .GT. 0) THEN
          E = CONFIG%MDN+CONFIG%MDE+ONE ! <- start of aggregator output
          CALL COMPUTE_SET_AGGREGATION(Y=X(E:,:))
       ELSE
          ! If there is no model after this, place results directly in Y.
          CALL COMPUTE_SET_AGGREGATION(Y=Y(:,:))
       END IF  ! MDO > 0
    END IF  ! ADO > 0
    ! 
    ! Fixed model evaluation.
    ! 
    IF (CONFIG%MDO .GT. 0) THEN
       ! Get the number of batches.
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       model_evaluation : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          ! Update "BS", "BE", and "BT" to coincide with the model.
          BS = BATCHM_STARTS(BATCH)
          BE = BATCHM_ENDS(BATCH)
          BT = BE-BS+1
          IF ((BT .LE. 0) .OR. (MIN(BS,BE) .LE. 0)) CYCLE model_evaluation
          ! Run the fixed model.
          CALL UNPACKED_EVALUATE(INT(BT), &
               CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MNC, CONFIG%MDSO, CONFIG%MDO, &
               MODEL(CONFIG%MSIV:CONFIG%MEIV), &
               MODEL(CONFIG%MSIS:CONFIG%MEIS), &
               MODEL(CONFIG%MSSV:CONFIG%MESV), &
               MODEL(CONFIG%MSSS:CONFIG%MESS), &
               MODEL(CONFIG%MSOV:CONFIG%MEOV), &
               X(:,BS:BE), Y(:,BS:BE), M_STATES(BS:BE,:,:,:), YTRANS=.FALSE._C_BOOL)
       END DO model_evaluation
    END IF
    ! Apply shift terms to final outputs.
    IF (CONFIG%NORMALIZE) THEN
       ! Set the pointers to the appropriate spots in model memory.
       IF (CONFIG%NEEDS_SHIFTING) THEN
          Y_SHIFT(1:CONFIG%DON) => MODEL(CONFIG%MOSS:CONFIG%MOSE)
       END IF
       IF (CONFIG%NEEDS_SCALING) THEN
          Y_RESCALE(1:CONFIG%DON,1:CONFIG%DON) => MODEL(CONFIG%MOMS:CONFIG%MOME)
       END IF
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       output_normalization : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          ! Update "BS", "BE", and "BT" to coincide with the model.
          BS = BATCHM_STARTS(BATCH)
          BE = BATCHM_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .LE. 0) CYCLE output_normalization
          ! Apply scaling multiplier.
          IF (CONFIG%NEEDS_SCALING) THEN
             Y(:CONFIG%DON,BS:BE) = MATMUL(TRANSPOSE(Y_RESCALE(:,:)), Y(:CONFIG%DON,BS:BE))
          END IF
          ! Apply shift.
          IF (CONFIG%NEEDS_SHIFTING) THEN
             DO I = BS, BE
                Y(:CONFIG%DON,I) = Y(:CONFIG%DON,I) - Y_SHIFT(:)
             END DO
          END IF
       END DO output_normalization
    END IF
    ! Deallocate batch sizes.
    DEALLOCATE(BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS)
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    !$OMP CRITICAL
    CONFIG%WEVL = CONFIG%WEVL + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CEVL = CONFIG%CEVL + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    !$OMP END CRITICAL


  CONTAINS

    SUBROUTINE COMPUTE_SET_AGGREGATION(Y)
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: Y
      INTEGER(KIND=INT64) :: I, J, GS, GE, FE
      !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, J, GS, GE, FE, CW) IF(NT > 1)
      set_aggregation_to_y : DO I = ONE, SIZE(SIZES,KIND=INT64)
         IF (SIZES(I) .GT. 0) THEN
            ! Take the mean of all outputs from the aggregator model, store
            !   as input to the model that proceeds this aggregation.
            GS = AGG_STARTS(I)
            GE = AGG_STARTS(I) + SIZES(I) - ONE
            FE = FIX_STARTS(I) + MAX(ZERO, SIZES(I) - ONE)
            IF (CONFIG%PARTIAL_AGGREGATION) THEN
               ! Assume that the fixed data has the same alignment as the aggregate data.
               CW = MAX(CONFIG%MIN_AGG_WEIGHT, AY(GE,CONFIG%ADO+1))  ! convex weight
               Y(:,FE) = AY(GE,:CONFIG%ADO) * CW
               ! Accumulate the running sum in the last position, updating the divisor.
               DO J = ONE, SIZES(I)-ONE
                  Y(:,FE) = Y(:,FE) + AY(GE-J,:CONFIG%ADO) * MAX(CONFIG%MIN_AGG_WEIGHT, AY(GE-J,CONFIG%ADO+1))
                  ! Accumulate the running sum of weights (to be made convex).
                  CW = CW + MAX(CONFIG%MIN_AGG_WEIGHT, AY(GE-J,CONFIG%ADO+1))
                  ! Compute this output.
                  Y(:,FE-J) = Y(:,FE) / CW
               END DO
               ! Revert the running sum in the last position to just the value (weight = 1).
               Y(:,FE) = AY(GE,:CONFIG%ADO)
               ! Incorporate AY output normalization.
               IF (CONFIG%MDO .GT. 0) THEN
                  DO J = FIX_STARTS(I), FE
                     Y(:,J) = (Y(:,J) - AY_SHIFT(:)) * AY_SCALE(:)
                  END DO
               END IF
            ELSE
               ! Otherwise, without partial aggregation, just put the weighted average aggregate into X.
               CW = SUM(MAX(CONFIG%MIN_AGG_WEIGHT, AY(GS:GE,CONFIG%ADO+1)))
               Y(:,I) = MATMUL(MAX(CONFIG%MIN_AGG_WEIGHT, AY(GS:GE,CONFIG%ADO+1)), AY(GS:GE,:CONFIG%ADO)) / CW
               ! Incorporate AY output normalization.
               IF (CONFIG%MDO .GT. 0) THEN
                  Y(:,I) = (Y(:,I) - AY_SHIFT(:)) * AY_SCALE(:)
               END IF
            END IF
         ELSE
            IF (CONFIG%PARTIAL_AGGREGATION) THEN
               Y(:,FIX_STARTS(I)) = 0.0_RT
            ELSE
               Y(:,I) = 0.0_RT
            END IF
         END IF
      END DO set_aggregation_to_y
    END SUBROUTINE COMPUTE_SET_AGGREGATION

    SUBROUTINE UNPACKED_EVALUATE(N, MDI, MDS, MNS, MNC, MDSO, MDO, INPUT_VECS, &
         INPUT_SHIFT, STATE_VECS, STATE_SHIFT, OUTPUT_VECS, X, Y, &
         STATES, YTRANS)
      INTEGER(KIND=INT32), INTENT(IN) :: N, MDI, MDS, MNS, MNC, MDSO, MDO
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDI, MDS, MNC) :: INPUT_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MNC) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MDS, MAX(0,MNS-1), MNC) :: STATE_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDS, MAX(0,MNS-1), MNC) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(IN),  DIMENSION(MDSO, MDO, MNC) :: OUTPUT_VECS
      REAL(KIND=RT), INTENT(IN),  DIMENSION(:,:) :: X
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: Y
      REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:,:,:) :: STATES
      LOGICAL(KIND=C_BOOL), INTENT(IN) :: YTRANS
      ! Local variables to evaluating a single batch.
      INTEGER(KIND=INT32) :: C, D, L, S1, S2, S3
      LOGICAL(KIND=C_BOOL) :: REUSE_STATES
      REAL(KIND=RT) :: RESULT_MULTIPLIER
      REUSE_STATES = (SIZE(STATES,3) .LT. MNS)
      IF (MNS .GT. 0) THEN
         RESULT_MULTIPLIER = 0.0_RT
         ! Sum over chords.
         DO C = 1, MNC
            ! Compute the input transformation.
            CALL GEMM('T', 'N', N, MDS, MDI, 1.0_RT, &
                 X(:,:), SIZE(X,1), &
                 INPUT_VECS(:,:,C), SIZE(INPUT_VECS,1), &
                 0.0_RT, STATES(:,:,1,C), N)
            ! Apply the truncation.
            DO D = 1, MDS
               STATES(:,D,1,C) = MAX(STATES(:,D,1,C) + INPUT_SHIFT(D,C), CONFIG%DISCONTINUITY)
            END DO
            ! Compute the next set of internal values with a truncated activation.
            DO L = 1, MNS-1
               ! Determine the storage locations of values based on number of states.
               IF (REUSE_STATES) THEN ; S1 = 1 ; S2 = 2   ; S3 = 1
               ELSE                   ; S1 = L ; S2 = L+1 ; S3 = L+1
               END IF
               ! Compute all vectors.
               CALL GEMM('N', 'N', N, MDS, MDS, 1.0_RT, &
                    STATES(:,:,S1,C), N, &
                    STATE_VECS(:,:,L,C), SIZE(STATE_VECS,1), &
                    0.0_RT, STATES(:,:,S2,C), N)
               ! Compute all piecewise linear functions and apply the truncation.
               DO D = 1, MDS
                  STATES(:,D,S3,C) = MAX(STATES(:,D,S2,C) + STATE_SHIFT(D,L,C), CONFIG%DISCONTINUITY)
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
                    STATES(:,:,S3,C), N, &
                    OUTPUT_VECS(:,:,C), SIZE(OUTPUT_VECS,1), &
                    RESULT_MULTIPLIER, Y(:,:), SIZE(Y,1))
            ELSE
               CALL GEMM('T', 'T', MDO, N, MDS, 1.0_RT, &
                    OUTPUT_VECS(:,:,C), SIZE(OUTPUT_VECS,1), &
                    STATES(:,:,S3,C), N, &
                    RESULT_MULTIPLIER, Y(:,:), SIZE(Y,1))
            END IF
            ! After one iteration is done, set the result multiplier to SUM instead of overwrite.
            RESULT_MULTIPLIER = 1.0_RT
         END DO
      ! Handle the linear model case, where there are only output vectors.
      ELSE
         ! Return the final output (default to assuming Y is contiguous
         !   by component unless PRESENT(YTRANS) and YTRANS = .TRUE.
         !   then assume it is contiguous by individual sample).
         IF (YTRANS) THEN
            CALL GEMM('T', 'N', N, MDO, MDI, 1.0_RT, &
                 X(:,:), SIZE(X,1), &
                 OUTPUT_VECS(:,:,1), SIZE(OUTPUT_VECS,1), &
                 0.0_RT, Y(:,:), SIZE(Y,1))
         ELSE
            CALL GEMM('T', 'N', MDO, N, MDI, 1.0_RT, &
                 OUTPUT_VECS(:,:,1), SIZE(OUTPUT_VECS,1), &
                 X(:,:), SIZE(X,1), &
                 0.0_RT, Y(:,:), SIZE(Y,1))
         END IF
      END IF
    END SUBROUTINE UNPACKED_EVALUATE
  END SUBROUTINE EVALUATE


  ! Compute the gradient with respect to embeddings given the input
  !  gradient by aggregating over the repeated occurrences of the embedding.
  SUBROUTINE EMBEDDING_GRADIENT(MDE, MNE, PAIRWISE, INT_INPUTS, GRAD, &
       EMBEDDING_GRAD, TEMP_GRAD)
    INTEGER(KIND=INT32), INTENT(IN) :: MDE, MNE
    LOGICAL(KIND=C_BOOL), INTENT(IN) :: PAIRWISE
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: INT_INPUTS
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: GRAD
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDE,MNE) :: EMBEDDING_GRAD
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: TEMP_GRAD
    ! Local variables.
    INTEGER(KIND=INT64) :: N, D, E, E1, E2
    ! Accumulate the gradients for all embedding vectors.
    TEMP_GRAD(:,:) = 0.0_RT
    IF (PAIRWISE) THEN
       DO N = 1, SIZE(INT_INPUTS,2,INT64)
          DO D = 1, SIZE(INT_INPUTS,1,INT64)
             E = INT(INT_INPUTS(D,N),INT64)
             IF ((E .GT. 0) .AND. (E .LE. MNE)) THEN
                TEMP_GRAD(:,E) = TEMP_GRAD(:,E) + GRAD(:,N)
             ELSE IF (E .GT. MNE) THEN
                CALL INDEX_TO_PAIR(INT(MNE,INT64)+ONE, E-MNE, E1, E2)
                IF (E1 .GT. ONE) THEN
                   E1 = E1 - ONE
                   TEMP_GRAD(:,E1) = TEMP_GRAD(:,E1) + GRAD(:,N)
                END IF
                IF (E2 .GT. ONE) THEN
                   E2 = E2 - ONE
                   TEMP_GRAD(:,E2) = TEMP_GRAD(:,E2) - GRAD(:,N)
                END IF
             END IF
          END DO
       END DO
    ELSE
       DO N = 1, SIZE(INT_INPUTS,2,INT64)
          DO D = 1, SIZE(INT_INPUTS,1,INT64)
             E = INT(INT_INPUTS(D,N),INT64)
             IF (E .GT. 0) THEN
                TEMP_GRAD(:,E) = TEMP_GRAD(:,E) + GRAD(:,N)
             END IF
          END DO
       END DO
    END IF
    ! Average the embedding gradient by dividing by the sum of occurrences (and threads).
    DO E = 1, MNE
       EMBEDDING_GRAD(:,E) = EMBEDDING_GRAD(:,E) + TEMP_GRAD(:,E)
    END DO
  END SUBROUTINE EMBEDDING_GRADIENT


  ! Given the values at all internal states in the model and an output
  !  gradient, propogate the output gradient through the model and
  !  return the gradient of all basis functions.
  SUBROUTINE BASIS_GRADIENT(CONFIG, MODEL, Y, X, AX, SIZES, &
       M_STATES, A_STATES, AY, GRAD, BATCHA_STARTS, BATCHA_ENDS, &
       AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, NT)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: M_STATES ! SIZE(X, 2), MDS, MNS+1, MNC
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: A_STATES ! SIZE(AX,2), ADS, ANS+1, ANC
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY ! SIZE(AX,2), ADO+1
    REAL(KIND=RT), INTENT(OUT),  DIMENSION(:,:) :: GRAD ! SIZE(MODEL), NUM_THREADS
    INTEGER(KIND=INT64), DIMENSION(:), INTENT(IN) :: &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS
    INTEGER(KIND=INT64), INTENT(IN) :: NT
    ! Set the dimension of the X gradient that should be calculated.
    REAL(KIND=RT) :: YSUM, CW, CWS
    INTEGER(KIND=INT64) :: I, J, FS, GS, GE, XDG, BATCH, TN
    IF (.FALSE.) THEN
       I = NT
    END IF
    ! Propogate the gradient through the fixed model.
    IF (CONFIG%MDO .GT. 0) THEN
       XDG = CONFIG%MDE
       IF (CONFIG%ADI .GT. 0) THEN
          XDG = XDG + CONFIG%ADO
       END IF
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(GS, GE, TN) IF(NT > 1)
       DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          GS = BATCHM_STARTS(BATCH)
          GE = BATCHM_ENDS(BATCH)
          IF (GS .GT. GE) CYCLE
          TN = OMP_GET_THREAD_NUM() + 1
          ! Do the backward gradient calculation assuming "Y" contains output gradient.
          CALL UNPACKED_BASIS_GRADIENT( CONFIG, Y(:,GS:GE), M_STATES(GS:GE,:,:,:), X(:,GS:GE), &
               CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MNC, CONFIG%MDSO, CONFIG%MDO, INT(XDG), 0, &
               MODEL(CONFIG%MSIV:CONFIG%MEIV), &
               MODEL(CONFIG%MSSV:CONFIG%MESV), &
               MODEL(CONFIG%MSOV:CONFIG%MEOV), &
               GRAD(CONFIG%MSIV:CONFIG%MEIV,TN), &
               GRAD(CONFIG%MSIS:CONFIG%MEIS,TN), &
               GRAD(CONFIG%MSSV:CONFIG%MESV,TN), &
               GRAD(CONFIG%MSSS:CONFIG%MESS,TN), &
               GRAD(CONFIG%MSOV:CONFIG%MEOV,TN), &
               YTRANS=.FALSE._C_BOOL) ! Y is in COLUMN vector format.
       END DO
    END IF
    ! Propogate the gradient from X into the aggregate outputs.
    IF (CONFIG%ADI .GT. 0) THEN
       ! Propogate gradient from the input to the fixed model.
       IF (CONFIG%MDO .GT. 0) THEN
          XDG = SIZE(X,1) - CONFIG%ADO + ONE  ! <- the first X column for aggregated values
          CALL COMPUTE_AGGREGATION_GRADIENT(OUT=X(XDG:,:))
       ! Propogate gradient directly from the aggregate output.
       ELSE
          CALL COMPUTE_AGGREGATION_GRADIENT(OUT=Y(:,:))
       END IF
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(BATCH, GS, GE, TN) IF(NT > 1)
       DO BATCH = ONE, SIZE(BATCHA_STARTS, KIND=INT64)
          GS = BATCHA_STARTS(BATCH)
          GE = BATCHA_ENDS(BATCH)
          IF (GS .GT. GE) CYCLE
          TN = OMP_GET_THREAD_NUM() + ONE
          ! Do the backward gradient calculation assuming "AY" contains output gradient.
          CALL UNPACKED_BASIS_GRADIENT( CONFIG, AY(GS:GE,:), A_STATES(GS:GE,:,:,:), AX(:,GS:GE), &
               CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ANC, CONFIG%ADSO, CONFIG%ADO, CONFIG%ADE, 1, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASSV:CONFIG%AESV), &
               MODEL(CONFIG%ASOV:CONFIG%AEOV), &
               GRAD(CONFIG%ASIV:CONFIG%AEIV,TN), &
               GRAD(CONFIG%ASIS:CONFIG%AEIS,TN), &
               GRAD(CONFIG%ASSV:CONFIG%AESV,TN), &
               GRAD(CONFIG%ASSS:CONFIG%AESS,TN), &
               GRAD(CONFIG%ASOV:CONFIG%AEOV,TN), &
               YTRANS=.TRUE._C_BOOL) ! AY is in ROW vector format.
       END DO
    END IF

  CONTAINS

    ! Compute the gradient from the output through the aggregation operation.
    SUBROUTINE COMPUTE_AGGREGATION_GRADIENT(OUT)
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: OUT
      !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(FS, GS, GE, I, J, CW, CWS, YSUM) IF(NT > 1)
      DO I = ONE, SIZE(SIZES, KIND=INT64)
         GS = AGG_STARTS(I)
         GE = GS + SIZES(I) - ONE
         IF (CONFIG%PARTIAL_AGGREGATION) THEN
            IF (SIZES(I) .GT. ZERO) THEN
               FS = FIX_STARTS(I)
               ! Compute the sum of all weights applied to this group.
               CW = SUM(MAX(CONFIG%MIN_AGG_WEIGHT, AY(GS:GE,CONFIG%ADO+1)))
               CWS = CW
               ! Compute the initial gradient for the element that occurs only once,
               !  the first position which is added to all other elements.
               AY(GS,:CONFIG%ADO) = OUT(:,FS) / CW
               ! Error term (2-norm output error over elements of this size or smaller).
               YSUM = SQRT(SUM(Y(:,FS)**2))
               IF (AY(GS, CONFIG%ADO+1) .GE. CONFIG%MIN_AGG_WEIGHT) THEN
                  AY(GS,CONFIG%ADO+1) = AY(GS,CONFIG%ADO+1) - (1.0_RT / (1.0_RT + YSUM))
               ELSE
                  AY(GS,CONFIG%ADO+1) = CONFIG%MIN_AGG_WEIGHT - AY(GS,CONFIG%ADO+1)
               END IF
               ! 
               ! Compute the gradient for each aggregate element, which is now
               !  the sum of the gradients from the X.
               DO J = ONE, SIZES(I) - ONE
                  ! Compute the sum of all weights applied to this group.
                  !  Avoiding subtracting from existing CW for numerical stability.
                  CW = SUM(MAX(CONFIG%MIN_AGG_WEIGHT, AY(GS+J:GE,CONFIG%ADO+1)))
                  ! Add to the running total gradient.
                  AY(GS,:CONFIG%ADO) = AY(GS,:CONFIG%ADO) + OUT(:,FS+J) / CW
                  ! Store this aggregate value's gradient term.
                  AY(GS+J,:CONFIG%ADO) = AY(GS,:CONFIG%ADO)
                  ! Error term (sum 2-norm errors here, average is computed in division).
                  YSUM = YSUM + SQRT(SUM(Y(:,FS+J)**2))
                  ! Divide by the number of elements to get average 2-norm error
                  !  of all predictions that utilize tihs specific input.
                  IF (AY(GS+J,CONFIG%ADO+1) .GE. CONFIG%MIN_AGG_WEIGHT) THEN
                     AY(GS+J,CONFIG%ADO+1) = AY(GS+J,CONFIG%ADO+1) - &
                          (1.0_RT / (1.0_RT + YSUM / REAL(J+1, KIND=RT)))
                  ELSE
                     AY(GS+J,CONFIG%ADO+1) = CONFIG%MIN_AGG_WEIGHT - &
                          AY(GS+J,CONFIG%ADO+1)
                  END IF
               END DO
               ! Reset the computation of the first AY that was used to aggregate.
               AY(GS,:CONFIG%ADO) = OUT(:,FS) / CWS
            END IF
         ELSE
            ! Without partial aggregation, all AY receive equal weight.
            CW = SUM(MAX(CONFIG%MIN_AGG_WEIGHT, AY(GS:GE,CONFIG%ADO+1)))
            YSUM = SQRT(SUM(Y(:,I)**2))
            DO J = GS, GE
               AY(J,:CONFIG%ADO) = OUT(:,I)
               ! Compute the target value for the last column of AY to be sum
               !  of componentwise squared errors values for all outputs.
               IF (AY(J,CONFIG%ADO+1) .GE. CONFIG%MIN_AGG_WEIGHT) THEN
                  AY(J,CONFIG%ADO+1) = AY(J,CONFIG%ADO+1) - (1.0_RT / (1.0_RT + YSUM))
               ELSE
                  AY(J,CONFIG%ADO+1) = CONFIG%MIN_AGG_WEIGHT - AY(J,CONFIG%ADO+1)
               END IF
            END DO
            ! Apply the same divisor that was applied in evaluation to all points.
            AY(GS:GE,:CONFIG%ADO) = AY(GS:GE,:CONFIG%ADO) / CW
         END IF
      END DO
    END SUBROUTINE COMPUTE_AGGREGATION_GRADIENT

    ! Compute the model gradient.
    SUBROUTINE UNPACKED_BASIS_GRADIENT( CONFIG, Y, STATES, X, &
         MDI, MDS, MNS, MNC, MDSO, MDO, MDE, EXTRA, &
         INPUT_VECS, STATE_VECS, OUTPUT_VECS, &
         INPUT_VECS_GRADIENT, INPUT_SHIFT_GRADIENT, &
         STATE_VECS_GRADIENT, STATE_SHIFT_GRADIENT, &
         OUTPUT_VECS_GRADIENT, YTRANS )
      TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
      REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: STATES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
      INTEGER(KIND=INT32), INTENT(IN) :: MDI, MDS, MNS, MNC, MDSO, MDO, MDE, EXTRA
      ! Model variables.
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDI,MDS,MNC) :: INPUT_VECS
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDS,MDS,MAX(0,MNS-1),MNC) :: STATE_VECS
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDSO,MDO+EXTRA,MNC) :: OUTPUT_VECS
      ! Model variable gradients.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI,MDS,MNC) :: INPUT_VECS_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MNC) :: INPUT_SHIFT_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MDS,MAX(0,MNS-1),MNC) :: STATE_VECS_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS,MAX(0,MNS-1),MNC) :: STATE_SHIFT_GRADIENT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDSO,MDO+EXTRA,MNC) :: OUTPUT_VECS_GRADIENT
      LOGICAL(KIND=C_BOOL), INTENT(IN) :: YTRANS
      ! D   - dimension index
      ! L   - layer index
      ! LP1 - layer index "plus 1" -> "P1"
      INTEGER(KIND=INT32) :: C, D, L, LP1
      CHARACTER :: YT
      ! Set default for assuming Y is transposed (row vectors).
      IF (YTRANS) THEN ; YT = 'N'
      ELSE             ; YT = 'T'
      END IF
      ! Handle propogation of the gradient through internal states.
      IF (MNS .GT. 0) THEN
         DO C = 1, MNC
            ! Compute the gradient of variables with respect to the "output gradient"
            CALL GEMM('T', YT, MDS, MDO+EXTRA, SIZE(X,2), 1.0_RT, &
                 STATES(:,:,MNS,C), SIZE(STATES,1), &
                 Y(:,:), SIZE(Y,1), &
                 1.0_RT, OUTPUT_VECS_GRADIENT(:,:,C), SIZE(OUTPUT_VECS_GRADIENT,1))
            ! Propogate the gradient back to the last internal vector space.
            IF (EXTRA .EQ. 0) THEN
               CALL GEMM(YT, 'T', SIZE(X,2), MDS, MDO, 1.0_RT, &
                    Y(:,:), SIZE(Y,1), &
                    OUTPUT_VECS(:,:,C), SIZE(OUTPUT_VECS,1), &
                    0.0_RT, STATES(:,:,MNS+1,C), SIZE(STATES,1))
               ! Handle (EXTRA>0) and Y is row vectors.
            ELSE IF (YTRANS) THEN
               CALL GEMM(YT, 'T', SIZE(X,2), MDS, MDO, 1.0_RT, &
                    Y(:,:MDO), SIZE(Y,1), &
                    OUTPUT_VECS(:,:,C), SIZE(OUTPUT_VECS,1), &
                    0.0_RT, STATES(:,:,MNS+1,C), SIZE(STATES,1))
               ! Handle (EXTRA>0) and Y is column vectors.
            ELSE
               CALL GEMM(YT, 'T', SIZE(X,2), MDS, MDO, 1.0_RT, &
                    Y(:MDO,:), SIZE(Y,1), &
                    OUTPUT_VECS(:,:,C), SIZE(OUTPUT_VECS,1), &
                    0.0_RT, STATES(:,:,MNS+1,C), SIZE(STATES,1))
            END IF
            ! Cycle over all internal layers.
            STATE_REPRESENTATIONS : DO L = MNS-1, 1, -1
               LP1 = L+1
               DO D = 1, MDS
                  ! Propogate the error gradient back to the preceding vectors.
                  WHERE (STATES(:,D,LP1,C) .GT. CONFIG%DISCONTINUITY)
                     STATES(:,D,LP1,C) = STATES(:,D,MNS+1,C)
                  END WHERE
               END DO
               ! Compute the shift gradient.
               STATE_SHIFT_GRADIENT(:,L,C) = SUM(STATES(:,:,LP1,C), 1) &
                    + STATE_SHIFT_GRADIENT(:,L,C)
               ! Compute the gradient with respect to each output and all inputs.
               CALL GEMM('T', 'N', MDS, MDS, SIZE(X,2), 1.0_RT, &
                    STATES(:,:,L,C), SIZE(STATES,1), &
                    STATES(:,:,LP1,C), SIZE(STATES,1), &
                    1.0_RT, STATE_VECS_GRADIENT(:,:,L,C), SIZE(STATE_VECS_GRADIENT,1))
               ! Propogate the gradient to the immediately preceding layer.
               CALL GEMM('N', 'T', SIZE(X,2), MDS, MDS, 1.0_RT, &
                    STATES(:,:,LP1,C), SIZE(STATES,1), &
                    STATE_VECS(:,:,L,C), SIZE(STATE_VECS,1), &
                    0.0_RT, STATES(:,:,MNS+1,C), SIZE(STATES,1))
            END DO STATE_REPRESENTATIONS
            ! Compute the gradients going into the first layer.
            DO D = 1, MDS
               ! Propogate the error gradient back to the preceding vectors.
               WHERE (STATES(:,D,1,C) .GT. CONFIG%DISCONTINUITY)
                  STATES(:,D,1,C) = STATES(:,D,MNS+1,C)
               END WHERE
            END DO
            ! Compute the input shift variable gradients.
            INPUT_SHIFT_GRADIENT(:,C) = SUM(STATES(:,:,1,C), 1) &
                 + INPUT_SHIFT_GRADIENT(:,C)
            ! Compute the gradient of all input variables.
            !   [the X are transposed already, shape = (MDI,N)]
            CALL GEMM('N', 'N', MDI, MDS, SIZE(X,2), 1.0_RT, &
                 X(:,:), SIZE(X,1), &
                 STATES(:,:,1,C), SIZE(STATES,1), &
                 1.0_RT, INPUT_VECS_GRADIENT(:,:,C), SIZE(INPUT_VECS_GRADIENT,1))
            ! Compute the gradient at the input if there are embeddings.
            IF (MDE .GT. 0) THEN
               LP1 = SIZE(X,1)-MDE+1
               CALL GEMM('N', 'T', MDE, SIZE(X,2), MDS, 1.0_RT, &
                    INPUT_VECS(LP1:,:,C), MDE, &
                    STATES(:,:,1,C), SIZE(STATES,1), &
                    0.0_RT, X(LP1:,:), MDE)
            END IF
         END DO
      ! Handle the purely linear case (no internal states).
      ELSE
         ! Compute the gradient of variables with respect to the "output gradient"
         CALL GEMM('N', YT, MDI, MDO, SIZE(X,2), 1.0_RT, &
              X(:,:), SIZE(X,1), &
              Y(:,:), SIZE(Y,1), &
              1.0_RT, OUTPUT_VECS_GRADIENT(:,:,1), SIZE(OUTPUT_VECS_GRADIENT,1))
         ! Propogate the gradient back to the input embeddings.
         IF (MDE .GT. 0) THEN
            LP1 = SIZE(X,1)-MDE+1
            IF (YTRANS) THEN ; YT = 'T'
            ELSE             ; YT = 'N'
            END IF
            CALL GEMM('N', YT, MDE, SIZE(X,2), MDO, 1.0_RT, &
                 OUTPUT_VECS(LP1:,:,1), MDE, &
                 Y(:,:), SIZE(Y,1), &
                 0.0_RT, X(LP1:,:), MDE)
         END IF
      END IF
    END SUBROUTINE UNPACKED_BASIS_GRADIENT
  END SUBROUTINE BASIS_GRADIENT

  
  ! Given the model output values "Y_GRADIENT", the 'true' numeric
  ! values "Y", and the true categorical values "YI", produce the
  ! gradient at the output and store it in "Y_GRADIENT".
  SUBROUTINE OUTPUT_GRADIENT(CONFIG, Y_GRADIENT, Y, YI, YW, O_EMB_VECS, O_EMB_GRAD, &
       EMB_OUTS, EMB_GRADS, SSG, DON)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_GRADIENT
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: YI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: YW
    REAL(KIND=RT), INTENT(IN), DIMENSION(CONFIG%DOE,CONFIG%NOE) :: O_EMB_VECS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(CONFIG%DOE,CONFIG%NOE) :: O_EMB_GRAD
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: EMB_OUTS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: EMB_GRADS
    REAL(KIND=RT), INTENT(INOUT) :: SSG
    INTEGER, INTENT(IN) :: DON
    INTEGER :: D, I, C, NT
    ! ----------------------------------------------------------------------------------------
    !                             Embedding gradient calculation
    ! Compute the number of threads based on number of points.
    NT = INT(MIN(SIZE(Y,2,KIND=INT64), INT(CONFIG%NUM_THREADS,KIND=INT64)), KIND=INT32)
    ! TODO: Remove MATMUL in favor of GEMM.
    ! Compute embedding outputs (1:NOE,1:N) by taking the dot product
    ! of output vectors with the matrix of all output emeddings.
    EMB_OUTS(:,:) = MATMUL( & ! (NOE, N)
         TRANSPOSE(O_EMB_VECS(:,:)), & ! TRANSPOSE((DOE, NOE))
         Y_GRADIENT(DON+ONE:,:)) ! (DOE, N)
    ! First, assume all categories are negative examples, compute those gradients.
    WHERE (EMB_OUTS(:,:) .GT. 1.0_RT - CONFIG%CATEGORY_GAP)
       EMB_GRADS(:,:) = EMB_OUTS(:,:) - (1.0_RT - CONFIG%CATEGORY_GAP)
    ELSEWHERE
       EMB_GRADS(:,:) = 0.0_RT
    END WHERE
    ! Then for each data point, we will compute the "correct embedding" gradient at that point.
    !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, D, C) IF(NT > 1)
    DO I = 1, SIZE(EMB_OUTS,2) ! Loop over N
       ! For each output component of YI we will compute a gradient score.
       DO D = 1, SIZE(YI,1)
          C = INT(YI(D,I), KIND=INT32) ! No category should exist in two columns YI, so this is unique to column D.
          IF (EMB_OUTS(C,I) .LT. 1.0_RT + CONFIG%CATEGORY_GAP) THEN
             EMB_GRADS(C,I) = EMB_OUTS(C,I) - (1.0_RT + CONFIG%CATEGORY_GAP)
          ELSE
             EMB_GRADS(C,I) = 0.0_RT
          END IF
       END DO
    END DO
    ! ----------------------------------------------------------------------------------------
    !                             Value gradient calculation
    ! Compute the gradient of the model outputs, overwriting "Y_GRADIENT"
    Y_GRADIENT(:DON,:) = Y_GRADIENT(:DON,:) - Y(:DON,:) ! squared error gradient
    ! ----------------------------------------------------------------------------------------
    !                   Weighting gradient (only relevant for a weighted fit)
    ! TODO: Handle case where YW has DON+ONE elements, assuming the +1 is for the embeddings.
    ! Apply weights to the computed gradients (if they were provided.
    IF (SIZE(YW,1) .EQ. DON) THEN
       ! Handle 1 weight per component of each point.
       WHERE (YW(:,:) .GT. 0.0_RT)
          Y_GRADIENT(:DON,:) = Y_GRADIENT(:DON,:) * YW(:,:)
       ELSEWHERE (YW(:,:) .LT. 0.0_RT)
          ! TODO: Check to see if this is even useful, because it is "costly".
          ! Compute a weighting function that translates [0, -inf) -> [1, 0).
          Y_GRADIENT(:DON,:) = Y_GRADIENT(:DON,:) * (1.0_RT / (1.0_RT - YW(:,:)))
       END WHERE
    ! Handle 1 weight per point.
    ELSE IF (SIZE(YW,1) .EQ. 1) THEN
       DO D = 1, DON
          WHERE (YW(1,:) .GT. 0.0_RT)
             Y_GRADIENT(D,:) = Y_GRADIENT(D,:) * YW(1,:)
          ELSEWHERE (YW(1,:) .LT. 0.0_RT)
             ! TODO: Check to see if this is even useful, because it is "costly".
             ! Compute a weighting function that translates [0, -inf) -> [1, 0).
             Y_GRADIENT(D,:) = Y_GRADIENT(D,:) * (1.0_RT / (1.0_RT - YW(1,:)))
          END WHERE
       END DO
       ! Apply weights to categorical output gradients.
       DO D = 1, CONFIG%NOE
          WHERE (YW(1,:) .GT. 0.0_RT)
             EMB_GRADS(D,:) = EMB_GRADS(D,:) * YW(1,:)
          ELSEWHERE (YW(1,:) .LT. 0.0_RT)
             ! TODO: Check to see if this is even useful, because it is "costly".
             ! Compute a weighting function that translates [0, -inf) -> [1, 0).
             EMB_GRADS(D,:) = EMB_GRADS(D,:) * (1.0_RT / (1.0_RT - YW(1,:)))
          END WHERE
       END DO
    END IF
    ! ----------------------------------------------------------------------------------------
    ! TODO: Find out what the "best" multiplier is here for the output errors.
    EMB_GRADS(:,:) = EMB_GRADS(:,:) * SQRT(REAL(CONFIG%DOE,RT))
    ! TODO: Remove MATMUL in favor of GEMM.
    ! Compute the gradient of the embeddings.
    O_EMB_GRAD(:,:) = MATMUL( & ! (DOE, NOE) = ...
         Y_GRADIENT(DON+ONE:,:), & ! (DOE, N)
         TRANSPOSE(EMB_GRADS(:,:))) ! TRANSPOSE((NOE, N))
    ! TODO: Remove MATMUL in favor of GEMM.
    ! Compute the parts of Y_GRADIENT that come from the embeddings.
    Y_GRADIENT(DON+ONE:,:) = MATMUL( & ! (DOE, N) = ...
         O_EMB_VECS(:,:), & ! (DOE, NOE)
         EMB_GRADS(:,:)) ! (NOE, N)
    ! Compute the total squared gradient.
    SSG = SSG + SUM(Y_GRADIENT(:,:)**2)
  END SUBROUTINE OUTPUT_GRADIENT


  ! Compute the gradient of the sum of squared error of this regression
  ! model with respect to its variables given input and output pairs.
  SUBROUTINE MODEL_GRADIENT(CONFIG, MODEL, &
       AX, AXI, SIZES, X, XI, Y, YI, YW, &
       SUM_SQUARED_GRADIENT, MODEL_GRAD, INFO, &
       AY_GRADIENT, Y_GRADIENT, A_GRADS, M_GRADS, &
       A_EMB_TEMP, M_EMB_TEMP, &
       EMB_OUTS, EMB_GRADS)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: YI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: YW
    ! Sum (over all data) squared error (summed over dimensions).
    REAL(KIND=RT), INTENT(INOUT) :: SUM_SQUARED_GRADIENT
    ! Gradient of the model variables.
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: MODEL_GRAD
    ! Output and optional inputs.
    INTEGER(KIND=INT32), INTENT(INOUT) :: INFO
    ! Work space.
    REAL(KIND=RT) :: SSG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: A_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: M_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_EMB_TEMP
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: M_EMB_TEMP
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: EMB_OUTS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: EMB_GRADS
    INTEGER(KIND=INT64) :: NT, TN, BATCH, SS, SE, MS, ME
    ! LOCAL ALLOCATION
    INTEGER(KIND=INT64), DIMENSION(:), ALLOCATABLE :: BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS
    ! Timing.
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Compute the batch start and end indices.
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.FALSE._C_BOOL, INFO=INFO)
    IF (INFO .NE. 0) RETURN
    ! Compute the number of threads based on number of points.
    NT = MIN(SIZE(Y,2,KIND=INT64), CONFIG%NUM_THREADS)
    ! Set gradients to zero initially.
    MODEL_GRAD(:,:) = 0.0_RT
    SSG = 0.0_RT
    !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(BATCH, MS, ME, TN) &
    !$OMP& REDUCTION(+:SSG) IF(NT > 1)
    error_gradient : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
       ! Set batch start and end indices. Exit early if there is no data.
       MS = BATCHM_STARTS(BATCH)
       ME = BATCHM_ENDS(BATCH)
       IF (MS .GT. ME) CYCLE
       TN = OMP_GET_THREAD_NUM() + ONE
       ! Compute the gradient of the output (handle any YI embeddings).
       CALL OUTPUT_GRADIENT(CONFIG, Y_GRADIENT(:,MS:ME), Y(:,MS:ME), YI(:,MS:ME), YW(:,MS:ME), &
            MODEL(CONFIG%OSEV:CONFIG%OEEV), MODEL_GRAD(CONFIG%OSEV:CONFIG%OEEV,TN), &
            EMB_OUTS(:,MS:ME), EMB_GRADS(:,MS:ME), SSG, CONFIG%DON)
    END DO error_gradient
    SUM_SQUARED_GRADIENT = SUM_SQUARED_GRADIENT + SSG
    ! Adjust the batches to be defined based on inputs (aggregate sets kept together).
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.TRUE._C_BOOL, INFO=INFO)
    ! Make the gradient the average across all points (so internal routines can sum).
    Y_GRADIENT(:,:) = Y_GRADIENT(:,:) / REAL(SIZE(Y,2),KIND=RT)
    ! Compute the gradient with respect to the model basis functions (does its own parallelism).
    CALL BASIS_GRADIENT(CONFIG, MODEL, Y_GRADIENT(:,:), X(:,:), AX(:,:), &
         SIZES(:), M_GRADS(:,:,:,:), A_GRADS(:,:,:,:), AY_GRADIENT(:,:), &
         MODEL_GRAD(:,:), BATCHA_STARTS(:), BATCHA_ENDS(:), AGG_STARTS(:), FIX_STARTS(:), &
         BATCHM_STARTS(:), BATCHM_ENDS(:), NT)
    ! Readjust the batches back to being equally dispersed.
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.FALSE._C_BOOL, INFO=INFO)
    IF (CONFIG%MDE .GT. 0) THEN
       !$OMP PARALLEL DO NUM_THREADS(NT) &
       !$OMP& PRIVATE(BATCH, MS, ME, TN) IF(NT > 1)
       m_embeddings_gradient : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          ! Set batch start and end indices. Exit early if there is no data.
          MS = BATCHM_STARTS(BATCH)
          ME = BATCHM_ENDS(BATCH)
          IF (MS .GT. ME) CYCLE
          TN = OMP_GET_THREAD_NUM() + 1
          ! Convert the computed input gradients into average gradients for each embedding.
          CALL EMBEDDING_GRADIENT(CONFIG%MDE, CONFIG%MNE, .FALSE._C_BOOL, &
               XI(:,MS:ME), X(CONFIG%MDN+ONE:CONFIG%MDN+CONFIG%MDE,MS:ME), &
               MODEL_GRAD(CONFIG%MSEV:CONFIG%MEEV,TN), M_EMB_TEMP(:,:,TN))
       END DO m_embeddings_gradient
    END IF
    IF (CONFIG%ADE .GT. 0) THEN
       !$OMP PARALLEL DO NUM_THREADS(NT) &
       !$OMP& PRIVATE(BATCH, SS, SE, TN) IF(NT > 1)
       a_embeddings_gradient : DO BATCH = 1, SIZE(BATCHA_STARTS, KIND=INT64)
          SS = BATCHA_STARTS(BATCH)
          SE = BATCHA_ENDS(BATCH)
          IF (SS .GT. SE) CYCLE
          TN = OMP_GET_THREAD_NUM() + 1          
          CALL EMBEDDING_GRADIENT(CONFIG%ADE, CONFIG%ANE, CONFIG%PAIRWISE_AGGREGATION, &
               AXI(:,SS:SE), AX(CONFIG%ADN+1:CONFIG%ADI,SS:SE), &
               MODEL_GRAD(CONFIG%ASEV:CONFIG%AEEV,TN), A_EMB_TEMP(:,:,TN))
       END DO a_embeddings_gradient
    END IF
    ! Free memory devoted to batces.
    DEALLOCATE(BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS)
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    !$OMP CRITICAL
    CONFIG%WGRD = CONFIG%WGRD + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CGRD = CONFIG%CGRD + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    !$OMP END CRITICAL
  END SUBROUTINE MODEL_GRADIENT


  ! Given the data for a single step of training, ensure the data has a normalized
  !  geometry that aligns with model assumptions (linear radialization).
  SUBROUTINE NORMALIZE_STEP(CONFIG, MODEL, RWORK, AX, X, Y, YW)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: RWORK
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW

    ! TODO: Make normalized step work only when it is supposed to.
    RETURN
    CALL NORMALIZE_STEP_UNPACKED( CONFIG, AX, X, Y, YW, &
         MODEL(CONFIG%AISS:CONFIG%AISE), & ! AX_SHIFT(ADN)
         MODEL(CONFIG%AIMS:CONFIG%AIME), & ! AX_RESCALE(ADN,ADN)
         RWORK(CONFIG%SAXIS : CONFIG%EAXIS), & ! AXI_SHIFT(ADE)
         RWORK(CONFIG%SAXIR : CONFIG%EAXIR), & ! AXI_RESCALE(ADE,ADE)
         MODEL(CONFIG%MISS:CONFIG%MISE), & ! X_SHIFT(MDN)
         MODEL(CONFIG%MIMS:CONFIG%MIME), & ! X_RESCALE(MDN,MDN)
         RWORK(CONFIG%SMXIS : CONFIG%EMXIS), & ! XI_SHIFT(MDE)
         RWORK(CONFIG%SMXIR : CONFIG%EMXIR), & ! XI_RESCALE(MDE,MDE)
         MODEL(CONFIG%MOSS:CONFIG%MOSE), & ! Y_SHIFT(DO-DOE)
         MODEL(CONFIG%MOMS:CONFIG%MOME), & ! Y_RESCALE(DO-DOE,DO-DOE)
         RWORK(CONFIG%SOXIS : CONFIG%EOXIS), & ! YI_SHIFT(DOE)
         RWORK(CONFIG%SOXIR : CONFIG%EOXIR), & ! YI_RESCALE(DOE,DOE)
         MODEL(CONFIG%ASEV : CONFIG%AEEV), & ! A_EMB_VECS
         MODEL(CONFIG%MSEV : CONFIG%MEEV), & ! M_EMB_VECS
         MODEL(CONFIG%OSEV : CONFIG%OEEV)) ! O_EMB_VECS


  CONTAINS
    SUBROUTINE NORMALIZE_STEP_UNPACKED(&
         CONFIG, &
         AX, X, Y, YW, &
         AX_SHIFT, AX_RESCALE, AXI_SHIFT, AXI_RESCALE, &
         X_SHIFT, X_RESCALE, XI_SHIFT, XI_RESCALE, &
         Y_SHIFT, Y_RESCALE, YI_SHIFT, YI_RESCALE, &
         A_EMB_VECS, M_EMB_VECS, O_EMB_VECS)
      TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW
      ! The following have assumed shapes (based on slices of a flat-packed array).
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADN) :: AX_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADN, CONFIG%ADN) :: AX_RESCALE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADE) :: AXI_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADE, CONFIG%ADE) :: AXI_RESCALE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDN) :: X_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDN, CONFIG%MDN) :: X_RESCALE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDE) :: XI_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDE, CONFIG%MDE) :: XI_RESCALE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%DON) :: Y_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%DON, CONFIG%DON) :: Y_RESCALE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%DOE) :: YI_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%DOE, CONFIG%DOE) :: YI_RESCALE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADE, CONFIG%ANE) :: A_EMB_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDE, CONFIG%MNE) :: M_EMB_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%DOE, CONFIG%NOE) :: O_EMB_VECS
      ! Local variables.
      INTEGER(KIND=INT64) :: D
      INTEGER :: TO_FLATTEN
      REAL(KIND=RT) :: SCALAR
      LOGICAL(KIND=C_BOOL), ALLOCATABLE, DIMENSION(:,:) :: YW_MASK ! LOCAL ALLOCATION
      REAL(KIND=RT), ALLOCATABLE, DIMENSION(:,:) :: Y_SCALE ! LOCAL ALLOCATION
      ALLOCATE( &
           YW_MASK(SIZE(YW,1), SIZE(YW,2)), &
           Y_SCALE(SIZE(Y_RESCALE,1), SIZE(Y_RESCALE,2)) &
      )
      ! 
      ! Assume EMBED has already happened.
      ! Decide whether or not to flatten.
      ! Compute [0,1] rescaling.
      ! Compute componentwise mean values and remove NAN, INF.
      ! Update mean values towards the result.
      ! Compute symmetric multiplication.
      ! Multiply current vectors by covariance matrix and orthonormalize.
      ! Update rescaling vectors towards orthonormal result.
      ! 
      ! PRINT *, "axy.f90 Line 2580: "
      ! PRINT *, "SHAPE(MODEL): ", SHAPE(MODEL)
      ! PRINT *, "SHAPE(AX): ", SHAPE(AX)
      ! PRINT *, "SHAPE(SIZES): ", SHAPE(SIZES)
      ! PRINT *, "SHAPE(X): ", SHAPE(X)
      ! PRINT *, "SHAPE(Y): ", SHAPE(Y)
      ! PRINT *, "SHAPE(YW): ", SHAPE(YW)
      ! PRINT *, "SHAPE(AX_SHIFT): ", SHAPE(AX_SHIFT)
      ! PRINT *, "SHAPE(AX_RESCALE): ", SHAPE(AX_RESCALE)
      ! PRINT *, "SHAPE(AXI_SHIFT): ", SHAPE(AXI_SHIFT)
      ! PRINT *, "SHAPE(AXI_RESCALE): ", SHAPE(AXI_RESCALE)
      ! PRINT *, "SHAPE(X_SHIFT): ", SHAPE(X_SHIFT)
      ! PRINT *, "SHAPE(X_RESCALE): ", SHAPE(X_RESCALE)
      ! PRINT *, "SHAPE(XI_SHIFT): ", SHAPE(XI_SHIFT)
      ! PRINT *, "SHAPE(XI_RESCALE): ", SHAPE(XI_RESCALE)
      ! PRINT *, "SHAPE(Y_SHIFT): ", SHAPE(Y_SHIFT)
      ! PRINT *, "SHAPE(Y_RESCALE): ", SHAPE(Y_RESCALE)
      ! PRINT *, "SHAPE(YI_SHIFT): ", SHAPE(YI_SHIFT)
      ! PRINT *, "SHAPE(YI_RESCALE): ", SHAPE(YI_RESCALE)
      ! PRINT *, "SHAPE(A_EMB_VECS): ", SHAPE(A_EMB_VECS)
      ! PRINT *, "SHAPE(M_EMB_VECS): ", SHAPE(M_EMB_VECS)
      ! PRINT *, "SHAPE(O_EMB_VECS): ", SHAPE(O_EMB_VECS)

      ! AX
      IF ((.NOT. CONFIG%AX_NORMALIZED) .AND. (CONFIG%ADN .GT. 0)) THEN
         IF (CONFIG%RESCALE_AX) THEN
            TO_FLATTEN = CONFIG%ADN
         ELSE
            TO_FLATTEN = 0
         END IF
         CALL RADIALIZE(AX(:CONFIG%ADN,:), AX_SHIFT(:), AX_RESCALE(:,:), &
              MAX_TO_FLATTEN=TO_FLATTEN, MAXBOUND=.TRUE.)
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, CONFIG%ADN
            AX(D,:) = AX(D,:) + AX_SHIFT(D)
         END DO
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, SIZE(AX,2,INT64)
            AX(:,D) = MATMUL(AX(:,D), AX_RESCALE(:,:))
         END DO
         ! Set all invalid values to zeros.
         WHERE (IS_NAN(AX(:,:)) .OR. (.NOT. IS_FINITE(AX(:,:))))
            AX(:,:) = 0.0_RT
         END WHERE
      ELSE IF (CONFIG%ADN .GT. 0) THEN
         ! Set all invalid values to zeros.
         WHERE (IS_NAN(AX(:,:)) .OR. (.NOT. IS_FINITE(AX(:,:))))
            AX(:,:) = 0.0_RT
         END WHERE
      END IF
      ! AXI
      IF ((.NOT. CONFIG%AXI_NORMALIZED) .AND. (CONFIG%ADE .GT. 0)) THEN
         CALL RADIALIZE(AX(CONFIG%ADN+1:CONFIG%ADN+CONFIG%ADE,:), AXI_SHIFT(:), AXI_RESCALE(:,:))
         ! Apply the shift to the source embeddings.
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, CONFIG%ADE
            A_EMB_VECS(D,:) = A_EMB_VECS(D,:) + AXI_SHIFT(D)
         END DO
         ! Apply the transformation to the source embeddings.
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, SIZE(A_EMB_VECS,2,INT64)
            A_EMB_VECS(:,D) = MATMUL(A_EMB_VECS(:,D), AXI_RESCALE(:,:))
         END DO
         ! Renormalize the embeddings to have a consistent maximum norm.
         SCALAR = MAXVAL(SUM(A_EMB_VECS(:,:)**2, 1))
         IF (SCALAR .GT. 0.0_RT) THEN
            A_EMB_VECS(:,:) = A_EMB_VECS(:,:) / SQRT(SCALAR)
         END IF
      END IF
      ! X
      IF ((.NOT. CONFIG%X_NORMALIZED) .AND. (CONFIG%MDN .GT. 0)) THEN
         IF (CONFIG%RESCALE_X) THEN
            TO_FLATTEN = CONFIG%MDN
         ELSE
            TO_FLATTEN = 0
         END IF
         CALL RADIALIZE(X(:CONFIG%MDN,:), X_SHIFT(:), X_RESCALE(:,:), &
              MAX_TO_FLATTEN=TO_FLATTEN, MAXBOUND=.TRUE.)
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, CONFIG%MDN
            X(D,:) = X(D,:) + X_SHIFT(D)
         END DO
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, SIZE(X,2,INT64)
            X(:,D) = MATMUL(X(:,D), X_RESCALE(:,:))
         END DO
         ! Set all invalid values to zeros.
         WHERE (IS_NAN(X(:,:)) .OR. (.NOT. IS_FINITE(X(:,:))))
            X(:,:) = 0.0_RT
         END WHERE
      ELSE IF (CONFIG%MDN .GT. 0) THEN
         X_SHIFT(:) = 0.0_RT
         X_RESCALE(:,:) = 0.0_RT
         FORALL (D=1:CONFIG%MDN) X_RESCALE(D,D) = 1.0_RT
         ! Set all invalid values to zeros.
         WHERE (IS_NAN(X(:,:)) .OR. (.NOT. IS_FINITE(X(:,:))))
            X(:,:) = 0.0_RT
         END WHERE
      END IF
      ! XI
      IF ((.NOT. CONFIG%XI_NORMALIZED) .AND. (CONFIG%MDE .GT. 0)) THEN
         CALL RADIALIZE(X(CONFIG%MDN+ONE:CONFIG%MDN+CONFIG%MDE,:), XI_SHIFT(:), XI_RESCALE(:,:))
         ! Apply the shift to the source embeddings.
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, CONFIG%MDE
            M_EMB_VECS(D,:) = M_EMB_VECS(D,:) + XI_SHIFT(D)
         END DO
         ! Apply the transformation to the source embeddings.
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, SIZE(M_EMB_VECS,2,INT64)
            M_EMB_VECS(:,D) = MATMUL(M_EMB_VECS(:,D), XI_RESCALE(:,:))
         END DO
         ! Renormalize the embeddings to have a consistent maximum norm.
         SCALAR = MAXVAL(SUM(M_EMB_VECS(:,:)**2, 1))
         IF (SCALAR .GT. 0.0_RT) THEN
            M_EMB_VECS(:,:) = M_EMB_VECS(:,:) / SQRT(SCALAR)
         END IF
      END IF
      ! Y
      IF (CONFIG%DO .GT. CONFIG%DOE) THEN
         IF (.NOT. CONFIG%Y_NORMALIZED) THEN
            IF (CONFIG%RESCALE_Y) THEN
               TO_FLATTEN = CONFIG%DON
            ELSE
               TO_FLATTEN = 0
            END IF
            CALL RADIALIZE( &
                 Y(:TO_FLATTEN,:), &
                 Y_SHIFT(:), &
                 Y_SCALE(:,:), &
                 INVERSE=Y_RESCALE(:,:), &
                 MAX_TO_FLATTEN=TO_FLATTEN &
            )
            !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
            DO D = 1, CONFIG%DON
               Y(D,:) = Y(D,:) + Y_SHIFT(D)
            END DO
            !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
            DO D = 1, SIZE(Y,2,INT64)
               Y(:,D) = MATMUL(Y(:,D), Y_SCALE(:,:))
            END DO
            ! Set all invalid values to zeros.
            WHERE (IS_NAN(Y(:,:)) .OR. (.NOT. IS_FINITE(Y(:,:))))
               Y(:,:) = 0.0_RT
            END WHERE
         ELSE
            Y_SHIFT(:) = 0.0_RT
            Y_RESCALE(:,:) = 0.0_RT
            FORALL (D=1:SIZE(Y,1)) Y_RESCALE(D,D) = 1.0_RT
            ! Set all invalid values to zeros.
            WHERE (IS_NAN(Y(:,:)) .OR. (.NOT. IS_FINITE(Y(:,:))))
               Y(:,:) = 0.0_RT
            END WHERE
         END IF
      END IF
      ! YI
      IF ((.NOT. CONFIG%YI_NORMALIZED) .AND. (CONFIG%DOE .GT. 0)) THEN
         ! ASSUME that the "target" values for the output embeddings are populated already.
         ! 
         ! Transform the embeddings so they are "uniformly spaced" after
         !  considering their occurrence. They will be re-transformed to
         !  be unit length again later, but that is left as a TODO.
         CALL RADIALIZE(Y(CONFIG%DON+ONE:,:), YI_SHIFT(:), YI_RESCALE(:,:))
         ! Apply the shift to the source embeddings.
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, CONFIG%DOE
            O_EMB_VECS(D,:) = O_EMB_VECS(D,:) + YI_SHIFT(D)
         END DO
         ! Apply the transformation to the source embeddings.
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
         DO D = 1, SIZE(O_EMB_VECS,2,INT64)
            O_EMB_VECS(:,D) = MATMUL(O_EMB_VECS(:,D), YI_RESCALE(:,:))
         END DO
         ! Renormalize the embeddings to have a consistent maximum norm.
         SCALAR = MAXVAL(SUM(O_EMB_VECS(:,:)**2, 1))
         IF (SCALAR .GT. 0.0_RT) THEN
            O_EMB_VECS(:,:) = O_EMB_VECS(:,:) / SQRT(SCALAR)
         END IF
      END IF
      ! YW
      IF (SIZE(YW,KIND=INT64) .GT. 0) THEN
         ! Divide by the average YW to make its mean 1 (separately for negative and positive YW).
         YW_MASK(:,:) = (YW .GE. 0.0_RT)
         ! TODO: Use YW instead of directly operating on YW_IN (to conserve memory).
         WHERE (YW_MASK(:,:)) ! The denominator is never 0 when this clause is executed, because YW_MASK(...) is necessary.
            YW(:,:) = YW(:,:) / (SUM(YW(:,:), MASK=YW_MASK(:,:)) / REAL(COUNT(YW_MASK(:,:)),RT))
         ELSEWHERE ! The denominator is never 0 when this clause is executed, because .NOT. YW_MASK(...) is necessary.
            YW(:,:) = YW(:,:) / (SUM(YW(:,:), MASK=.NOT. YW_MASK(:,:)) / REAL(COUNT(.NOT. YW_MASK(:,:)),RT))
         END WHERE
         ! Set all invalid values to zero.
         WHERE (IS_NAN(YW(:,:)) .OR. (.NOT. IS_FINITE(YW(:,:))))
            YW(:,:) = 0.0_RT
         END WHERE
      END IF


    END SUBROUTINE NORMALIZE_STEP_UNPACKED
  END SUBROUTINE NORMALIZE_STEP


  ! Make inputs and outputs radially symmetric (to make initialization
  !  more well spaced and lower the curvature of the error gradient).
  SUBROUTINE NORMALIZE_DATA(CONFIG, MODEL, AGG_ITERATORS, &
       AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YI_IN, YW_IN, &
       AX, AXI, SIZES, X, XI, Y, YI, YW, &
       AX_SHIFT, AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_SHIFT, AY_SCALE, &
       X_SHIFT, X_RESCALE, XI_SHIFT, XI_RESCALE, &
       Y_SHIFT, Y_RESCALE, YI_SHIFT, YI_RESCALE, &
       A_EMB_VECS, M_EMB_VECS, O_EMB_VECS, A_STATES, AY, INFO)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: AGG_ITERATORS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: YI_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: AXI
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: YI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AX_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AXI_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AXI_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AY_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AY_SCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: X_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: XI_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: XI_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: Y_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: YI_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YI_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_EMB_VECS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: M_EMB_VECS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: O_EMB_VECS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:,:,:) :: A_STATES
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: AY
    INTEGER(KIND=INT32), INTENT(INOUT) :: INFO
    ! WARNING: Local variables for processing aggreagte values (and checking outputs).
    !          These will be sized as large as reasonably possible given memory limits.
    LOGICAL(KIND=C_BOOL), ALLOCATABLE, DIMENSION(:,:) :: YW_MASK ! LOCAL ALLOCATION
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:,:) :: Y_SCALE ! LOCAL ALLOCATION
    LOGICAL(KIND=C_BOOL) :: NORMALIZE
    INTEGER(KIND=INT64) :: I, D, E, NA, NM
    INTEGER :: TO_FLATTEN
    REAL(KIND=RT) :: SCALAR
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    NORMALIZE = CONFIG%NORMALIZE
    CONFIG%NORMALIZE = .FALSE.
    ! Allocate local variables.
    ALLOCATE( &
         YW_MASK(SIZE(YW,1), SIZE(YW,2)), &
         Y_SCALE(SIZE(Y_RESCALE,1), SIZE(Y_RESCALE,2)) &
    )
    ! Get some data to use in the normalization process.
    CALL FETCH_DATA(CONFIG, AGG_ITERATORS, &
         AX_IN, AX, AXI_IN, AXI, SIZES_IN, SIZES, &
         X_IN, X, XI_IN, XI, Y_IN, Y, YI_IN, YI, YW_IN, YW, NA, NM)
    ! Encode embeddings if they are provided.
    IF ((CONFIG%MDE + CONFIG%ADE .GT. 0) .AND. (&
         (.NOT. CONFIG%XI_NORMALIZED) .OR. (.NOT. CONFIG%AXI_NORMALIZED))) THEN
       CALL EMBED(CONFIG, MODEL, AXI(:,:NA), XI(:,:NM), AX(:,:NA), X(:,:NM))
    END IF
    ! AX
    IF ((.NOT. CONFIG%AX_NORMALIZED) .AND. (CONFIG%ADN .GT. 0)) THEN
       IF (CONFIG%RESCALE_AX) THEN
          TO_FLATTEN = CONFIG%ADN
       ELSE
          TO_FLATTEN = 0
       END IF
       CALL RADIALIZE(AX(:CONFIG%ADN,:NA), AX_SHIFT(:), AX_RESCALE(:,:), &
            MAX_TO_FLATTEN=TO_FLATTEN, MAXBOUND=.TRUE.)
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, CONFIG%ADN
          AX_IN(D,:) = AX_IN(D,:) + AX_SHIFT(D)
       END DO
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, SIZE(AX_IN,2,INT64)
          AX_IN(:,D) = MATMUL(AX_IN(:,D), AX_RESCALE(:,:))
       END DO
       ! Set all invalid values to zeros.
       WHERE (IS_NAN(AX_IN(:,:)) .OR. (.NOT. IS_FINITE(AX_IN(:,:))))
          AX_IN(:,:) = 0.0_RT
       END WHERE
       CONFIG%AX_NORMALIZED = .TRUE.
    ELSE IF (CONFIG%ADN .GT. 0) THEN
       AX_SHIFT(:) = 0.0_RT
       AX_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:CONFIG%ADN) AX_RESCALE(D,D) = 1.0_RT
       ! Set all invalid values to zeros.
       WHERE (IS_NAN(AX_IN(:,:)) .OR. (.NOT. IS_FINITE(AX_IN(:,:))))
          AX_IN(:,:) = 0.0_RT
       END WHERE
    END IF
    ! AXI
    IF ((.NOT. CONFIG%AXI_NORMALIZED) .AND. (CONFIG%ADE .GT. 0)) THEN
       CALL RADIALIZE(AX(CONFIG%ADN+1:CONFIG%ADN+CONFIG%ADE,:NA), AXI_SHIFT(:), AXI_RESCALE(:,:))
       ! Apply the shift to the source embeddings.
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, CONFIG%ADE
          A_EMB_VECS(D,:) = A_EMB_VECS(D,:) + AXI_SHIFT(D)
       END DO
       ! Apply the transformation to the source embeddings.
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, SIZE(A_EMB_VECS,2,INT64)
          A_EMB_VECS(:,D) = MATMUL(A_EMB_VECS(:,D), AXI_RESCALE(:,:))
       END DO
       ! Renormalize the embeddings to have a consistent maximum norm.
       SCALAR = MAXVAL(SUM(A_EMB_VECS(:,:)**2, 1))
       IF (SCALAR .GT. 0.0_RT) THEN
          A_EMB_VECS(:,:) = A_EMB_VECS(:,:) / SQRT(SCALAR)
       END IF
       CONFIG%AXI_NORMALIZED = .TRUE.
    END IF
    ! X
    IF ((.NOT. CONFIG%X_NORMALIZED) .AND. (CONFIG%MDN .GT. 0)) THEN
       IF (CONFIG%RESCALE_X) THEN
          TO_FLATTEN = CONFIG%MDN
       ELSE
          TO_FLATTEN = 0
       END IF
       CALL RADIALIZE(X(:CONFIG%MDN,:NM), X_SHIFT(:), X_RESCALE(:,:), &
            MAX_TO_FLATTEN=TO_FLATTEN, MAXBOUND=.TRUE.)
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, CONFIG%MDN
          X_IN(D,:) = X_IN(D,:) + X_SHIFT(D)
       END DO
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, SIZE(X_IN,2,INT64)
          X_IN(:,D) = MATMUL(X_IN(:,D), X_RESCALE(:,:))
       END DO
       CONFIG%X_NORMALIZED = .TRUE.
       ! Set all invalid values to zeros.
       WHERE (IS_NAN(X_IN(:,:)) .OR. (.NOT. IS_FINITE(X_IN(:,:))))
          X_IN(:,:) = 0.0_RT
       END WHERE
    ELSE IF (CONFIG%MDN .GT. 0) THEN
       X_SHIFT(:) = 0.0_RT
       X_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:CONFIG%MDN) X_RESCALE(D,D) = 1.0_RT
       ! Set all invalid values to zeros.
       WHERE (IS_NAN(X_IN(:,:)) .OR. (.NOT. IS_FINITE(X_IN(:,:))))
          X_IN(:,:) = 0.0_RT
       END WHERE
    END IF
    ! XI
    IF ((.NOT. CONFIG%XI_NORMALIZED) .AND. (CONFIG%MDE .GT. 0)) THEN
       CALL RADIALIZE(X(CONFIG%MDN+ONE:CONFIG%MDN+CONFIG%MDE,:NM), XI_SHIFT(:), XI_RESCALE(:,:))
       ! Apply the shift to the source embeddings.
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, CONFIG%MDE
          M_EMB_VECS(D,:) = M_EMB_VECS(D,:) + XI_SHIFT(D)
       END DO
       ! Apply the transformation to the source embeddings.
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, SIZE(M_EMB_VECS,2,INT64)
          M_EMB_VECS(:,D) = MATMUL(M_EMB_VECS(:,D), XI_RESCALE(:,:))
       END DO
       ! Renormalize the embeddings to have a consistent maximum norm.
       SCALAR = MAXVAL(SUM(M_EMB_VECS(:,:)**2, 1))
       IF (SCALAR .GT. 0.0_RT) THEN
          M_EMB_VECS(:,:) = M_EMB_VECS(:,:) / SQRT(SCALAR)
       END IF
       CONFIG%XI_NORMALIZED = .TRUE.
    END IF
    ! Y
    IF (CONFIG%DO .GT. CONFIG%DOE) THEN
       IF (.NOT. CONFIG%Y_NORMALIZED) THEN
          IF (CONFIG%RESCALE_Y) THEN
             TO_FLATTEN = CONFIG%DON
          ELSE
             TO_FLATTEN = 0
          END IF
          CALL RADIALIZE(Y(:TO_FLATTEN,:NM), Y_SHIFT(:), Y_SCALE(:,:), &
               INVERSE=Y_RESCALE(:,:), MAX_TO_FLATTEN=TO_FLATTEN)
          !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
          DO D = 1, CONFIG%DON
             Y_IN(D,:) = Y_IN(D,:) + Y_SHIFT(D)
          END DO
          !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
          DO D = 1, SIZE(Y_IN,2,INT64)
             Y_IN(:,D) = MATMUL(Y_IN(:,D), Y_SCALE(:,:))
          END DO
          ! Set all invalid values to zeros.
          WHERE (IS_NAN(Y_IN(:,:)) .OR. (.NOT. IS_FINITE(Y_IN(:,:))))
             Y_IN(:,:) = 0.0_RT
          END WHERE
          CONFIG%Y_NORMALIZED = .TRUE.
       ELSE
          Y_SHIFT(:) = 0.0_RT
          Y_RESCALE(:,:) = 0.0_RT
          FORALL (D=1:SIZE(Y,1)) Y_RESCALE(D,D) = 1.0_RT
          ! Set all invalid values to zeros.
          WHERE (IS_NAN(Y_IN(:,:)) .OR. (.NOT. IS_FINITE(Y_IN(:,:))))
             Y_IN(:,:) = 0.0_RT
          END WHERE
       END IF
    END IF
    ! YI
    IF ((.NOT. CONFIG%YI_NORMALIZED) .AND. (CONFIG%DOE .GT. 0)) THEN
       ! Populate the "target" values for the output embeddings.
       Y(CONFIG%DON+ONE:,:NM) = 0.0_RT
       DO I = 1, NM
          DO D = 1, SIZE(YI,1,KIND=INT64)
             E = YI(D,I)
             Y(CONFIG%DON+ONE:,I) = Y(CONFIG%DON+ONE:,I) + O_EMB_VECS(:,E)
          END DO
       END DO
       ! Transform the embeddings so they are "uniformly spaced" after
       !  considering their occurrence. They will be re-transformed to
       !  be unit length again later, but that is left as a TODO.
       CALL RADIALIZE(Y(CONFIG%DON+ONE:,:NM), YI_SHIFT(:), YI_RESCALE(:,:))
       ! Apply the shift to the source embeddings.
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, CONFIG%DOE
          O_EMB_VECS(D,:) = O_EMB_VECS(D,:) + YI_SHIFT(D)
       END DO
       ! Apply the transformation to the source embeddings.
       !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
       DO D = 1, SIZE(O_EMB_VECS,2,INT64)
          O_EMB_VECS(:,D) = MATMUL(O_EMB_VECS(:,D), YI_RESCALE(:,:))
       END DO
       ! Renormalize the embeddings to have a consistent maximum norm.
       SCALAR = MAXVAL(SUM(O_EMB_VECS(:,:)**2, 1))
       IF (SCALAR .GT. 0.0_RT) THEN
          O_EMB_VECS(:,:) = O_EMB_VECS(:,:) / SQRT(SCALAR)
       END IF
       CONFIG%YI_NORMALIZED = .TRUE.
    END IF
    ! YW
    IF (SIZE(YW_IN,KIND=INT64) .GT. 0) THEN
       ! Divide by the average YW to make its mean 1 (separately for negative and positive YW).
       YW_MASK(:,:) = (YW_IN .GE. 0.0_RT)
       ! TODO: Use YW instead of directly operating on YW_IN (to conserve memory).
       WHERE (YW_MASK(:,:)) ! The denominator is never 0 when this clause is executed, because YW_MASK(...) is necessary.
          YW_IN(:,:) = YW_IN(:,:) / (SUM(YW_IN(:,:), MASK=YW_MASK(:,:)) / REAL(COUNT(YW_MASK(:,:)),RT))
       ELSEWHERE ! The denominator is never 0 when this clause is executed, because .NOT. YW_MASK(...) is necessary.
          YW_IN(:,:) = YW_IN(:,:) / (SUM(YW_IN(:,:), MASK=.NOT. YW_MASK(:,:)) / REAL(COUNT(.NOT. YW_MASK(:,:)),RT))
       END WHERE  
       ! Set all invalid values to zero.
       WHERE (IS_NAN(YW_IN(:,:)) .OR. (.NOT. IS_FINITE(YW_IN(:,:))))
          YW_IN(:,:) = 0.0_RT
       END WHERE
    END IF
    ! 
    ! Normalize AY (AX must already be normalized, EVALUATE contains parallelization).
    IF ((.NOT. CONFIG%AY_NORMALIZED) .AND. (CONFIG%ADO .GT. 0)) THEN
       AY_SHIFT(:) = 0.0_RT
       ! Only apply the normalization to AY if there is a model afterwards.
       IF (CONFIG%MDO .GT. 0) THEN
          ! Encode embeddings if they are provided.
          IF (CONFIG%ADE .GT. 0) THEN
             D = CONFIG%MDE ; CONFIG%MDE = 0
             CALL EMBED(CONFIG, MODEL, AXI(:,:NA), XI(:,:NM), AX(:,:NA), X(:,:NM))
             CONFIG%MDE = INT(D, KIND=INT32)
          END IF
          ! Compute the beginning of ADO storage in X
          E = CONFIG%MDN + CONFIG%MDE + 1 
          ! Disable "model" evaluation for this forward pass.
          !   (Give "A_STATES" for the "M_STATES" argument, since it will not be used.)
          D = CONFIG%MDO ; CONFIG%MDO = 0
          CALL EVALUATE(CONFIG, MODEL, AX(:,:NA), AY(:NA,:), SIZES(:), &
               X(:,:NM), & ! This is never used when MDO = 0.
               X(E:,:NM), & ! This is "Y", where the mean-aggregated outputs will be stored.
               A_STATES(:NA,:,:,:), A_STATES(:NA,:,:,:), INFO)
          CONFIG%MDO = INT(D, KIND=INT32)
          ! Compute AY shift as the mean of mean-aggregated outputs, apply it.
          AY_SHIFT(:) = SUM(X(E:,:NM),2) / REAL(NM,RT)
          ! Apply the shift to the data before computing the variance.
          !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
          DO D = 0, CONFIG%ADO-1
             X(E+D,:NM) = X(E+D,:NM) - AY_SHIFT(D+1)
          END DO
          ! Compute the AY scale as the standard deviation of mean-aggregated outputs.
          AY_SCALE(:) = SUM(X(E:,:NM)**2,2) / REAL(NM,RT)
          ! Guard for potential 0 values in the output standard deviations.
          AY_SCALE(:) = 1.0_RT / SQRT(MAX(AY_SCALE(:), SQRT(EPSILON(0.0_RT))))
          ! Apply the scale to the data.
          !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS)
          DO D = 0, CONFIG%ADO-1
             X(E+D,:NM) = X(E+D,:NM) * AY_SCALE(D+1)
          END DO
       END IF
       CONFIG%AY_NORMALIZED = .TRUE.
    ELSE
       AY_SCALE(:) = 1.0_RT
    END IF
    ! Reset the normalize setting.
    CONFIG%NORMALIZE = NORMALIZE
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    !$OMP CRITICAL
    CONFIG%WNRM = CONFIG%WNRM + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CNRM = CONFIG%CNRM + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    !$OMP END CRITICAL
    ! Deallocate local variables.
    DEALLOCATE(YW_MASK, Y_SCALE)
  END SUBROUTINE NORMALIZE_DATA

  
  ! Performing conditioning related operations on this model 
  !  (ensure that mean squared error is reducible).
  SUBROUTINE CONDITION_MODEL(CONFIG, MODEL, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, &
       AX, AXI, AY, AY_GRADIENT, SIZES, X, XI, Y_GRADIENT, &
       NUM_THREADS, FIT_STEP, &
       A_STATES, M_STATES, A_GRADS, M_GRADS, &
       A_LENGTHS, M_LENGTHS, A_STATE_TEMP, M_STATE_TEMP, A_ORDER, M_ORDER, &
       TOTAL_EVAL_RANK, TOTAL_GRAD_RANK)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL_GRAD_MEAN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL_GRAD_CURV
    ! Data.
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AY
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AY_GRADIENT
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y_GRADIENT
    ! Configuration.
    INTEGER(KIND=INT64), INTENT(IN) :: NUM_THREADS, FIT_STEP
    ! States, gradients, lengths, temporary storage, and order (of ranks).
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: A_STATES, M_STATES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: A_GRADS, M_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_LENGTHS, M_LENGTHS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_STATE_TEMP, M_STATE_TEMP
    INTEGER(KIND=INT32), INTENT(INOUT), DIMENSION(:,:) :: A_ORDER, M_ORDER
    INTEGER(KIND=INT32), INTENT(INOUT) :: TOTAL_EVAL_RANK, TOTAL_GRAD_RANK
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! 
    ! Maintain a constant max-norm across the magnitue of input and internal vectors.
    ! 
    CALL UNIT_MAX_NORM(CONFIG, &
         MODEL(CONFIG%AECS:CONFIG%AECE), & ! A embeddings mean
         MODEL(CONFIG%ASEV:CONFIG%AEEV), & ! A embeddings
         AX(CONFIG%ADN+ONE:,:), & ! A embedded values
         MODEL(CONFIG%ASIV:CONFIG%AEIV), & ! A input vecs
         MODEL(CONFIG%ASIS:CONFIG%AEIS), & ! A input shift
         MODEL(CONFIG%ASSV:CONFIG%AESV), & ! A state vecs
         MODEL(CONFIG%ASSS:CONFIG%AESS), & ! A state shift
         MODEL(CONFIG%AOSS:CONFIG%AOSE), & ! AY shift
         MODEL(CONFIG%AOMS:CONFIG%AOME), & ! AY scale
         AY(:,:), & ! AY values (to update shift)
         SIZES(:), & ! Aggregate set sizes (for AY shift computation).
         X(CONFIG%MDN+CONFIG%MDE+ONE:CONFIG%MDN+CONFIG%MDE+CONFIG%ADO,:), & ! AY aggregated
         MODEL(CONFIG%MECS:CONFIG%MECE), & ! M embeddings mean
         MODEL(CONFIG%MSEV:CONFIG%MEEV), & ! M embeddings
         X(CONFIG%MDN+ONE:CONFIG%MDN+CONFIG%MDE,:), & ! M embedded values
         MODEL(CONFIG%MSIV:CONFIG%MEIV), & ! M input vecs
         MODEL(CONFIG%MSIS:CONFIG%MEIS), & ! M input shift
         MODEL(CONFIG%MSSV:CONFIG%MESV), & ! M state vecs
         MODEL(CONFIG%MSSS:CONFIG%MESS), & ! M state shift
         MODEL(CONFIG%OSEV:CONFIG%OEEV)) ! Output embeddings
    ! 
    ! Check rank and optionally replace redundant basis functions.
    ! 
    IF ((CONFIG%RANK_CHECK_FREQUENCY .GT. 0) .AND. &
         (MOD(FIT_STEP-1,CONFIG%RANK_CHECK_FREQUENCY) .EQ. 0)) THEN
       ! 
       ! TODO: Model variables have stepped, so the embeddings changed.
       !       Should the following code be re-embedding new values or leaving
       !       the old ones in place? Maybe this call to EMBED is unadvisable.
       ! 
       ! Embed all integer inputs into real vector inputs.
       CALL EMBED(CONFIG, MODEL, AXI, XI, AX, X)
       ! Compute total rank for values at all internal layers.
       TOTAL_EVAL_RANK = 0
       TOTAL_GRAD_RANK = 0
       ! Update for the aggregator model.
       CALL CHECK_MODEL_RANK( &
            CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ANC, CONFIG%ADSO, CONFIG%ADO, INT(NUM_THREADS), &
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
            A_STATE_TEMP(:,:), A_STATES(:,:,:,:), A_LENGTHS(:,:), A_ORDER(:,:), A_GRADS(:,:,:,:))
       ! Update for the fixed model.
       CALL CHECK_MODEL_RANK( &
            CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MNC, CONFIG%MDSO, CONFIG%MDO, INT(NUM_THREADS), &
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
            M_STATE_TEMP(:,:), M_STATES(:,:,:,:), M_LENGTHS(:,:), M_ORDER(:,:), M_GRADS(:,:,:,:))
    END IF
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    !$OMP CRITICAL
    CONFIG%WCON = CONFIG%WCON + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CCON = CONFIG%CCON + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    !$OMP END CRITICAL

  CONTAINS

    ! Make max length vector in each weight matrix have unit length.
    SUBROUTINE UNIT_MAX_NORM(CONFIG, &
         A_EMBEDDINGS_MEAN, A_EMBEDDINGS, A_EMBEDDED_VALUES, A_INPUT_VECS, A_INPUT_SHIFT, &
         A_STATE_VECS, A_STATE_SHIFT, AY_SHIFT, AY_SCALE, AY, SIZES, AY_AGGREGATED, &
         M_EMBEDDINGS_MEAN, M_EMBEDDINGS, M_EMBEDDED_VALUES, M_INPUT_VECS, M_INPUT_SHIFT, &
         M_STATE_VECS, M_STATE_SHIFT, O_EMB_VECS)
      TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADE) :: A_EMBEDDINGS_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADE, CONFIG%ANE) :: A_EMBEDDINGS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_EMBEDDED_VALUES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADI, CONFIG%ADS, CONFIG%ANC) :: A_INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS, CONFIG%ANC) :: A_INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS, CONFIG%ADS, MAX(0,CONFIG%ANS-1), CONFIG%ANC) :: A_STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS, MAX(0,CONFIG%ANS-1), CONFIG%ANC) :: A_STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADO) :: AY_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADO) :: AY_SCALE
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AY
      INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY_AGGREGATED
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDE) :: M_EMBEDDINGS_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDE, CONFIG%MNE) :: M_EMBEDDINGS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: M_EMBEDDED_VALUES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDI, CONFIG%MDS, CONFIG%MNC) :: M_INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDS, CONFIG%MNC) :: M_INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDS, CONFIG%MDS, MAX(0,CONFIG%MNS-1), CONFIG%MNC) :: M_STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%MDS, MAX(0,CONFIG%MNS-1), CONFIG%MNC) :: M_STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%DOE, CONFIG%NOE) :: O_EMB_VECS
      ! Local variables.
      INTEGER(KIND=INT64) :: L, C, D, N
      REAL(KIND=RT) :: SCALAR
      REAL(KIND=RT), ALLOCATABLE, DIMENSION(:) :: AYX_MEAN, A_EMB_MEAN, M_EMB_MEAN
      ! TODO: Variables for normalizing output embeddings.
      ! REAL(KIND=RT), DIMENSION(CONFIG%DOE,CONFIG%DOE) :: O_ROT, O_TMP
      ! REAL(KIND=RT), DIMENSION(CONFIG%DOE) :: O_LENS
      ! INTEGER, DIMENSION(CONFIG%DOE) :: O_ORDER
      ! INTEGER :: O_RANK
      !
      ! Limit the maximum 2-norm of any state projection vector to 1.
      !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) PRIVATE(L, D, C, SCALAR)
      DO L = 1, CONFIG%MNS+CONFIG%ANS+ONE
         ! [1,ANS-1] -> A_STATE_VECS
         IF (L .LT. CONFIG%ANS) THEN
            DO C = 1, CONFIG%ANC
               SCALAR = SQRT(MAXVAL(SUM(A_STATE_VECS(:,:,L,C)**2, 1)))
               A_STATE_VECS(:,:,L,C) = A_STATE_VECS(:,:,L,C) / SCALAR
               A_STATE_SHIFT(:,L,C) = A_STATE_SHIFT(:,L,C) / SCALAR
            END DO
         ! [ANS] -> A_INPUT_VECS
         ELSE IF (L .EQ. CONFIG%ANS) THEN
            DO C = 1, CONFIG%ANC
               SCALAR = SQRT(MAXVAL(SUM(A_INPUT_VECS(:,:,C)**2, 1)))
               A_INPUT_VECS(:,:,C) = A_INPUT_VECS(:,:,C) / SCALAR
               A_INPUT_SHIFT(:,C) = A_INPUT_SHIFT(:,C) / SCALAR
            END DO
         ! [ANS+1, ANS+MNS-1] -> M_STATE_VECS
         ELSE IF (L-CONFIG%ANS .LT. CONFIG%MNS) THEN
            DO C = 1, CONFIG%MNC
               SCALAR = SQRT(MAXVAL(SUM(M_STATE_VECS(:,:,L-CONFIG%ANS,C)**2, 1)))
               M_STATE_VECS(:,:,L-CONFIG%ANS,C) = M_STATE_VECS(:,:,L-CONFIG%ANS,C) / SCALAR
               M_STATE_SHIFT(:,L-CONFIG%ANS,C) = M_STATE_SHIFT(:,L-CONFIG%ANS,C) / SCALAR
            END DO
         ! [ANS+MNS] -> M_INPUT_VECS
         ELSE IF (L-CONFIG%ANS .EQ. CONFIG%MNS) THEN
            DO C = 1, CONFIG%MNC
               SCALAR = SQRT(MAXVAL(SUM(M_INPUT_VECS(:,:,C)**2, 1)))
               M_INPUT_VECS(:,:,C) = M_INPUT_VECS(:,:,C) / SCALAR
               M_INPUT_SHIFT(:,C) = M_INPUT_SHIFT(:,C) / SCALAR
            END DO
         ! [ANS+MNS+1] -> O_EMB_VECS
         ELSE
            SCALAR = SQRT(MAXVAL(SUM(O_EMB_VECS(:,:)**2, 1)))
            O_EMB_VECS(:,:) = O_EMB_VECS(:,:) / SCALAR
         END IF
      END DO
      ! 
      ! Update the aggregator model output shift to produce componentwise mean-zero
      !  unit variance values (prevent divergence), but only when there is a model afterwards. 
      IF ((CONFIG%MDO .GT. 0) .AND. (CONFIG%ADO .GT. 0) .AND. &
           (SIZE(AY,1,INT64) .GT. ZERO) .AND. (CONFIG%STEP_AY_CHANGE .GT. 0.0_RT)) THEN
         ALLOCATE(AYX_MEAN(1:CONFIG%ADO))
         ! AY->X mean.
         AYX_MEAN(:) = 0.0_RT
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) REDUCTION(+:AYX_MEAN)
         DO N = 1, SIZE(SIZES,KIND=INT64)
            IF (SIZES(N) .GT. ZERO) THEN
               AYX_MEAN(:) = AYX_MEAN(:) + AY_AGGREGATED(:,N)
            END IF
         END DO
         AYX_MEAN(:) = AYX_MEAN(:) / REAL(SIZE(SIZES,KIND=INT64),RT)
         WHERE ((.NOT. IS_FINITE(AYX_MEAN(:))) .OR. IS_NAN(AYX_MEAN(:)))
            AYX_MEAN(:) = 0.0
         END WHERE
         ! Update the shift term.
         IF (CONFIG%STEP_AY_CHANGE .LT. 1.0_RT) THEN
            AY_SHIFT(:) = &
                 (1.0_RT - CONFIG%STEP_AY_CHANGE) * AY_SHIFT(:) &
               + (CONFIG%STEP_AY_CHANGE         ) * (AY_SHIFT(:) + AYX_MEAN(:) / AY_SCALE(:))
         ELSE
            AY_SHIFT(:) = AY_SHIFT(:) + AYX_MEAN(:)
         END IF
         ! 
         ! AY->X variance.
         AYX_MEAN(:) = 0.0_RT
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) REDUCTION(+:AYX_MEAN)
         DO N = 1, SIZE(SIZES,KIND=INT64)
            IF (SIZES(N) .GT. ZERO) THEN
               AYX_MEAN(:) = AYX_MEAN(:) + AY_AGGREGATED(:,N)**2
            END IF
         END DO
         AYX_MEAN(:) = AYX_MEAN(:) / REAL(SIZE(SIZES,KIND=INT64),RT)
         WHERE ((.NOT. IS_FINITE(AYX_MEAN(:))) .OR. IS_NAN(AYX_MEAN(:)))
            AYX_MEAN(:) = 1.0_RT
         END WHERE
         AYX_MEAN(:) = SQRT(MAX(AYX_MEAN(:), SQRT(EPSILON(0.0_RT)))) ! Convert to deviation.
         ! Update the scale term.
         IF (CONFIG%STEP_AY_CHANGE .LT. 1.0_RT) THEN
            AY_SCALE(:) = &
                   (1.0_RT - CONFIG%STEP_AY_CHANGE) * AY_SCALE(:) &
                 + (CONFIG%STEP_AY_CHANGE         ) * AY_SCALE(:) / AYX_MEAN(:)
         ELSE
            AY_SCALE(:) = 1.0_RT / AYX_MEAN(:)
         END IF
         !
         DEALLOCATE(AYX_MEAN)
      END IF
      ! 
      ! TODO: Embedding normalization should probably fall under "input conditioning"
      !       instead of falling under "model conditioning"
      ! 
      ! A_EMBEDDINGS  sliding window mean zero variance one
      IF ((CONFIG%ANE .GT. 0) .AND. (CONFIG%STEP_EMB_CHANGE .GT. 0.0_RT)) THEN
         ! Update the exponential trailing mean term and subtract it from current values.
         ! WARNING: Local allocation.
         ALLOCATE(A_EMB_MEAN(1:CONFIG%ADE))
         A_EMB_MEAN(:) = 0.0_RT
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) REDUCTION(+:A_EMB_MEAN)
         DO N = 1, SIZE(A_EMBEDDED_VALUES,2,KIND=INT64)
            A_EMB_MEAN(:) = A_EMB_MEAN(:) + A_EMBEDDED_VALUES(:,N)
         END DO
         A_EMB_MEAN(:) = A_EMB_MEAN(:) / REAL(SIZE(A_EMBEDDED_VALUES,2,KIND=INT64), KIND=RT)
         ! Update the embeddings center (and in turn the shift).
         A_EMBEDDINGS_MEAN(:) = &
              (1.0_RT - CONFIG%STEP_EMB_CHANGE) * A_EMBEDDINGS_MEAN(:) + &
              (         CONFIG%STEP_EMB_CHANGE) * A_EMB_MEAN(:)
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS)
         DO D = 1, SIZE(A_EMBEDDINGS,1,KIND=INT64)
            A_EMBEDDINGS(D,:) = A_EMBEDDINGS(D,:) - A_EMB_MEAN(D)
         END DO
         ! Update the scale so that the variance of the embedded values is 1.
         A_EMB_MEAN(:) = 0.0_RT
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) REDUCTION(+:A_EMB_MEAN)
         DO N = 1, SIZE(A_EMBEDDED_VALUES,2,KIND=INT64)
            A_EMB_MEAN(:) = A_EMB_MEAN(:) + A_EMBEDDED_VALUES(:,N)**2
         END DO
         A_EMB_MEAN(:) = A_EMB_MEAN(:) / REAL(SIZE(A_EMBEDDED_VALUES,2,KIND=INT64),KIND=RT)
         WHERE ((.NOT. IS_FINITE(A_EMB_MEAN(:))) .OR. IS_NAN(A_EMB_MEAN(:)))
            A_EMB_MEAN(:) = 1.0_RT
         END WHERE
         A_EMB_MEAN(:) = SQRT(MAX(A_EMB_MEAN(:), SQRT(EPSILON(0.0_RT))))
         A_EMB_MEAN(:) = MAX(A_EMB_MEAN(:), 0.5_RT)
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS)
         DO D = 1, CONFIG%ANE
            A_EMBEDDINGS(:,D) = &
                 (1.0_RT - CONFIG%STEP_EMB_CHANGE) * A_EMBEDDINGS(:,D) &
                 + (CONFIG%STEP_EMB_CHANGE       ) * A_EMBEDDINGS(:,D) / A_EMB_MEAN(:)
         END DO
         DEALLOCATE(A_EMB_MEAN)
      END IF
      ! M_EMBEDDINGS  sliding window mean zero variance one
      IF ((CONFIG%MNE .GT. 0) .AND. (CONFIG%STEP_EMB_CHANGE .GT. 0.0_RT)) THEN
         ! Update the exponential trailing mean term and subtract it from current values.
         ! WARNING: Local allocation.
         ALLOCATE(M_EMB_MEAN(1:CONFIG%MDE))
         M_EMB_MEAN(:) = 0.0_RT
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) REDUCTION(+:M_EMB_MEAN)
         DO N = 1, SIZE(M_EMBEDDED_VALUES,2,KIND=INT64)
            M_EMB_MEAN(:) = M_EMB_MEAN(:) + M_EMBEDDED_VALUES(:,N)
         END DO
         M_EMB_MEAN(:) = M_EMB_MEAN(:) / REAL(SIZE(M_EMBEDDED_VALUES,2,KIND=INT64), KIND=RT)
         ! Update the embeddings center (and in turn the shift).
         M_EMBEDDINGS_MEAN(:) = &
              (1.0_RT - CONFIG%STEP_EMB_CHANGE) * M_EMBEDDINGS_MEAN(:) + &
              (         CONFIG%STEP_EMB_CHANGE) * M_EMB_MEAN(:)
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS)
         DO D = 1, SIZE(M_EMBEDDINGS,1,KIND=INT64)
            M_EMBEDDINGS(D,:) = M_EMBEDDINGS(D,:) - M_EMB_MEAN(D)
         END DO
         ! Update the scale so that the variance of the embedded values is 1.
         M_EMB_MEAN(:) = 0.0_RT
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS) REDUCTION(+:M_EMB_MEAN)
         DO N = 1, SIZE(M_EMBEDDED_VALUES,2,KIND=INT64)
            M_EMB_MEAN(:) = M_EMB_MEAN(:) + M_EMBEDDED_VALUES(:,N)**2
         END DO
         M_EMB_MEAN(:) = M_EMB_MEAN(:) / REAL(SIZE(M_EMBEDDED_VALUES,2,KIND=INT64),KIND=RT)
         WHERE ((.NOT. IS_FINITE(M_EMB_MEAN(:))) .OR. IS_NAN(M_EMB_MEAN(:)))
            M_EMB_MEAN(:) = 1.0_RT
         END WHERE
         M_EMB_MEAN(:) = SQRT(MAX(M_EMB_MEAN(:), SQRT(EPSILON(0.0_RT))))
         M_EMB_MEAN(:) = MAX(M_EMB_MEAN(:), 0.5_RT)
         !$OMP PARALLEL DO NUM_THREADS(NUM_THREADS)
         DO D = 1, CONFIG%MNE
            M_EMBEDDINGS(:,D) = &
                 (1.0_RT - CONFIG%STEP_EMB_CHANGE) * M_EMBEDDINGS(:,D) &
                 + (CONFIG%STEP_EMB_CHANGE       ) * M_EMBEDDINGS(:,D) / M_EMB_MEAN(:)
         END DO
         DEALLOCATE(M_EMB_MEAN)
      END IF
      ! ! O_EMB_VECS  shifted such that the first DOE embeddings are the "root".
      ! IF (CONFIG%DOE .GT. 0) THEN
      !    O_ROT(:,:) = O_EMB_VECS(1:CONFIG%DOE,1:CONFIG%DOE) ! Copy first DOE embeddings into matrix.
      !    FORALL (D=1:CONFIG%DOE) O_ORDER(D) = INT(D) ! Initialize original positions of vectors.
      !    CALL ORTHONORMALIZE(O_ROT, O_LENS, O_RANK, O_ORDER) ! Perform orthonormalization.
      !    O_TMP(:,O_ORDER(:)) = O_ROT(:,:) ! Copy vectors into their original order.
      !    O_ROT(:,:) = O_TMP(:,:)
      !    ! Apply the rotation.
      !    O_EMB_VECS(:,:) = MATMUL(TRANSPOSE(O_ROT(:,:)), O_EMB_VECS(:,:))
      ! END IF
    END SUBROUTINE UNIT_MAX_NORM

    ! Check the rank of all internal states.
    SUBROUTINE CHECK_MODEL_RANK(DI, DS, NS, NC, DSO, DO, NUM_THREADS, X, Y_GRADIENT, &
         INPUT_VECS, INPUT_SHIFT, STATE_VECS, STATE_SHIFT, OUTPUT_VECS, &
         INPUT_VECS_GRAD_MEAN, INPUT_SHIFT_GRAD_MEAN, STATE_VECS_GRAD_MEAN, STATE_SHIFT_GRAD_MEAN, OUTPUT_VECS_GRAD_MEAN, &
         INPUT_VECS_GRAD_CURV, INPUT_SHIFT_GRAD_CURV, STATE_VECS_GRAD_CURV, STATE_SHIFT_GRAD_CURV, OUTPUT_VECS_GRAD_CURV, &
         STATE_TEMP, STATES, LENGTHS, ORDER, GRADS)
      INTEGER(KIND=INT32), INTENT(IN) :: DI, DS, NS, NC, DSO, DO, NUM_THREADS
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X, Y_GRADIENT
      ! Model variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DI, DS, NC) :: INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, NC) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, DS, MAX(0,NS-1), NC) :: STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, MAX(0,NS-1), NC) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DSO, DO, NC) :: OUTPUT_VECS
      ! Gradient means for all variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DI, DS, NC) :: INPUT_VECS_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, NC) :: INPUT_SHIFT_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, DS, MAX(0,NS-1), NC) :: STATE_VECS_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, MAX(0,NS-1), NC) :: STATE_SHIFT_GRAD_MEAN
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DSO, DO, NC) :: OUTPUT_VECS_GRAD_MEAN
      ! Gradient curvatures for all variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DI, DS, NC) :: INPUT_VECS_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, NC) :: INPUT_SHIFT_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, DS, MAX(0,NS-1), NC) :: STATE_VECS_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DS, MAX(0,NS-1), NC) :: STATE_SHIFT_GRAD_CURV
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(DSO, DO, NC) :: OUTPUT_VECS_GRAD_CURV
      ! Temporary variables.
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: STATE_TEMP
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: STATES
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: LENGTHS
      INTEGER(KIND=INT32), INTENT(INOUT), DIMENSION(:,:) :: ORDER
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:,:) :: GRADS
      INTEGER(KIND=INT32) :: BATCH, BS, BE, BN, C, I, N, NT, TER, TGR
      ! TODO: This allocation should occur at workspace initialization.
      INTEGER(KIND=INT32), DIMENSION(DS, NUM_THREADS) :: STATE_USAGE ! LOCAL ALLOCATION
      ! 
      ! Batch computation formula.
       IF (NS .GT. 0) THEN
          N = SIZE(STATE_TEMP,1)
          NT = MIN(NUM_THREADS, MAX(1, N / DS)) ! number of threads (as not to artificially reduce rank)
          BN = (N + NT - 1) / NT ! = CEIL(N / NT)
       END IF
       DO C = 1, NC
          DO I = 1, NS
             STATE_USAGE(:,:) = 0
             TER = 0; TGR = 0;
             !$OMP PARALLEL DO PRIVATE(BATCH,BS,BE) NUM_THREADS(NT) &
             !$OMP& REDUCTION(MAX: TER, TGR)
             DO BATCH = 1, NT
                BS = BN*(BATCH-1) + 1
                BE = MIN(N, BN*BATCH)
                ! Compute model state rank.
                STATE_TEMP(BS:BE,:) = STATES(BS:BE,:,I,C)
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
                CALL ORTHONORMALIZE(STATE_TEMP(BS:BE,:), LENGTHS(:,BATCH), TER, ORDER(:,BATCH))
                STATE_USAGE(ORDER(:TER,BATCH),BATCH) = 1
                ! Compute grad state rank.
                STATE_TEMP(BS:BE,:) = GRADS(BS:BE,:,I,C)
                CALL ORTHONORMALIZE(STATE_TEMP(BS:BE,:), LENGTHS(:,BATCH), TGR, ORDER(:,BATCH))
             END DO
             TOTAL_EVAL_RANK = TOTAL_EVAL_RANK + TER
             TOTAL_GRAD_RANK = TOTAL_GRAD_RANK + TGR
             ! --------------------------------------------------------------------------------
             ! ! If basis replacement is enabled..
             ! IF (CONFIG%BASIS_REPLACEMENT) THEN
             !    ! Sum the "usage" of internal nodes to see which are entirely unuseful.
             !    STATE_USAGE(:,1) = SUM(STATE_USAGE(:,:), 2)
             !    ! Replace the basis functions with a policy that prevents rank collapse.
             !    IF (I .EQ. 1) THEN
             !       IF (NS .GT. 1) THEN
             !          CALL REPLACE_BASIS_FUNCTIONS( &
             !               STATE_USAGE(:,1), &
             !               X(:,:), &
             !               STATES(:,:,I), &
             !               GRADS(:,:,I+1), &
             !               INPUT_VECS(:,:),   INPUT_VECS_GRAD_MEAN(:,:),   INPUT_VECS_GRAD_CURV(:,:), &
             !               INPUT_SHIFT(:),    INPUT_SHIFT_GRAD_MEAN(:),    INPUT_SHIFT_GRAD_CURV(:),  &
             !               STATE_VECS(:,:,I), STATE_VECS_GRAD_MEAN(:,:,I), STATE_VECS_GRAD_CURV(:,:,I))
             !       ELSE
             !          CALL REPLACE_BASIS_FUNCTIONS( &
             !               STATE_USAGE(:,1), &
             !               X(:,:), &
             !               STATES(:,:,I), &
             !               Y_GRADIENT(:,:), &
             !               INPUT_VECS(:,:),  INPUT_VECS_GRAD_MEAN(:,:),  INPUT_VECS_GRAD_CURV(:,:), &
             !               INPUT_SHIFT(:),   INPUT_SHIFT_GRAD_MEAN(:),   INPUT_SHIFT_GRAD_CURV(:),  &
             !               OUTPUT_VECS(:,:), OUTPUT_VECS_GRAD_MEAN(:,:), OUTPUT_VECS_GRAD_CURV(:,:))
             !       END IF
             !    ELSE IF (I .EQ. NS) THEN
             !       CALL REPLACE_BASIS_FUNCTIONS( &
             !            STATE_USAGE(:,1), &
             !            STATES(:,:,I-1), &
             !            STATES(:,:,I), &
             !            Y_GRADIENT(:,:), &
             !            STATE_VECS(:,:,I-1), STATE_VECS_GRAD_MEAN(:,:,I-1), STATE_VECS_GRAD_CURV(:,:,I-1), &
             !            STATE_SHIFT(:,I-1),  STATE_SHIFT_GRAD_MEAN(:,I-1),  STATE_SHIFT_GRAD_CURV(:,I-1),  &
             !            OUTPUT_VECS(:,:),    OUTPUT_VECS_GRAD_MEAN(:,:),    OUTPUT_VECS_GRAD_CURV(:,:))
             !    ELSE
             !       CALL REPLACE_BASIS_FUNCTIONS( &
             !            STATE_USAGE(:,1), &
             !            STATES(:,:,I-1), &
             !            STATES(:,:,I), &
             !            GRADS(:,:,I+1), &
             !            STATE_VECS(:,:,I-1), STATE_VECS_GRAD_MEAN(:,:,I-1), STATE_VECS_GRAD_CURV(:,:,I-1), &
             !            STATE_SHIFT(:,I-1),  STATE_SHIFT_GRAD_MEAN(:,I-1),  STATE_SHIFT_GRAD_CURV(:,I-1),  &
             !            STATE_VECS(:,:,I),   STATE_VECS_GRAD_MEAN(:,:,I),   STATE_VECS_GRAD_CURV(:,:,I))
             !    END IF
             ! END IF ! END basis replacement
             ! --------------------------------------------------------------------------------
          END DO
       END DO
    END SUBROUTINE CHECK_MODEL_RANK

    ! ! Create new basis functions when the total rank of the current
    ! ! state is not full with the following priorities:
    ! !   Pick directions that align with the gradient at next state.
    ! !   Pick directions that are not already captured in this state.
    ! !   Pick directions that are different from those already captured.
    ! SUBROUTINE REPLACE_BASIS_FUNCTIONS(USAGE, &
    !      PREV_STATE, CURR_STATE, NEXT_GRADS, &
    !      IN_VECS, IN_VECS_GRAD_MEAN, IN_VECS_GRAD_CURV, &
    !      SHIFTS, SHIFTS_GRAD_MEAN, SHIFTS_GRAD_CURV, &
    !      OUT_VECS, OUT_VECS_GRAD_MEAN, OUT_VECS_GRAD_CURV)
    !   INTEGER(KIND=INT32), INTENT(IN), DIMENSION(:) :: USAGE
    !   REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: PREV_STATE
    !   REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: CURR_STATE
    !   REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: NEXT_GRADS
    !   REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: IN_VECS, IN_VECS_GRAD_MEAN, IN_VECS_GRAD_CURV
    !   REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: SHIFTS, SHIFTS_GRAD_MEAN, SHIFTS_GRAD_CURV
    !   REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: OUT_VECS, OUT_VECS_GRAD_MEAN, OUT_VECS_GRAD_CURV
    !   ! Local variables.
    !   ! REAL(KIND=RT), DIMENSION(SIZE(USAGE)) :: VALUES ! LOCAL ALLOCATION
    !   ! REAL(KIND=RT), DIMENSION(SIZE(PREV_STATE,1), SIZE(PREV_STATE,2)) :: PREV_TEMP ! LOCAL ALLOCATION
    !   ! REAL(KIND=RT), DIMENSION(SIZE(CURR_STATE,1), SIZE(CURR_STATE,2)) :: CURR_TEMP ! LOCAL ALLOCATION
    !   ! REAL(KIND=RT), DIMENSION(SIZE(NEXT_GRADS,1), SIZE(NEXT_GRADS,2)) :: GRAD_TEMP ! LOCAL ALLOCATION
    !   ! REAL(KIND=RT), DIMENSION(SIZE(IN_VECS,2), SIZE(IN_VECS,1)) :: VECS_TEMP ! LOCAL ALLOCATION
    !   ! REAL(KIND=RT), DIMENSION(SIZE(CURR_STATE,2)) :: VECS_TEMP ! LOCAL ALLOCATION
    !   ! INTEGER(KIND=INT32), DIMENSION(SIZE(USAGE)) :: ORDER ! LOCAL ALLOCATION
    !   INTEGER(KIND=INT32) :: RANK, I, GRAD_RANK, MISS_RANK
    !   ! TODO:
    !   !  - Multiply value columns by the 2-norm of all outgoing
    !   !    weights before doing the orthogonalization and ranking.
    !   ! 
    !   !  - Set new shift term such that the sum of the gradient is maximized?
    !   ! 
    !   !  - Create a function that does LEAST_SQUARES with a truncation factor
    !   !    that uses the SVD to truncate the number of vectors generated.
    !   ! 
    !   ! - When measuring alignment of two vectors come up with way to
    !   !   quickly find the "most aligned" shift term (the shift that
    !   !   maximizes the dot product of the vectors assuming truncation).
    !   ! 
    !   ! - Update REPLACE_BASIS_FUNCTIONS to:
    !   !    sum the number of times a component had no rank across threads
    !   !    (not necessary) swap weights for the no-rank components to the back
    !   !    (not necessary) swap no-rank state component values into contiguous memory at back
    !   !    linearly regress the kept-components onto the next-layer dropped difference
    !   !    compute the first no-rank principal components of the gradient, store in droped slots
    !   !    regress previous layer onto the gradient components
    !   !    fill any remaining nodes (if not enough from gradient) with "uncaptured" principal components
    !   !    set new shift terms as the best of 5 well spaced values in [-1,1], or random given no order
    !   ! 
    !   ! ! Find the first zero-valued (unused) basis function (after orthogonalization).
    !   ! FORALL (RANK = 1 :SIZE(ORDER(:))) ORDER(RANK) = RANK
    !   ! VALUES(:) = -REAL(USAGE,RT)
    !   ! CALL ARGSORT(VALUES(:), ORDER(:))
    !   ! DO RANK = 1, SIZE(ORDER(:))
    !   !    IF (USAGE(ORDER(RANK)) .EQ. 0) EXIT
    !   ! END DO
    !   ! IF (RANK .GT. SIZE(ORDER)) RETURN
    !   ! 
    !   ! Pack the ORDER(:RANK) nodes into the front of the weights:
    !   !   - update input weights, mean gradient, gradient curvature
    !   !   - update input shifts, mean gradient, gradient curvature
    !   ! Perform a least squares fit of the ORDER(:RANK) nodes to the output values.
    !   ! If the residual is low, then replace the output weights of the ORDER(:RANK)
    !   !  nodes and set the other values to zeros.
    !   ! Reset all gradients to zero and curvatures to zero for the directly affected weights.
    !   ! 
    ! END SUBROUTINE REPLACE_BASIS_FUNCTIONS

  END SUBROUTINE CONDITION_MODEL


  ! Check all of the same inputs for FIT_MODEL to make sure shapes ane sizes match.
  SUBROUTINE FIT_CHECK(CONFIG, MODEL, RWORK, IWORK, LWORK, &
       AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YI_IN, YW_IN, &
       YW, AGG_ITERATORS, INFO)
    ! TODO: Take an output file name (STDERR and STDOUT are handled).
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: RWORK
    INTEGER(KIND=INT32), INTENT(IN), DIMENSION(:) :: IWORK
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: LWORK
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AX_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES_IN
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI_IN
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: YI_IN
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: YW_IN
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: YW  ! (SIZE(YW_IN,1),NMS)
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AGG_ITERATORS ! (6,SIZE(SIZES_IN))
    INTEGER(KIND=INT32), INTENT(OUT) :: INFO
    ! Check for a valid data shape given the model.
    INFO = 0
    ! Check the shape of all inputs (to make sure they match this model).
    CALL CHECK_SHAPE(CONFIG, MODEL, AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YI_IN, INFO)
    ! Do shape checks on the work space provided.
    IF (SIZE(RWORK,KIND=INT64) .LT. CONFIG%RWORK_SIZE) THEN
       INFO = 18 ! Provided RWORK is not large enough.
    ELSE IF (SIZE(IWORK,KIND=INT64) .LT. CONFIG%IWORK_SIZE) THEN
       INFO = 19 ! Provided IWORK is not large enough.
    ELSE IF (SIZE(LWORK,KIND=INT64) .LT. CONFIG%LWORK_SIZE) THEN
       INFO = 20 ! Provided LWORK is not large enough.
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (CONFIG%NA .LT. 1)) THEN
       INFO = 21 ! Aggregate batch size is zero with nonzero expected aggregate input.
    ELSE IF ((CONFIG%MDI .GT. 0) .AND. (CONFIG%NM .LT. 1)) THEN
       INFO = 22 ! Model batch size is zero with nonzero expected model input.
    END IF
    ! Do shape checks on the YW (weights for Y's) provided.
    IF (SIZE(YW_IN,2) .NE. SIZE(Y_IN,2)) THEN
       INFO = 23 ! Bad YW number of points.
    ELSE IF ((SIZE(YW_IN,1) .NE. 0) & ! some weights provided
         .AND. (SIZE(YW_IN,1) .NE. 1) & ! not one weight per point
         .AND. (SIZE(YW_IN,1) .NE. SIZE(Y_IN,1))) THEN ! one weight per output component
       INFO = 24 ! Bad YW dimension (either 1 per point or commensurate with Y).
    ELSE IF (MINVAL(YW_IN(:,:)) .LT. 0.0_RT) THEN
       INFO = 25 ! Bad YW values (negative numbers are not allowed).
    END IF
    ! Check YW shape.
    IF (SIZE(YW,1,KIND=INT64) .NE. SIZE(YW_IN,1,KIND=INT64)) THEN
       INFO = 26 ! Bad YW first dimension, does not match YW_IN.
    ELSE IF (SIZE(YW,2,KIND=INT64) .NE. CONFIG%NMS) THEN
       INFO = 27 ! Bad YW second dimension, does not match NMS.
    END IF
    ! Check AGG_ITERATORS shape.
    IF (SIZE(AGG_ITERATORS,1) .NE. 6) THEN
       INFO = 28 ! Bad AGG_ITERATORS first dimension, should be 6.
    ELSE IF (SIZE(AGG_ITERATORS,2,KIND=INT64) .NE. (SIZE(SIZES_IN,KIND=INT64))) THEN
       INFO = 29 ! Bad AGG_ITERATORS second dimension, should match SIZES_IN.
    END IF
  END SUBROUTINE FIT_CHECK
    

  ! Fit input / output pairs by minimizing mean squared error.
  SUBROUTINE FIT_MODEL(CONFIG, MODEL, RWORK, IWORK, LWORK, &
       AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YI_IN, YW_IN, &
       YW, AGG_ITERATORS, STEPS, RECORD, SUM_SQUARED_ERROR, &
       CONTINUING, INFO)
    ! TODO: Take an output file name (STDERR and STDOUT are handled).
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: RWORK
    INTEGER(KIND=INT32), INTENT(INOUT), DIMENSION(:) :: IWORK
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:) :: LWORK
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: AXI_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:) :: SIZES_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: XI_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_IN
    INTEGER(KIND=INT64), INTENT(IN), DIMENSION(:,:) :: YI_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW  ! (SIZE(YW_IN,1),NMS)
    INTEGER(KIND=INT64), INTENT(INOUT), DIMENSION(:,:) :: AGG_ITERATORS ! (6,SIZE(SIZES_IN))
    INTEGER(KIND=INT32), INTENT(IN) :: STEPS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(6,STEPS), OPTIONAL :: RECORD
    LOGICAL(KIND=C_BOOL), INTENT(IN), OPTIONAL :: CONTINUING
    REAL(KIND=RT), INTENT(OUT) :: SUM_SQUARED_ERROR
    INTEGER(KIND=INT32), INTENT(OUT) :: INFO
    ! Local variables.
    LOGICAL(KIND=C_BOOL) :: CONTINUING_FIT  ! Local storage for whether this is initial or continuing.
    INTEGER(KIND=INT64) :: C ! Often used as "chord" for DO loops.
    INTEGER(KIND=INT64) :: D ! Often used as "dimension" for DO loops.
    INTEGER(KIND=INT64) :: I ! Often used as "index" for DO loops.
    INTEGER(KIND=INT64) :: BE ! Used as "batch end" index.
    INTEGER(KIND=INT64) :: BS ! Used as "batch start" index.
    INTEGER(KIND=INT64) :: BT ! Used as "batch total", the number of elements in a batch.
    INTEGER(KIND=INT64) :: NA ! Used as "number aggregate" that is the number of aggregates in a batch.
    INTEGER(KIND=INT64) :: NM ! Used as "number model" that is the number of fixeds in a batch.
    INTEGER(KIND=INT64) :: SE ! Used as "sizes end" index of last element in SIZES.
    INTEGER(KIND=INT64) :: SS ! Used as "sizes start" index of first element in SIZES.
    INTEGER(KIND=INT64) :: TN ! Used as "thread number" for parallelism.
    INTEGER(KIND=INT64) :: TT ! Used as "total threads" for parallelism.
    INTEGER(KIND=INT64) :: BEA ! Used as "batch end aggregate" the first aggregate index of a batch.
    INTEGER(KIND=INT64) :: BSA ! Used as "batch start aggregate" the first aggregate index of a batch.
    INTEGER(KIND=INT64) :: BATCH ! The current batch number.
    INTEGER(KIND=INT64) :: CURRENT_TIME ! The current time from the system clock.
    ! Batching.
    INTEGER(KIND=INT64), DIMENSION(:), ALLOCATABLE :: &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS
    ! Timing.
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    ! 
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Set whether or not we are resuming a previous call.
    CONTINUING_FIT = .FALSE.
    IF (PRESENT(CONTINUING)) THEN
       CONTINUING_FIT = CONTINUING
    END IF
    ! 
    ! TODO: For deciding which points to keep when doing batching:
    !        track the error VALUE for each point
    !        track the trailing average error CHANGE for each point (^.5)
    !        track the trailing average error CHANGE for all points (E^.5)
    !        keep the largest X% of (point's error) * (-error change)
    !        introduce new points with the trailing average error change
    !       
    ! Unpack all of the work storage into the expected shapes.
    CALL UNPACKED_FIT_MODEL(&
         ! Data batch holders.
         LWORK(CONFIG%SAXI : CONFIG%EAXI), & ! AXI
         RWORK(CONFIG%SAXB : CONFIG%EAXB), & ! AX
         LWORK(CONFIG%SSB : CONFIG%ESB), & ! SIZES
         LWORK(CONFIG%SMXI : CONFIG%EMXI), & ! XI
         RWORK(CONFIG%SMXB : CONFIG%EMXB), & ! X
         RWORK(CONFIG%SMYB : CONFIG%EMYB), & ! Y
         LWORK(CONFIG%SOXI : CONFIG%EOXI), & ! YI
         ! Model components.
         MODEL(CONFIG%ASIV : CONFIG%AEIV), & ! AGGREGATOR_INPUT_VECS
         MODEL(CONFIG%ASOV : CONFIG%AEOV), & ! AGGREGATOR_OUTPUT_VECS
         MODEL(CONFIG%MSIV : CONFIG%MEIV), & ! MODEL_INPUT_VECS
         MODEL(CONFIG%MSOV : CONFIG%MEOV), & ! MODEL_OUTPUT_VECS
         MODEL(CONFIG%ASEV : CONFIG%AEEV), & ! AGGREGATOR_EMBEDDING_VECS
         MODEL(CONFIG%MSEV : CONFIG%MEEV), & ! MODEL_EMBEDDING_VECS
         MODEL(CONFIG%OSEV : CONFIG%OEEV), & ! OUTPUT_EMBEDDING_VECS
         ! States and gradients for model optimization.
         RWORK(CONFIG%SMG : CONFIG%EMG), & ! MODEL_GRAD(NUM_VARS,NUM_THREADS)
         RWORK(CONFIG%SMGM : CONFIG%EMGM), & ! MODEL_GRAD_MEAN(NUM_VARS)
         RWORK(CONFIG%SMGC : CONFIG%EMGC), & ! MODEL_GRAD_CURV(NUM_VARS)
         RWORK(CONFIG%SBM : CONFIG%EBM), & ! BEST_MODEL(TOTAL_SIZE)
         ! States and gradients in parellel regions (work space).
         RWORK(CONFIG%SYG : CONFIG%EYG), & ! Y_GRADIENT(MDO,NMS)
         RWORK(CONFIG%SMXG : CONFIG%EMXG), & ! M_GRADS(NMS,MDS,MNS+1)
         RWORK(CONFIG%SMXS : CONFIG%EMXS), & ! M_STATES(NMS,MDS,MNS+1)
         RWORK(CONFIG%SXG : CONFIG%EXG), & ! X_GRADIENT(MDI,NMS)
         RWORK(CONFIG%SAYG : CONFIG%EAYG), & ! AY_GRADIENT(NA,ADO+1)
         RWORK(CONFIG%SAY : CONFIG%EAY), & ! AY(NA,ADO+1)
         RWORK(CONFIG%SAXG : CONFIG%EAXG), & ! A_GRADS(NA,ADS,ANS+1)
         RWORK(CONFIG%SAXS : CONFIG%EAXS), & ! A_STATES(NA,ADS,ANS+1)
         RWORK(CONFIG%SAG : CONFIG%EAG), & ! AX_GRADIENT(ADI,NA)
         ! Data scaling and normalization.
         MODEL(CONFIG%AISS:CONFIG%AISE), & ! AX_SHIFT(ADN)
         MODEL(CONFIG%AIMS:CONFIG%AIME), & ! AX_RESCALE(ADN,ADN)
         RWORK(CONFIG%SAXIS : CONFIG%EAXIS), & ! AXI_SHIFT(ADE)
         RWORK(CONFIG%SAXIR : CONFIG%EAXIR), & ! AXI_RESCALE(ADE,ADE)
         MODEL(CONFIG%AOSS : CONFIG%AOSE), & ! AY_SHIFT(ADO)
         MODEL(CONFIG%AOMS : CONFIG%AOME), & ! AY_SCALE(ADO)
         MODEL(CONFIG%MISS:CONFIG%MISE), & ! X_SHIFT(MDN)
         MODEL(CONFIG%MIMS:CONFIG%MIME), & ! X_RESCALE(MDN,MDN)
         RWORK(CONFIG%SMXIS : CONFIG%EMXIS), & ! XI_SHIFT(MDE)
         RWORK(CONFIG%SMXIR : CONFIG%EMXIR), & ! XI_RESCALE(MDE,MDE)
         MODEL(CONFIG%MOSS:CONFIG%MOSE), & ! Y_SHIFT(DO-DOE)
         MODEL(CONFIG%MOMS:CONFIG%MOME), & ! Y_RESCALE(DO-DOE,DO-DOE)
         RWORK(CONFIG%SOXIS : CONFIG%EOXIS), & ! YI_SHIFT(DOE)
         RWORK(CONFIG%SOXIR : CONFIG%EOXIR), & ! YI_RESCALE(DOE,DOE)
         ! Work space for orthogonalization (conditioning) or gradient calculation.
         RWORK(CONFIG%SAL : CONFIG%EAL), & ! A_LENGTHS
         RWORK(CONFIG%SML : CONFIG%EML), & ! M_LENGTHS
         RWORK(CONFIG%SAST : CONFIG%EAST), & ! A_STATE_TEMP
         RWORK(CONFIG%SMST : CONFIG%EMST), & ! M_STATE_TEMP
         RWORK(CONFIG%SAET : CONFIG%EAET), & ! A_EMB_TEMP
         RWORK(CONFIG%SMET : CONFIG%EMET), & ! M_EMB_TEMP
         RWORK(CONFIG%SEOS : CONFIG%EEOS), & ! EMB_OUTS
         RWORK(CONFIG%SEOG : CONFIG%EEOG), & ! EMB_GRADS
         ! Rank evaluation (when conditioning model).
         IWORK(CONFIG%SAO : CONFIG%EAO), & ! A_ORDER
         IWORK(CONFIG%SMO : CONFIG%EMO), & ! M_ORDER
         ! Update indicies.
         LWORK(CONFIG%SUI : CONFIG%EUI) & ! UPDATE_INDICES
         )
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    !$OMP CRITICAL
    CONFIG%WFIT = CONFIG%WFIT + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CFIT = CONFIG%CFIT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
    !$OMP END CRITICAL

  CONTAINS

    ! Unpack the work arrays into the proper shapes.
    SUBROUTINE UNPACKED_FIT_MODEL(&
         AXI, AX, SIZES, XI, X, Y, YI, &
         A_IN_VECS, A_OUT_VECS, M_IN_VECS, M_OUT_VECS, &
         A_EMB_VECS, M_EMB_VECS, O_EMB_VECS, &
         MODEL_GRAD, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, BEST_MODEL, &
         Y_GRADIENT, M_GRADS, M_STATES, X_GRADIENT, AY_GRADIENT, AY, A_GRADS, A_STATES, AX_GRADIENT, &
         AX_SHIFT, AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_SHIFT, AY_SCALE, &
         X_SHIFT, X_RESCALE, XI_SHIFT, XI_RESCALE, &
         Y_SHIFT, Y_RESCALE, YI_SHIFT, YI_RESCALE, &
         A_LENGTHS, M_LENGTHS, A_STATE_TEMP, M_STATE_TEMP, &
         A_EMB_TEMP, M_EMB_TEMP, EMB_OUTS, &
         EMB_GRADS, A_ORDER, M_ORDER, UPDATE_INDICES)
      ! Definition of unpacked work storage.
      INTEGER(KIND=INT64), DIMENSION(SIZE(AXI_IN,1), CONFIG%NA) :: AXI
      REAL(KIND=RT), DIMENSION(CONFIG%ADI, CONFIG%NA) :: AX
      INTEGER(KIND=INT64), DIMENSION(CONFIG%ESB-CONFIG%SSB+ONE) :: SIZES
      INTEGER(KIND=INT64), DIMENSION(SIZE(XI_IN,1), CONFIG%NMS) :: XI
      REAL(KIND=RT), DIMENSION(CONFIG%MDI, CONFIG%NMS) :: X
      REAL(KIND=RT), DIMENSION(CONFIG%DO, CONFIG%NMS) :: Y
      INTEGER(KIND=INT64), DIMENSION(SIZE(YI_IN,1), CONFIG%NMS) :: YI
      REAL(KIND=RT), DIMENSION(CONFIG%ADI, CONFIG%ADS, CONFIG%ANC) :: A_IN_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%ADSO, CONFIG%ADO, CONFIG%ANC) :: A_OUT_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDI, CONFIG%MDS, CONFIG%MNC) :: M_IN_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDSO, CONFIG%MDO, CONFIG%MNC) :: M_OUT_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%ADE, CONFIG%ANE) :: A_EMB_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDE, CONFIG%MNE) :: M_EMB_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%DOE, CONFIG%NOE) :: O_EMB_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%NUM_VARS,CONFIG%NUM_THREADS) :: MODEL_GRAD
      REAL(KIND=RT), DIMENSION(CONFIG%NUM_VARS) :: MODEL_GRAD_MEAN, MODEL_GRAD_CURV
      REAL(KIND=RT), DIMENSION(CONFIG%TOTAL_SIZE) :: BEST_MODEL
      REAL(KIND=RT), DIMENSION(CONFIG%DO, CONFIG%NMS) :: Y_GRADIENT
      REAL(KIND=RT), DIMENSION(CONFIG%NMS, CONFIG%MDS, CONFIG%MNS+1, CONFIG%MNC) :: M_GRADS, M_STATES
      REAL(KIND=RT), DIMENSION(CONFIG%MDI, CONFIG%NMS) :: X_GRADIENT
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADO+ONE) :: AY_GRADIENT, AY
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADS, CONFIG%ANS+1, CONFIG%ANC) :: A_GRADS, A_STATES
      REAL(KIND=RT), DIMENSION(CONFIG%ADI, CONFIG%NA) :: AX_GRADIENT
      REAL(KIND=RT), DIMENSION(CONFIG%ADN) :: AX_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%ADN, CONFIG%ADN) :: AX_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADE) :: AXI_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%ADE, CONFIG%ADE) :: AXI_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADO) :: AY_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%ADO) :: AY_SCALE
      REAL(KIND=RT), DIMENSION(CONFIG%MDN) :: X_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%MDN, CONFIG%MDN) :: X_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%MDE) :: XI_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%MDE, CONFIG%MDE) :: XI_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%DON) :: Y_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%DON, CONFIG%DON) :: Y_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%DOE) :: YI_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%DOE, CONFIG%DOE) :: YI_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADS, CONFIG%NUM_THREADS) :: A_LENGTHS
      REAL(KIND=RT), DIMENSION(CONFIG%MDS, CONFIG%NUM_THREADS) :: M_LENGTHS
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADS) :: A_STATE_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%NMS, CONFIG%MDS) :: M_STATE_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%ADE, CONFIG%ANE, CONFIG%NUM_THREADS) :: A_EMB_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%MDE, CONFIG%MNE, CONFIG%NUM_THREADS) :: M_EMB_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%NOE, CONFIG%NMS) :: EMB_OUTS
      REAL(KIND=RT), DIMENSION(CONFIG%NOE, CONFIG%NMS) :: EMB_GRADS
      INTEGER(KIND=INT32), DIMENSION(CONFIG%ADS, CONFIG%NUM_THREADS) :: A_ORDER
      INTEGER(KIND=INT32), DIMENSION(CONFIG%MDS, CONFIG%NUM_THREADS) :: M_ORDER
      INTEGER(KIND=INT64), DIMENSION(CONFIG%NUM_VARS) :: UPDATE_INDICES
      ! Timing.
      REAL :: CPU_TIME_START, CPU_TIME_END
      INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
      ! ----------------------------------------------------------------
      !                 Initialization and preparation
      ! 
      ! Store the start time of this routine (to make sure updates can
      !  be shown to the user at a reasonable frequency).
      CALL SYSTEM_CLOCK(CONFIG%FIT_LAST_INTERRUPT_TIME, CLOCK_RATE, CLOCK_MAX)
      IF (.NOT. CONTINUING_FIT) THEN
         ! Establis the amount of time to wait between interrupts.
         CONFIG%FIT_WAIT_TIME = CLOCK_RATE * CONFIG%INTERRUPT_DELAY_SEC
         ! Initialize the info / error code to 0.
         INFO = 0
         ! Cap the "number [of variables] to update" at the model size.
         CONFIG%NUM_TO_UPDATE = MAX(ONE, MIN(CONFIG%NUM_TO_UPDATE, CONFIG%NUM_VARS))
         ! Set the "total rank", the number of internal state components.
         CONFIG%FIT_TOTAL_RANK = CONFIG%MDS*CONFIG%MNS + CONFIG%ADS*CONFIG%ANS
         ! Compute the minimum number of model variables to update.
         CONFIG%FIT_MIN_TO_UPDATE = MAX(1,INT(CONFIG%MIN_UPDATE_RATIO * REAL(CONFIG%NUM_VARS,RT)))
         ! Set the initial "number of steps taken since best" counter.
         CONFIG%FIT_NS = 0
         ! Set the "num threads" to be the maximum achievable data parallelism.
         CONFIG%FIT_NT = MIN(SIZE(Y,2,KIND=INT64), CONFIG%NUM_THREADS)
         ! Initial rates of change of mean and variance values.
         CONFIG%FIT_STEP_MEAN_REMAIN = 1.0_RT - CONFIG%STEP_MEAN_CHANGE
         CONFIG%FIT_STEP_CURV_REMAIN = 1.0_RT - CONFIG%STEP_CURV_CHANGE
         ! Initial mean squared error is "max representable value".
         CONFIG%FIT_PREV_MSE = HUGE(CONFIG%FIT_PREV_MSE)
         CONFIG%FIT_BEST_MSE = HUGE(CONFIG%FIT_BEST_MSE)
         ! Set the initial curvature values for the model gradient.
         MODEL_GRAD_CURV(:) = CONFIG%INITIAL_CURV_ESTIMATE
         ! Disable the application of SHIFT (since data is / will be normalized).
         CONFIG%FIT_NORMALIZE = CONFIG%NORMALIZE
         CONFIG%NORMALIZE = .FALSE.
         ! Initialize the aggregate iterators.
         ! 
         ! TODO: Move this code into data fetching, shouldn't happen here.
         !       First fetch of fixed input initializes aggregator, repeated fetch leaves it.
         ! 
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS) &
         !$OMP& IF((CONFIG%NUM_THREADS > 0) .AND. (SIZE(SIZES_IN) > 0))
         DO I = 1, SIZE(SIZES_IN)
            IF (SIZES_IN(I) .EQ. 0) THEN
               AGG_ITERATORS(:,I) = ZERO
            ELSE
               AGG_ITERATORS(1,I) = INT(SIZES_IN(I), INT64)
               IF (CONFIG%PAIRWISE_AGGREGATION) THEN
                  AGG_ITERATORS(1,I) = AGG_ITERATORS(1,I)**TWO
               END IF
               CALL INITIALIZE_ITERATOR( &
                    I_LIMIT=AGG_ITERATORS(1,I), &
                    I_NEXT=AGG_ITERATORS(2,I), &
                    I_MULT=AGG_ITERATORS(3,I), &
                    I_STEP=AGG_ITERATORS(4,I), &
                    I_MOD=AGG_ITERATORS(5,I), &
                    I_ITER=AGG_ITERATORS(6,I) &
               )
            END IF
         END DO
         ! Make all iterators deterministic when all pairs will fit into the model.
         NA = SUM(AGG_ITERATORS(1,:))
         IF (NA .LE. CONFIG%NA) THEN
            AGG_ITERATORS(2,:) = ZERO
            AGG_ITERATORS(3,:) = ONE
            AGG_ITERATORS(4,:) = ONE
            AGG_ITERATORS(5,:) = AGG_ITERATORS(1,:)
            AGG_ITERATORS(6,:) = ZERO
         END IF
         ! 
         ! TODO: Set up validation data (separate from fit data).
         !       Add swap scratch space to the memory somewhere.
         !       Put the validation data at the end of the array of data.
         !       Make sure the "real" data is at the front.
         !       Whichever set of data is smaller should be generated by well-spacedness.
         ! 
         ! TODO: Apply normalizations separately to the validation data.
         ! 
         ! TODO: Parameter updating policies should be determined by fit MSE change.
         !       Model saving and early stopping should be determined by validation MSE.
         ! 
         ! 
         ! Normalize the *_IN data before fitting the model.
         CALL NORMALIZE_DATA(CONFIG, MODEL, AGG_ITERATORS, &
              AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YI_IN, YW_IN, &
              AX, AXI, SIZES, X, XI, Y, YI, YW, &
              AX_SHIFT, AX_RESCALE, &
              AXI_SHIFT, AXI_RESCALE, &
              AY_SHIFT, &
              AY_SCALE, &
              X_SHIFT, X_RESCALE, &
              XI_SHIFT, XI_RESCALE, &
              Y_SHIFT, Y_RESCALE, &
              YI_SHIFT, YI_RESCALE, &
              A_EMB_VECS, M_EMB_VECS, O_EMB_VECS, &
              A_STATES, AY, INFO)
         IF (INFO .NE. 0) RETURN
         ! Set the initial value of STEP.
         CONFIG%FIT_STEP = 1
         ! Write the status update to the command line.
         CALL SYSTEM_CLOCK(CURRENT_TIME, CLOCK_RATE, CLOCK_MAX)
         IF (CURRENT_TIME - CONFIG%FIT_LAST_INTERRUPT_TIME &
              .GT. CONFIG%FIT_WAIT_TIME) THEN
            CONFIG%FIT_LAST_INTERRUPT_TIME = CURRENT_TIME
            RETURN
         END IF
      END IF
      ! 
      ! TODO: Compute batches once, reuse for all of fit.
      ! 
      ! ----------------------------------------------------------------
      !                    Minimizing mean squared error
      ! 
      ! Iterate, taking steps with the average gradient over all data.
      fit_loop : DO WHILE (CONFIG%FIT_STEP .LE. STEPS)
         ! TODO: Consider wrapping the embed, evaluate, model gradient code in
         !       a higher level thread block to include parallelization over
         !       larger scopes. Will have to be done after the batch is constructed.
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !                 Pack data into the input work space. 
         ! Pack the data into the work space for a single batch forward pass operation.
         ! If pairwise aggregation is enabled, this also computes the appropriate differences.
         ! 
         ! TODO: This function should *NOT* modify the data if none of it needs to be updated.
         !       That will allow for normalization to be skipped.
         CALL FETCH_DATA(CONFIG, AGG_ITERATORS, &
              AX_IN, AX_GRADIENT, AXI_IN, AXI, SIZES_IN, SIZES, &
              X_IN, X_GRADIENT, XI_IN, XI, Y_IN, Y, YI_IN, YI, YW_IN, YW, NA, NM )
         ! Normalize the batch of data and update the normalization factors based on
         !  the batch size and the current step. Otherwise if this is not a new batch
         !  then does nothing.
         CALL NORMALIZE_STEP(CONFIG, MODEL, RWORK, AX_GRADIENT(:,:NA), X_GRADIENT(:,:NM), Y(:,:NM), YW(:,:NM))
         ! 
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !               Use broad parallelism to distribute fit work.
         ! 
         ! Compute the batch start and end indices.
         CALL COMPUTE_BATCHES(CONFIG, NA, NM, SIZES, &
              BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, FIX_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
              JOINT=.TRUE._C_BOOL, INFO=INFO)
         IF (INFO .NE. 0) RETURN
         ! Temporarily set the number of threads to 1, since we are already in a parallel region,
         !  the contained methods do not need to utilize another layer of parallelism.
         TT = CONFIG%NUM_THREADS
         CONFIG%NUM_THREADS = 1
         ! Holds the sum of squared error.
         SUM_SQUARED_ERROR = 0.0_RT
         ! Compute all indices (for parallelism) related to this batch.
         !$OMP PARALLEL DO NUM_THREADS(CONFIG%FIT_NT) PRIVATE(BATCH, BS, BE, BT, SS, SE, BSA, BEA, TN) &
         !$OMP& REDUCTION(+:SUM_SQUARED_ERROR) IF(CONFIG%FIT_NT > 1)
         DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
            IF (INFO .NE. 0) CYCLE
            BS = BATCHM_STARTS(BATCH)
            BE = BATCHM_ENDS(BATCH)
            BT = BE-BS+1
            IF (BT .LE. 0) CYCLE
            IF (SIZE(SIZES,KIND=INT64) .GT. ZERO) THEN
               ! Set the start and end indices for the SIZES array.
               IF (CONFIG%PARTIAL_AGGREGATION) THEN
                  SS = 1
                  DO WHILE ((SS .LE. SIZE(SIZES,KIND=INT64)) .AND. (SUM(SIZES(1:SS)) .LT. BS))
                     SS = SS + 1
                  END DO
                  SE = SS
                  DO WHILE ((SE .LT. SIZE(SIZES,KIND=INT64)) .AND. (SUM(SIZES(SS:SE)) .LT. (BE-BS)))
                     SE = SE + 1
                  END DO
                  IF (SS .GT. SIZE(SIZES,KIND=INT64)) THEN
                     SS = 1
                     SE = 0
                  END IF
               ELSE
                  SS = BS
                  SE = BE
               END IF
               ! Get the aggregate start and end indices.
               BSA = BATCHA_STARTS(BATCH)
               BEA = BATCHA_ENDS(BATCH)
            ELSE
               SS = 1
               SE = 0
               BSA = 1
               BEA = 0
            END IF
            TN = OMP_GET_THREAD_NUM() + 1
            ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ! 
            !             Evaluate the model at all points, storing states.
            ! Embed all integer inputs into real vector inputs.
            CALL EMBED(CONFIG, MODEL, AXI(:,BSA:BEA), XI(:,BS:BE), AX_GRADIENT(:,BSA:BEA), X_GRADIENT(:,BS:BE))
            ! 
            ! Evaluate the model, storing internal states (for gradient calculation).
            ! If we are checking rank, we need to store evaluations and gradients separately.
            CALL EVALUATE(CONFIG, MODEL, AX_GRADIENT(:,BSA:BEA), AY_GRADIENT(BSA:BEA,:), SIZES(SS:SE), &
                 X_GRADIENT(:,BS:BE), Y_GRADIENT(:,BS:BE), A_GRADS(BSA:BEA,:,:,:), M_GRADS(BS:BE,:,:,:), INFO)

            ! Copy the state values into holders if rank checking or condintioning will be done.
            IF (CONFIG%RANK_CHECK_FREQUENCY .GT. 0) THEN
               IF (MOD(CONFIG%FIT_STEP-1,CONFIG%RANK_CHECK_FREQUENCY) .EQ. 0) THEN
                  AX(:,BSA:BEA) = AX_GRADIENT(:,BSA:BEA)
                  A_STATES(BSA:BEA,:,:,:) = A_GRADS(BSA:BEA,:,:,:)
                  AY(BSA:BEA,:) = AY_GRADIENT(BSA:BEA,:)
                  X(:,BS:BE) = X_GRADIENT(:,BS:BE)
                  M_STATES(BS:BE,:,:,:) = M_GRADS(BS:BE,:,:,:)
               ELSE IF (CONFIG%MODEL_CONDITION_FREQUENCY .GT. 0) THEN
                  IF (MOD(CONFIG%STEPS_TAKEN-1,CONFIG%MODEL_CONDITION_FREQUENCY) .EQ. 0) THEN
                     AX(:,BSA:BEA) = AX_GRADIENT(:,BSA:BEA)
                     AY(BSA:BEA,:) = AY_GRADIENT(BSA:BEA,:)
                  END IF
               END IF
            ELSE IF (CONFIG%MODEL_CONDITION_FREQUENCY .GT. 0) THEN
               IF (MOD(CONFIG%STEPS_TAKEN-1,CONFIG%MODEL_CONDITION_FREQUENCY) .EQ. 0) THEN
                  AX(:,BSA:BEA) = AX_GRADIENT(:,BSA:BEA)
                  AY(BSA:BEA,:) = AY_GRADIENT(BSA:BEA,:)
                  X(:,BS:BE) = X_GRADIENT(:,BS:BE)
               END IF
            END IF
            IF (INFO .NE. 0) CYCLE
            ! 
            ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            !                       Compute model gradient 
            ! 
            ! Sum the gradient over all data. If a rank check will be
            !  performed then store the states separate from the gradients.
            !  Otherwise, only compute the gradients and reuse that memory space.
            CALL MODEL_GRADIENT(CONFIG, MODEL(:), &
                 AX_GRADIENT(:,BSA:BEA), AXI(:,BSA:BEA), SIZES(SS:SE), X_GRADIENT(:,BS:BE), XI(:,BS:BE), &
                 Y(:,BS:BE), YI(:,BS:BE), YW(:,BS:BE), &
                 SUM_SQUARED_ERROR, MODEL_GRAD(:,TN:TN), INFO, AY_GRADIENT(BSA:BEA,:),  &
                 Y_GRADIENT(:,BS:BE), A_GRADS(BSA:BEA,:,:,:), M_GRADS(BS:BE,:,:,:), &
                 A_EMB_TEMP(:,:,TN:TN), M_EMB_TEMP(:,:,TN:TN), &
                 EMB_OUTS(:,BS:BE), EMB_GRADS(:,BS:BE))
            IF (INFO .NE. 0) CYCLE
         END DO
         CONFIG%NUM_THREADS = TT
         IF (INFO .NE. 0) RETURN
         ! 
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !           Update the step factors, early stop if appropaite.
         ! 
         CALL ADJUST_RATES(BEST_MODEL, MODEL_GRAD_MEAN(:), MODEL_GRAD_CURV(:))
         IF (INFO .NE. 0) RETURN
         IF (CONFIG%FIT_NS .EQ. HUGE(CONFIG%FIT_NS)) EXIT fit_loop
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !              Modify the model variables (take step).
         ! 
         CALL STEP_VARIABLES(MODEL_GRAD(:,1:CONFIG%FIT_NT), MODEL_GRAD_MEAN(:), &
              MODEL_GRAD_CURV(:), UPDATE_INDICES(:), CONFIG%FIT_NT)
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !  Project the model parameters back into a safely constrained region.
         ! 
         ! Rescale internal vectors to have a maximum 2-norm of 1.
         ! Center the outputs of the aggregator model about the origin.
         ! Measure the "total rank" of all internal state representations of data.
         IF (CONFIG%MODEL_CONDITION_FREQUENCY .GT. 0) THEN
            IF (MOD(CONFIG%STEPS_TAKEN-1,CONFIG%MODEL_CONDITION_FREQUENCY) .EQ. 0) THEN
               CALL CONDITION_MODEL(CONFIG, &
                    MODEL(:), MODEL_GRAD_MEAN(:), MODEL_GRAD_CURV(:), & ! Model and gradient.
                    AX(:,:NA), AXI(:,:NA), AY(:NA,:), AY_GRADIENT(:NA,:), SIZES(:), & ! Data.
                    X(:,:), XI(:,:), Y_GRADIENT(:,:), &
                    CONFIG%NUM_THREADS, CONFIG%STEPS_TAKEN, & ! Configuration for conditioning.
                    A_STATES(:NA,:,:,:), M_STATES(:,:,:,:), & ! State values at basis functions.
                    A_GRADS(:NA,:,:,:), M_GRADS(:,:,:,:), & ! Gradient values at basis functions.
                    A_LENGTHS(:,:), M_LENGTHS(:,:), & ! Work space for orthogonalization.
                    A_STATE_TEMP(:,:), M_STATE_TEMP(:,:), & ! Work space for state values.
                    A_ORDER(:,:), M_ORDER(:,:), &
                    CONFIG%FIT_TOTAL_EVAL_RANK, CONFIG%FIT_TOTAL_GRAD_RANK)
            END IF
         END IF
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         ! Record statistics about the model and write an update about 
         ! step and convergence to the command line.
         CALL RECORD_STATS(MODEL_GRAD)
         ! Update the step.
         CONFIG%FIT_STEP = CONFIG%FIT_STEP + 1
         ! Write the status update to the command line.
         CALL SYSTEM_CLOCK(CURRENT_TIME, CLOCK_RATE, CLOCK_MAX)
         IF (CURRENT_TIME - CONFIG%FIT_LAST_INTERRUPT_TIME .GT. CONFIG%FIT_WAIT_TIME) THEN
            CONFIG%FIT_LAST_INTERRUPT_TIME = CURRENT_TIME
            EXIT fit_loop
         END IF
      END DO fit_loop
      ! Only preform the encoding if the fit is complete.
      IF (CONFIG%FIT_STEP .GT. STEPS) THEN
         CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
         CALL CPU_TIME(CPU_TIME_START)
         ! 
         ! ----------------------------------------------------------------
         !                 Finalization, prepare for return.
         ! 
         ! Restore the best model seen so far (if enough steps were taken).
         IF (CONFIG%KEEP_BEST .AND. (STEPS .GT. 0)) THEN
            CONFIG%FIT_MSE = CONFIG%FIT_BEST_MSE
            MODEL(:) = BEST_MODEL(:)
         END IF
         ! 
         ! Apply the data normalizing scaling factors to the weight
         !  matrices to embed normalization into the model.
         IF (CONFIG%ENCODE_SCALING) THEN
            IF (CONFIG%ADN .GT. 0) THEN
               DO C = 1, CONFIG%ANC
                  IF (CONFIG%ANS .GT. 0) THEN
                     A_IN_VECS(:CONFIG%ADN,:,C) = MATMUL(AX_RESCALE(:,:), A_IN_VECS(:CONFIG%ADN,:,C))
                  ELSE
                     A_OUT_VECS(:CONFIG%ADN,:,C) = MATMUL(AX_RESCALE(:,:), A_OUT_VECS(:CONFIG%ADN,:,C))
                  END IF
               END DO
               AX_RESCALE(:,:) = 0.0_RT
               DO D = 1, SIZE(AX_RESCALE,1)
                  AX_RESCALE(D,D) = 1.0_RT
               END DO
            END IF
            IF (CONFIG%MDN .GT. 0) THEN
               DO C = 1, CONFIG%MNC
                  IF (CONFIG%MNS .GT. 0) THEN
                     M_IN_VECS(:CONFIG%MDN,:,C) = MATMUL(X_RESCALE(:,:), M_IN_VECS(:CONFIG%MDN,:,C))
                  ELSE
                     M_OUT_VECS(:CONFIG%MDN,:,C) = MATMUL(X_RESCALE(:,:), M_OUT_VECS(:CONFIG%MDN,:,C))
                  END IF
               END DO
               X_RESCALE(:,:) = 0.0_RT
               DO D = 1, SIZE(X_RESCALE,1)
                  X_RESCALE(D,D) = 1.0_RT
               END DO
            END IF
            ! Apply the output rescale to whichever part of the model produces output.
            IF (CONFIG%MDO .GT. 0) THEN
               DO C = 1, CONFIG%MNC
                  M_OUT_VECS(:,:,C) = MATMUL(M_OUT_VECS(:,:,C), Y_RESCALE(:,:))
               END DO
            ELSE
               DO C = 1, CONFIG%ANC
                  A_OUT_VECS(:,:,C) = MATMUL(A_OUT_VECS(:,:,C), Y_RESCALE(:,:))
               END DO
            END IF
            Y_RESCALE(:,:) = 0.0_RT
            DO D = 1, SIZE(Y_RESCALE,1)
               Y_RESCALE(D,D) = 1.0_RT
            END DO
            ! Store the fact that scaling has already been encoded into the model.
            CONFIG%NEEDS_SCALING = .FALSE.
         END IF
         ! 
         ! Reset configuration settings that were modified.
         CONFIG%NORMALIZE = CONFIG%FIT_NORMALIZE
         CALL CPU_TIME(CPU_TIME_END)
         CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
         !$OMP CRITICAL
         CONFIG%WENC = CONFIG%WENC + (WALL_TIME_END - WALL_TIME_START)
         CONFIG%CENC = CONFIG%CENC + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
         !$OMP END CRITICAL
      END IF
    END SUBROUTINE UNPACKED_FIT_MODEL

    
    ! Adjust the rates of the model optimization parameters.
    SUBROUTINE ADJUST_RATES(BEST_MODEL, MODEL_GRAD_MEAN, MODEL_GRAD_CURV)
      REAL(KIND=RT), DIMENSION(:) :: BEST_MODEL
      REAL(KIND=RT), DIMENSION(:) :: MODEL_GRAD_MEAN
      REAL(KIND=RT), DIMENSION(:) :: MODEL_GRAD_CURV
      REAL :: CPU_TIME_START, CPU_TIME_END
      INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
      CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
      CALL CPU_TIME(CPU_TIME_START)
      ! Convert the sum of squared errors into the mean squared error.
      CONFIG%FIT_MSE = SUM_SQUARED_ERROR / REAL(NM * CONFIG%DO, RT) ! RNY * SIZE(Y,1)
      IF (IS_NAN(CONFIG%FIT_MSE) .OR. (.NOT. IS_FINITE(CONFIG%FIT_MSE))) THEN
         INFO = 30 ! Encountered NaN or Inf mean squared error during fitting, this should not happen. Are any values extremely large?
         RETURN
      END IF
      ! Adjust exponential sliding windows based on change in error.
      IF (CONFIG%FIT_MSE .LE. CONFIG%FIT_PREV_MSE) THEN
         CONFIG%STEP_FACTOR = MIN(CONFIG%STEP_FACTOR * CONFIG%FASTER_RATE, CONFIG%MAX_STEP_FACTOR)
         CONFIG%NUM_TO_UPDATE = CONFIG%NUM_TO_UPDATE + &
              INT(CONFIG%UPDATE_RATIO_STEP * REAL(CONFIG%NUM_VARS,RT))
         ! TODO: Should the mean and curvature adjustment rates be updated too?
      ! If the MSE has gotten too large, then do a reset of the model fit process from the previous best.
      ELSE IF (CONFIG%FIT_MSE .GT. CONFIG%MSE_UPPER_LIMIT) THEN
         CONFIG%STEP_FACTOR = CONFIG%MIN_STEP_FACTOR
         CONFIG%NUM_TO_UPDATE = CONFIG%NUM_VARS
         MODEL(:) = BEST_MODEL(:)
         ! TODO: Should the mean be randomly modified a small amount for coverage?
         MODEL_GRAD_MEAN(:) = MODEL_GRAD_MEAN(:) * CONFIG%MIN_STEP_FACTOR
         MODEL_GRAD_CURV(:) = 1.0_RT
      ELSE
         CONFIG%STEP_FACTOR = CONFIG%STEP_FACTOR * CONFIG%SLOWER_RATE
         CONFIG%STEP_FACTOR = MAX(CONFIG%STEP_FACTOR, CONFIG%MIN_STEP_FACTOR)
         CONFIG%NUM_TO_UPDATE = CONFIG%NUM_TO_UPDATE - &
              INT(CONFIG%UPDATE_RATIO_STEP * REAL(CONFIG%NUM_VARS,RT))
         ! TODO: Should the mean and curvature adjustment rates be updated too?
      END IF
      ! Project the number of variables to update into allowable bounds.
      CONFIG%NUM_TO_UPDATE = MIN(CONFIG%NUM_VARS, MAX(CONFIG%FIT_MIN_TO_UPDATE, CONFIG%NUM_TO_UPDATE))
      ! Store the previous error for tracking the best-so-far.
      CONFIG%FIT_PREV_MSE = CONFIG%FIT_MSE
      ! Update the step number.
      CONFIG%FIT_NS = CONFIG%FIT_NS + 1
      ! Update the saved "best" model based on error.
      IF ((CONFIG%FIT_MSE .LT. CONFIG%FIT_BEST_MSE) .AND. &
           (CONFIG%STEPS_TAKEN .GE. CONFIG%MIN_STEPS_TO_STABILITY)) THEN
         CONFIG%FIT_NS = 0
         CONFIG%FIT_BEST_MSE = CONFIG%FIT_MSE
         IF (CONFIG%KEEP_BEST) THEN
            BEST_MODEL(:) = MODEL(:)
         END IF
      ! Early stop if we don't expect to see a better solution
      !  by the time the fit operation is complete.
      ELSE IF (CONFIG%EARLY_STOP .AND. (CONFIG%FIT_NS .GT. STEPS - CONFIG%FIT_STEP)) THEN
         CONFIG%FIT_NS = HUGE(CONFIG%FIT_NS)
      END IF
      ! Record the end of the total time.
      CALL CPU_TIME(CPU_TIME_END)
      CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
      !$OMP CRITICAL
      CONFIG%WRAT = CONFIG%WRAT + (WALL_TIME_END - WALL_TIME_START)
      CONFIG%CRAT = CONFIG%CRAT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
      !$OMP END CRITICAL
    END SUBROUTINE ADJUST_RATES

    
    ! Step the model variables.
    SUBROUTINE STEP_VARIABLES(MODEL_GRAD, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, UPDATE_INDICES, NB)
      REAL(KIND=RT), DIMENSION(:,:) :: MODEL_GRAD
      REAL(KIND=RT), DIMENSION(:) :: MODEL_GRAD_MEAN, MODEL_GRAD_CURV
      INTEGER(KIND=INT64), DIMENSION(:) :: UPDATE_INDICES
      INTEGER(KIND=INT64) :: NB
      INTEGER(KIND=INT64) :: I, NT, NP, S, E
      REAL :: CPU_TIME_START, CPU_TIME_END
      INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
      CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
      CALL CPU_TIME(CPU_TIME_START)
      NT = MIN(CONFIG%NUM_VARS, CONFIG%NUM_THREADS)
      NP = MAX(ONE, CONFIG%NUM_VARS / NT)
      ! TODO: Profile this code and find out what takes the longest for
      !       large number of aggregate embeddings (next letter prediction).
      ! 
      !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(S, E)
      DO I = 0, NT-1
         S = ONE+I*NP
         E = MIN((I+ONE)*NP, CONFIG%NUM_VARS)
         ! Aggregate over computed batches and compute average gradient.
         MODEL_GRAD(S:E,1) = SUM(MODEL_GRAD(S:E,:),2) / REAL(NB,RT)
         ! Mean.
         MODEL_GRAD_MEAN(S:E) = CONFIG%FIT_STEP_MEAN_REMAIN * MODEL_GRAD_MEAN(S:E) &
              + CONFIG%STEP_MEAN_CHANGE * MODEL_GRAD(S:E,1)
         ! Clip the mean to be small enough to be numerically stable.
         WHERE (ABS(MODEL_GRAD_MEAN(S:E)) .GT. CONFIG%MAX_STEP_COMPONENT)
            MODEL_GRAD_MEAN(S:E) = SIGN(CONFIG%MAX_STEP_COMPONENT, MODEL_GRAD_MEAN(S:E))
         END WHERE
         ! Curvature.
         MODEL_GRAD_CURV(S:E) = CONFIG%FIT_STEP_CURV_REMAIN * MODEL_GRAD_CURV(S:E) &
              + CONFIG%STEP_CURV_CHANGE * (MODEL_GRAD_MEAN(S:E) - MODEL_GRAD(S:E,1))**2
         ! Clip the curvature to be large enough to be numerically stable.
         WHERE (MODEL_GRAD_CURV(S:E) .LT. CONFIG%MIN_CURV_COMPONENT)
            MODEL_GRAD_CURV(S:E) = CONFIG%MIN_CURV_COMPONENT
         END WHERE
         ! Clip the curvature to be small enough to be numerically stable.
         WHERE (ABS(MODEL_GRAD_CURV(S:E)) .GT. CONFIG%MAX_CURV_COMPONENT)
            MODEL_GRAD_CURV(S:E) = SIGN(CONFIG%MAX_CURV_COMPONENT, MODEL_GRAD_CURV(S:E))
         END WHERE
         ! Set the step as the mean direction (over the past few steps).
         MODEL_GRAD(S:E,1) = MODEL_GRAD_MEAN(S:E)
         ! Start scaling by step magnitude by curvature once enough data is collected.
         IF (CONFIG%FIT_STEP .GE. CONFIG%MIN_STEPS_TO_STABILITY) THEN
            MODEL_GRAD(S:E,1) = MODEL_GRAD(S:E,1) / SQRT(MODEL_GRAD_CURV(S:E))
         END IF
         IF (CONFIG%NUM_TO_UPDATE .EQ. CONFIG%NUM_VARS) THEN
            ! Take the gradient steps (based on the computed "step" above).
            MODEL(S:E) = MODEL(S:E) - MODEL_GRAD(S:E,1) * CONFIG%STEP_FACTOR
         END IF
      END DO
      ! Update as many variables as it seems safe to update (and still converge).
      IF (CONFIG%NUM_TO_UPDATE .LT. CONFIG%NUM_VARS) THEN
         ! Identify the subset of components that will be updapted this step.
         CALL ARGSELECT(-ABS(MODEL_GRAD(:,1)), &
              INT(CONFIG%NUM_TO_UPDATE, KIND=INT64), UPDATE_INDICES(:))
         ! Take the gradient steps (based on the computed "step" above).
         !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(S, E)
         DO I = 0, NT-1
            S = ONE+I*NP
            E = MIN((I+ONE)*NP, CONFIG%NUM_VARS)
            MODEL(UPDATE_INDICES(S:E)) = MODEL(UPDATE_INDICES(S:E)) &
                 - MODEL_GRAD(UPDATE_INDICES(S:E),1) * CONFIG%STEP_FACTOR
         END DO
      END IF
      CONFIG%STEPS_TAKEN = CONFIG%STEPS_TAKEN + 1
      ! Record the end of the total time.
      CALL CPU_TIME(CPU_TIME_END)
      CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
      !$OMP CRITICAL
      CONFIG%WOPT = CONFIG%WOPT + (WALL_TIME_END - WALL_TIME_START)
      CONFIG%COPT = CONFIG%COPT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT), INT64) * CLOCK_RATE
      !$OMP END CRITICAL
    END SUBROUTINE STEP_VARIABLES

    ! Record various statistics that are currently of interest (for research).
    !   TODO: Remove this entirely in favor of tracking done in python.
    SUBROUTINE RECORD_STATS(MODEL_GRAD)
      REAL(KIND=RT), DIMENSION(:,:) :: MODEL_GRAD
      IF (PRESENT(RECORD)) THEN
         ! Store the mean squared error at this iteration.
         RECORD(1,CONFIG%FIT_STEP) = CONFIG%FIT_MSE
         ! Store the current multiplier on the step.
         RECORD(2,CONFIG%FIT_STEP) = CONFIG%STEP_FACTOR
         ! Store the norm of the step that was taken (intermittently).
         IF ((CONFIG%LOG_GRAD_NORM_FREQUENCY .GT. 0) .AND. &
              (MOD(CONFIG%STEPS_TAKEN-1,CONFIG%LOG_GRAD_NORM_FREQUENCY) .EQ. 0)) THEN
            RECORD(3,CONFIG%FIT_STEP) = SQRT(MAX(EPSILON(0.0_RT), SUM(MODEL_GRAD(:,1)**2))) / SQRT(REAL(CONFIG%NUM_VARS,RT))
         ELSE
            RECORD(3,CONFIG%FIT_STEP) = RECORD(3,CONFIG%FIT_STEP-1)
         END IF
         ! Store the percentage of variables updated in this step.
         RECORD(4,CONFIG%FIT_STEP) = REAL(CONFIG%NUM_TO_UPDATE,RT) / REAL(CONFIG%NUM_VARS)
         IF (CONFIG%FIT_TOTAL_RANK .GT. 0) THEN
            ! Store the evaluative utilization rate (total data rank over full rank)
            RECORD(5,CONFIG%FIT_STEP) = REAL(CONFIG%FIT_TOTAL_EVAL_RANK,RT) / REAL(CONFIG%FIT_TOTAL_RANK,RT)
            ! Store the gradient utilization rate (total gradient rank over full rank)
            RECORD(6,CONFIG%FIT_STEP) = REAL(CONFIG%FIT_TOTAL_GRAD_RANK,RT) / REAL(CONFIG%FIT_TOTAL_RANK,RT)
         END IF
      END IF
    END SUBROUTINE RECORD_STATS

  END SUBROUTINE FIT_MODEL

END MODULE AXY
