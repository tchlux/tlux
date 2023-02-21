! TODO:
! 
! - Make sure batch number is not used as THREAD_NUMBER anywhere.
! 
! - Experiment with 'OMP TARGET TEAMS DISTRIBUTE PARALLEL' to see if it uses GPU correctly.
! 
! - Rotate out the points that have the lowest expected change in error when batching.
! 
! - Add (PAIRWISE_AGGREGATION, MAX_PAIRS) functionality to the aggregator model input,
!   where a nonrepeating random number generator is used in conjunction with a mapping
!   from the integer line to pairs of points. When MAX_PAIRS is less than the total
!   number of pairs, then do greedy rotation of points identically to above.
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
!   gradient estimates to reflect those changes as well (might have to reset gradient).
! 
! - Update Python testing code to test all combinations of AX, AXI, AY, X, XI, and Y.
! - Update Python testing code to attempt different edge-case model sizes
!    (linear regression, no aggregator, no model).
! - Verify that the *condition model* operation correctly updates the gradient
!   related variables (mean and curvature). (resets back to initialization)
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
  USE ISO_FORTRAN_ENV, ONLY: RT => REAL32, INT64, INT8
  USE IEEE_ARITHMETIC, ONLY: IS_NAN => IEEE_IS_NAN, IS_FINITE => IEEE_IS_FINITE
  USE RANDOM, ONLY: RANDOM_UNIT_VECTORS
  USE SORT_AND_SELECT, ONLY: ARGSORT, ARGSELECT
  USE MATRIX_OPERATIONS, ONLY: GEMM, ORTHOGONALIZE, RADIALIZE, LEAST_SQUARES

  IMPLICIT NONE

  INTEGER(KIND=INT64) :: CLOCK_RATE, CLOCK_MAX

  ! Model configuration, internal sizes and fit parameters.
  TYPE, BIND(C) :: MODEL_CONFIG
     ! Aggregator model configuration.
     INTEGER :: ADN      ! aggregator dimension numeric (input)
     INTEGER :: ADE = 0  ! aggregator dimension of embeddings
     INTEGER :: ANE = 0  ! aggregator number of embeddings
     INTEGER :: ADS = 16 ! aggregator dimension of state
     INTEGER :: ANS = 1  ! aggregator number of states
     INTEGER :: ADO      ! aggregator dimension of output
     INTEGER :: ADI      ! aggregator dimension of input (internal usage only)
     INTEGER :: ADSO     ! aggregator dimension of state output (internal usage only)
     ! (Fixed) model configuration.
     INTEGER :: MDN      ! model dimension numeric (input)
     INTEGER :: MDE = 0  ! model dimension of embeddings
     INTEGER :: MNE = 0  ! model number of embeddings
     INTEGER :: MDS = 32 ! model dimension of state
     INTEGER :: MNS = 8  ! model number of states
     INTEGER :: MDO      ! model dimension of output
     INTEGER :: MDI      ! model dimension of input (internal usage only)
     INTEGER :: MDSO     ! model dimension of state output (internal usage only)
     ! Summary numbers that are computed.
     INTEGER :: DO  ! Dimension output (either MDO or ADO). 
     ! Model descriptors.
     INTEGER(KIND=INT64) :: TOTAL_SIZE
     INTEGER(KIND=INT64) :: NUM_VARS
     ! Index subsets of total size vector naming scheme:
     !   M___ -> model,   A___ -> aggregator model
     !   _S__ -> start,   _E__ -> end
     !   __I_ -> input,   __S_ -> states, __O_ -> output, __E_ -> embedding
     !   ___V -> vectors, ___S -> shifts
     INTEGER(KIND=INT64) :: ASIV, AEIV, ASIS, AEIS ! aggregator input (vec & shift)
     INTEGER(KIND=INT64) :: ASSV, AESV, ASSS, AESS ! aggregator states (vec & shift)
     INTEGER(KIND=INT64) :: ASOV, AEOV             ! aggregator output
     INTEGER(KIND=INT64) :: ASEV, AEEV             ! aggregator embedding
     INTEGER(KIND=INT64) :: MSIV, MEIV, MSIS, MEIS ! model input (vec & shift)
     INTEGER(KIND=INT64) :: MSSV, MESV, MSSS, MESS ! model states (vec & shift)
     INTEGER(KIND=INT64) :: MSOV, MEOV             ! model output
     INTEGER(KIND=INT64) :: MSEV, MEEV             ! model embedding
     ! Index subsets for data normalization.
     !   M___ -> model,  A___ -> aggregator (/ aggregate) model
     !   _I__ -> input,  _O__ -> output
     !   __S_ -> shift,  __M_ -> multiplier
     !   ___S -> start,  ___E -> end
     INTEGER(KIND=INT64) :: AISS, AISE, AOSS, AOSE
     INTEGER(KIND=INT64) :: AIMS, AIME
     INTEGER(KIND=INT64) :: MISS, MISE, MOSS, MOSE
     INTEGER(KIND=INT64) :: MIMS, MIME, MOMS, MOME
     ! Function parameter.
     REAL(KIND=RT) :: DISCONTINUITY = 0.0_RT
     ! Optimization related parameters.
     REAL(KIND=RT) :: STEP_FACTOR = 0.001_RT ! Initial multiplier on gradient steps.
     REAL(KIND=RT) :: MAX_STEP_FACTOR = 0.05_RT ! Maximum multiplier on gradient steps.
     REAL(KIND=RT) :: STEP_MEAN_CHANGE = 0.1_RT ! Rate of exponential sliding average over gradient steps.
     REAL(KIND=RT) :: STEP_CURV_CHANGE = 0.01_RT ! Rate of exponential sliding average over gradient variation.
     REAL(KIND=RT) :: STEP_AY_CHANGE = 0.05_RT ! Rate of exponential sliding average over AY (forcing mean to zero).
     REAL(KIND=RT) :: FASTER_RATE = 1.01_RT ! Rate of increase of optimization factors.
     REAL(KIND=RT) :: SLOWER_RATE = 0.99_RT ! Rate of decrease of optimization factors.
     REAL(KIND=RT) :: MIN_UPDATE_RATIO = 0.5_RT ! Minimum ratio of model variables to update in any optimizaiton step.
     REAL(KIND=RT) :: UPDATE_RATIO_STEP = 0.025_RT ! The step change in ratio of parameters updated when error is decreased.
     REAL(KIND=RT) :: ERROR_CHECK_RATIO = 0.0_RT ! Ratio of points used only to evaluate model error (set to 0 to use same data from optimization).
     INTEGER(KIND=INT64) :: MIN_STEPS_TO_STABILITY = 1 ! Minimum number of steps before allowing model saves and curvature approximation.
     INTEGER(KIND=INT64) :: BATCH_SIZE = 10000 ! Max number of points in one batch matrix multiplication.
     INTEGER(KIND=INT64) :: NUM_THREADS = 1 ! Number of parallel threads to use in fit & evaluation.
     INTEGER(KIND=INT64) :: PRINT_DELAY_SEC = 2 ! Delay between output logging during fit.
     INTEGER(KIND=INT64) :: STEPS_TAKEN = 0 ! Total number of updates made to model variables.
     INTEGER(KIND=INT64) :: LOG_GRAD_NORM_FREQUENCY = 100 ! Frequency with which to log expensive records (model variable 2-norm step size).
     INTEGER(KIND=INT64) :: RANK_CHECK_FREQUENCY = 0 ! Frequency with which to orthogonalize internal basis functions.
     INTEGER(KIND=INT64) :: NUM_TO_UPDATE = HUGE(0) ! Number of model variables to update (initialize to large number).
     LOGICAL(KIND=INT8) :: AX_NORMALIZED = .FALSE. ! False if AX data needs to be normalized.
     LOGICAL(KIND=INT8) :: RESCALE_AX = .TRUE. ! Rescale all AX components to be equally weighted.
     LOGICAL(KIND=INT8) :: AXI_NORMALIZED = .TRUE. ! False if AXI embeddings need to be normalized.
     LOGICAL(KIND=INT8) :: AY_NORMALIZED = .FALSE. ! False if aggregator outputs need to  be normalized.
     LOGICAL(KIND=INT8) :: X_NORMALIZED = .FALSE. ! False if X data needs to be normalized.
     LOGICAL(KIND=INT8) :: RESCALE_X = .TRUE. ! Rescale all X components to be equally weighted.
     LOGICAL(KIND=INT8) :: XI_NORMALIZED = .TRUE. ! False if XI embeddings need to be normalized.
     LOGICAL(KIND=INT8) :: Y_NORMALIZED = .FALSE. ! False if Y data needs to be normalized.
     LOGICAL(KIND=INT8) :: RESCALE_Y = .TRUE. ! Rescale all Y components to be equally weighted.
     LOGICAL(KIND=INT8) :: ENCODE_SCALING = .FALSE. ! True if input and output weight matrices shuld embed normalization scaling.
     LOGICAL(KIND=INT8) :: NEEDS_SCALING = .TRUE. ! True if input and output weight matrices are NOT already rescaled.
     LOGICAL(KIND=INT8) :: NORMALIZE = .TRUE. ! True if shifts and multipliers should be applied to inputs and outputs.
     LOGICAL(KIND=INT8) :: RANDOMIZE_ERROR_CHECK = .TRUE. ! True if the collection of points used for error checking should be randomly selected, false to use tail.
     LOGICAL(KIND=INT8) :: KEEP_BEST = .TRUE. ! True if best observed model should be greedily kept at end of optimization.
     LOGICAL(KIND=INT8) :: EARLY_STOP = .TRUE. ! True if optimization should end when num-steps since best model is greater than the num-steps remaining.
     LOGICAL(KIND=INT8) :: BASIS_REPLACEMENT = .FALSE. ! True if linearly dependent basis functions should be replaced during optimization rank checks.
     ! Descriptions of the number of points that can be in one batch.
     INTEGER(KIND=INT64) :: RWORK_SIZE = 0
     INTEGER(KIND=INT64) :: IWORK_SIZE = 0
     INTEGER(KIND=INT64) :: NA = 0
     INTEGER(KIND=INT64) :: NM = 0
     INTEGER(KIND=INT64) :: I_NEXT = 0
     INTEGER(KIND=INT64) :: I_STEP = 1
     INTEGER(KIND=INT64) :: I_MULT = 1
     INTEGER(KIND=INT64) :: I_MOD = 1
     ! Real work space (for model optimization).
     INTEGER(KIND=INT64) :: SMG, EMG ! MODEL_GRAD(NUM_VARS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMGM, EMGM ! MODEL_GRAD_MEAN(NUM_VARS)
     INTEGER(KIND=INT64) :: SMGC, EMGC ! MODEL_GRAD_CURV(NUM_VARS)
     INTEGER(KIND=INT64) :: SBM, EBM ! BEST_MODEL(NUM_VARS)
     INTEGER(KIND=INT64) :: SAXB, EAXB ! AX_BATCH(ADI,NA)
     INTEGER(KIND=INT64) :: SAXS, EAXS ! A_STATES(NA,ADS,ANS+1)
     INTEGER(KIND=INT64) :: SAXG, EAXG ! A_GRADS(NA,ADS,ANS+1)
     INTEGER(KIND=INT64) :: SAY, EAY ! AY(NA,ADO+1)
     INTEGER(KIND=INT64) :: SAYG, EAYG ! AY_GRADIENT(NA,ADO+1)
     INTEGER(KIND=INT64) :: SMXB, EMXB ! X_BATCH(MDI,NM)
     INTEGER(KIND=INT64) :: SMXS, EMXS ! M_STATES(NM,MDS,MNS+1)
     INTEGER(KIND=INT64) :: SMXG, EMXG ! M_GRADS(NM,MDS,MNS+1)
     INTEGER(KIND=INT64) :: SMYB, EMYB ! Y_BATCH(DO,NM)
     INTEGER(KIND=INT64) :: SYG, EYG ! Y_GRADIENT(MDO,NM)
     INTEGER(KIND=INT64) :: SAXIS, EAXIS ! AXI_SHIFT(ADE)
     INTEGER(KIND=INT64) :: SAXIR, EAXIR ! AXI_RESCALE(ADE,ADE)
     INTEGER(KIND=INT64) :: SMXIS, EMXIS ! XI_SHIFT(MDE)
     INTEGER(KIND=INT64) :: SMXIR, EMXIR ! XI_RESCALE(MDE,MDE)
     INTEGER(KIND=INT64) :: SAL, EAL ! A_LENGTHS(ADS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SML, EML ! M_LENGTHS(MDS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SAST, EAST ! A_STATE_TEMP(NA,ADS)
     INTEGER(KIND=INT64) :: SMST, EMST ! M_STATE_TEMP(NM,MDS)
     INTEGER(KIND=INT64) :: SAET, EAET ! A_EMB_TEMP(ADE,ANE,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMET, EMET ! M_EMB_TEMP(MDE,MNE,NUM_THREADS)
     INTEGER(KIND=INT64) :: SAEC, EAEC ! A_EMB_COUNTS(ANE,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMEC, EMEC ! M_EMB_COUNTS(MNE,NUM_THREADS)
     ! Integer workspace (for model optimization).
     INTEGER(KIND=INT64) :: SAXI, EAXI ! AXI (aggregate batch indices)
     INTEGER(KIND=INT64) :: SMXI, EMXI ! XI (model batch indices)
     INTEGER(KIND=INT64) :: SSB, ESB ! SB (sizes for batch)
     INTEGER(KIND=INT64) :: SAO, EAO ! A_ORDER(ADS,NUM_THREADS)
     INTEGER(KIND=INT64) :: SMO, EMO ! M_ORDER(MDS,NUM_THREADS)
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
        CONFIG%ADSO = 0
        CONFIG%ADO = 0
     ELSE IF (MIN(CONFIG%ANS,CONFIG%ADS) .EQ. 0) THEN
        CONFIG%ADO = MIN(16, CONFIG%ADI)
     ELSE
        CONFIG%ADO = MIN(16, CONFIG%ADS)
     END IF
     IF ((CONFIG%ADI .EQ. 0) .AND. (CONFIG%ADO .NE. 0)) THEN
        CONFIG%ADSO = 0
        CONFIG%ADO = 0
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
     ! DO
     IF (CONFIG%MDO .GT. 0_INT64) THEN
        CONFIG%DO = CONFIG%MDO
     ELSE
        CONFIG%DO = CONFIG%ADO
     END IF
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
     !   aggregator input vecs [ADI by ADS]
     CONFIG%ASIV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AEIV = CONFIG%ASIV-1_INT64 +  CONFIG%ADI * CONFIG%ADS
     CONFIG%TOTAL_SIZE = CONFIG%AEIV
     !   aggregator input shift [ADS]
     CONFIG%ASIS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AEIS = CONFIG%ASIS-1_INT64 +  CONFIG%ADS
     CONFIG%TOTAL_SIZE = CONFIG%AEIS
     !   aggregator state vecs [ADS by ADS by MAX(0,ANS-1)]
     CONFIG%ASSV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AESV = CONFIG%ASSV-1_INT64 +  CONFIG%ADS * CONFIG%ADS * MAX(0_INT64,CONFIG%ANS-1_INT64)
     CONFIG%TOTAL_SIZE = CONFIG%AESV
     !   aggregator state shift [ADS by MAX(0,ANS-1)]
     CONFIG%ASSS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AESS = CONFIG%ASSS-1_INT64 +  CONFIG%ADS * MAX(0_INT64,CONFIG%ANS-1_INT64)
     CONFIG%TOTAL_SIZE = CONFIG%AESS
     !   aggregator output vecs [ADSO by ADO+1]
     CONFIG%ASOV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AEOV = CONFIG%ASOV-1_INT64 +  CONFIG%ADSO * (CONFIG%ADO+1_INT64)
     CONFIG%TOTAL_SIZE = CONFIG%AEOV
     !   aggregator embedding vecs [ADE by ANE]
     CONFIG%ASEV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AEEV = CONFIG%ASEV-1_INT64 +  CONFIG%ADE * CONFIG%ANE
     CONFIG%TOTAL_SIZE = CONFIG%AEEV
     ! ---------------------------------------------------------------
     !   model input vecs [MDI by MDS]
     CONFIG%MSIV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MEIV = CONFIG%MSIV-1_INT64 +  CONFIG%MDI * CONFIG%MDS
     CONFIG%TOTAL_SIZE = CONFIG%MEIV
     !   model input shift [MDS]
     CONFIG%MSIS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MEIS = CONFIG%MSIS-1_INT64 +  CONFIG%MDS
     CONFIG%TOTAL_SIZE = CONFIG%MEIS
     !   model state vecs [MDS by MDS by MAX(0,MNS-1)]
     CONFIG%MSSV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MESV = CONFIG%MSSV-1_INT64 +  CONFIG%MDS * CONFIG%MDS * MAX(0_INT64,CONFIG%MNS-1_INT64)
     CONFIG%TOTAL_SIZE = CONFIG%MESV
     !   model state shift [MDS by MAX(0,MNS-1)]
     CONFIG%MSSS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MESS = CONFIG%MSSS-1_INT64 +  CONFIG%MDS * MAX(0_INT64,CONFIG%MNS-1_INT64)
     CONFIG%TOTAL_SIZE = CONFIG%MESS
     !   model output vecs [MDSO by MDO]
     CONFIG%MSOV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MEOV = CONFIG%MSOV-1_INT64 +  CONFIG%MDSO * CONFIG%MDO
     CONFIG%TOTAL_SIZE = CONFIG%MEOV
     !   model embedding vecs [MDE by MNE]
     CONFIG%MSEV = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MEEV = CONFIG%MSEV-1_INT64 +  CONFIG%MDE * CONFIG%MNE
     CONFIG%TOTAL_SIZE = CONFIG%MEEV
     ! THIS IS SPECIAL, IT IS PART OF THE MODEL AND CHANGES DURING OPTIMIZATION
     !   aggregator post-output shift [ADO]
     CONFIG%AOSS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AOSE = CONFIG%AOSS-1_INT64 + CONFIG%ADO
     CONFIG%TOTAL_SIZE = CONFIG%AOSE
     ! ---------------------------------------------------------------
     !   number of variables
     CONFIG%NUM_VARS = CONFIG%TOTAL_SIZE
     ! ---------------------------------------------------------------
     !   aggregator pre-input shift
     CONFIG%AISS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AISE = CONFIG%AISS-1_INT64 + CONFIG%ADN
     CONFIG%TOTAL_SIZE = CONFIG%AISE
     !   aggregator pre-input multiplier
     CONFIG%AIMS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%AIME = CONFIG%AIMS-1_INT64 + CONFIG%ADN * CONFIG%ADN
     CONFIG%TOTAL_SIZE = CONFIG%AIME
     !   model pre-input shift
     CONFIG%MISS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MISE = CONFIG%MISS-1_INT64 + CONFIG%MDN
     CONFIG%TOTAL_SIZE = CONFIG%MISE
     !   model pre-input multiplier
     CONFIG%MIMS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MIME = CONFIG%MIMS-1_INT64 + CONFIG%MDN * CONFIG%MDN
     CONFIG%TOTAL_SIZE = CONFIG%MIME
     !   model post-output shift
     CONFIG%MOSS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MOSE = CONFIG%MOSS-1_INT64 + CONFIG%DO
     CONFIG%TOTAL_SIZE = CONFIG%MOSE
     !   model post-output multiplier
     CONFIG%MOMS = 1_INT64 + CONFIG%TOTAL_SIZE
     CONFIG%MOME = CONFIG%MOMS-1_INT64 + CONFIG%DO * CONFIG%DO
     CONFIG%TOTAL_SIZE = CONFIG%MOME
     !   set all time counters to zero
     CONFIG%WFIT = 0_INT64
     CONFIG%CFIT = 0_INT64
     CONFIG%WNRM = 0_INT64
     CONFIG%CNRM = 0_INT64
     CONFIG%WGEN = 0_INT64
     CONFIG%CGEN = 0_INT64
     CONFIG%WEMB = 0_INT64
     CONFIG%CEMB = 0_INT64
     CONFIG%WEVL = 0_INT64
     CONFIG%CEVL = 0_INT64
     CONFIG%WGRD = 0_INT64
     CONFIG%CGRD = 0_INT64
     CONFIG%WRAT = 0_INT64
     CONFIG%CRAT = 0_INT64
     CONFIG%WOPT = 0_INT64
     CONFIG%COPT = 0_INT64
     CONFIG%WCON = 0_INT64
     CONFIG%CCON = 0_INT64
     CONFIG%WREC = 0_INT64
     CONFIG%CREC = 0_INT64
     CONFIG%WENC = 0_INT64
     CONFIG%CENC = 0_INT64
  END SUBROUTINE NEW_MODEL_CONFIG

  ! Given a number of X points "NM", and a number of aggregator X points
  ! "NA", update the "RWORK_SIZE" and "IWORK_SIZE" attributes in "CONFIG"
  ! as well as all related work indices for that size data.
  SUBROUTINE NEW_FIT_CONFIG(NM, NA, NMT, NAT, ADI, MDI, SEED, CONFIG)
    INTEGER(KIND=INT64), INTENT(IN) :: NM
    INTEGER(KIND=INT64), INTENT(IN), OPTIONAL :: NA, NMT, NAT, ADI, MDI
    INTEGER, INTENT(IN), OPTIONAL :: SEED
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    INTEGER(KIND=INT64) :: AXI_COLS, XI_COLS, NM_TOTAL, NA_TOTAL
    CONFIG%NM = NM
    IF (PRESENT(NA)) THEN
       CONFIG%NA = NA
    ELSE
       CONFIG%NA = 0
    END IF
    IF (PRESENT(NMT)) THEN
       NM_TOTAL = NMT
    ELSE
       NM_TOTAL = NM
    END IF
    IF (PRESENT(NAT)) THEN
       NA_TOTAL = NAT
    ELSE
       NA_TOTAL = NA
    END IF
    IF (PRESENT(ADI)) THEN
       AXI_COLS = ADI
    ELSE
       AXI_COLS = 0_INT64
    END IF
    IF (PRESENT(MDI)) THEN
       XI_COLS = MDI
    ELSE
       XI_COLS = 0_INT64
    END IF
    ! ------------------------------------------------------------
    ! Set up the indexing for batch iteration.
    IF (PRESENT(SEED)) THEN
       CALL SRAND(SEED)
    ELSE
       CALL SRAND(0)
    END IF
    ! Construct an additive term, multiplier, and modulus for a linear
    ! congruential generator. These generators are cyclic and do not
    ! repeat when they maintain the properties:
    ! 
    !   1) "modulus" and "additive term" are relatively prime.
    !   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    !   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    ! 
    CONFIG%I_NEXT = MOD(INT(IRAND(),INT64), NM_TOTAL) ! Pick a random initial value.
    CONFIG%I_STEP = MOD(INT(IRAND(),INT64), NM_TOTAL)*2_INT64 + 1_INT64 ! Pick a random odd-valued additive term.
    CONFIG%I_MULT = 4_INT64 * (NM * MOD(INT(IRAND(),INT64), NM_TOTAL)) + 1_INT64 ! Pick a multiplier 1 greater than a multiple of 4.
    CONFIG%I_MOD = 2_INT64 ** CEILING(LOG(REAL(NM_TOTAL)) / LOG(2.0)) ! Pick a power-of-2 modulus just big enough to generate all numbers.
    ! ------------------------------------------------------------
    ! Set up the real valued work array.
    CONFIG%RWORK_SIZE = 0
    ! model gradient
    CONFIG%SMG = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMG = CONFIG%SMG-1_INT64 + CONFIG%NUM_VARS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EMG
    ! model gradient mean
    CONFIG%SMGM = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMGM = CONFIG%SMGM-1_INT64 + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EMGM
    ! model gradient curvature
    CONFIG%SMGC = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMGC = CONFIG%SMGC-1_INT64 + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EMGC
    ! best model
    CONFIG%SBM = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EBM = CONFIG%SBM-1_INT64 + CONFIG%NUM_VARS
    CONFIG%RWORK_SIZE = CONFIG%EBM
    ! ---------------------------------------------------------------
    ! AX batch
    CONFIG%SAXB = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAXB = CONFIG%SAXB-1_INT64 + CONFIG%NA * CONFIG%ADI
    CONFIG%RWORK_SIZE = CONFIG%EAXB
    ! aggregator states
    CONFIG%SAXS = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAXS = CONFIG%SAXS-1_INT64 + CONFIG%NA * CONFIG%ADS * (CONFIG%ANS+1_INT64)
    CONFIG%RWORK_SIZE = CONFIG%EAXS
    ! aggregator gradients at states
    CONFIG%SAXG = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAXG = CONFIG%SAXG-1_INT64 + CONFIG%NA * CONFIG%ADS * (CONFIG%ANS+1_INT64)
    CONFIG%RWORK_SIZE = CONFIG%EAXG
    ! AY
    CONFIG%SAY = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAY = CONFIG%SAY-1_INT64 + CONFIG%NA * (CONFIG%ADO+1_INT64)
    CONFIG%RWORK_SIZE = CONFIG%EAY
    ! AY gradient
    CONFIG%SAYG = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAYG = CONFIG%SAYG-1_INT64 + CONFIG%NA * (CONFIG%ADO+1_INT64)
    CONFIG%RWORK_SIZE = CONFIG%EAYG
    ! X batch
    CONFIG%SMXB = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMXB = CONFIG%SMXB-1_INT64 + CONFIG%NM * CONFIG%MDI
    CONFIG%RWORK_SIZE = CONFIG%EMXB
    ! model states
    CONFIG%SMXS = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMXS = CONFIG%SMXS-1_INT64 + CONFIG%NM * CONFIG%MDS * (CONFIG%MNS+1_INT64)
    CONFIG%RWORK_SIZE = CONFIG%EMXS
    ! model gradients at states
    CONFIG%SMXG = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMXG = CONFIG%SMXG-1_INT64 + CONFIG%NM * CONFIG%MDS * (CONFIG%MNS+1_INT64)
    CONFIG%RWORK_SIZE = CONFIG%EMXG
    ! Y batch
    CONFIG%SMYB = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMYB = CONFIG%SMYB-1_INT64 + CONFIG%DO * CONFIG%NM
    CONFIG%RWORK_SIZE = CONFIG%EMYB
    ! Y gradient
    CONFIG%SYG = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EYG = CONFIG%SYG-1_INT64 + CONFIG%DO * CONFIG%NM
    CONFIG%RWORK_SIZE = CONFIG%EYG
    ! AXI shift
    CONFIG%SAXIS = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAXIS = CONFIG%SAXIS-1_INT64 + CONFIG%ADE
    CONFIG%RWORK_SIZE = CONFIG%EAXIS
    ! AXI rescale
    CONFIG%SAXIR = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAXIR = CONFIG%SAXIR-1_INT64 + CONFIG%ADE * CONFIG%ADE
    ! XI shift
    CONFIG%SMXIS = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMXIS = CONFIG%SMXIS-1_INT64 + CONFIG%MDE
    CONFIG%RWORK_SIZE = CONFIG%EMXIS
    ! XI rescale
    CONFIG%SMXIR = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMXIR = CONFIG%SMXIR-1_INT64 + CONFIG%MDE * CONFIG%MDE
    CONFIG%RWORK_SIZE = CONFIG%EMXIR
    ! A lengths (lengths of state values after orthogonalization)
    CONFIG%SAL = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAL = CONFIG%SAL-1_INT64 + CONFIG%ADS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EAL
    ! M lengths (lengths of state values after orthogonalization)
    CONFIG%SML = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EML = CONFIG%SML-1_INT64 + CONFIG%MDS * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EML
    ! A state temp holder (for orthogonality computation)
    CONFIG%SAST = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAST = CONFIG%SAST-1_INT64 + CONFIG%NA * CONFIG%ADS
    CONFIG%RWORK_SIZE = CONFIG%EAST
    ! M state temp holder (for orthogonality computation)
    CONFIG%SMST = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMST = CONFIG%SMST-1_INT64 + CONFIG%NM * CONFIG%MDS
    CONFIG%RWORK_SIZE = CONFIG%EMST
    ! A embedding temp holder
    CONFIG%SAET = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAET = CONFIG%SAET-1_INT64 + CONFIG%ADE * CONFIG%ANE * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EAET
    ! M embedding temp holder
    CONFIG%SMET = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMET = CONFIG%SMET-1_INT64 + CONFIG%MDE * CONFIG%MNE * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EMET
    ! A embedding temp counter
    CONFIG%SAEC = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EAEC = CONFIG%SAEC-1_INT64 + CONFIG%ANE * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EAEC
    ! M embedding temp counter
    CONFIG%SMEC = 1_INT64 + CONFIG%RWORK_SIZE
    CONFIG%EMEC = CONFIG%SMEC-1_INT64 + CONFIG%MNE * CONFIG%NUM_THREADS
    CONFIG%RWORK_SIZE = CONFIG%EMEC
    ! ------------------------------------------------------------
    ! Set up the integer valued work array.
    CONFIG%IWORK_SIZE = 0_INT64
    ! aggregate batch indices
    CONFIG%SAXI = 1_INT64 + CONFIG%IWORK_SIZE
    CONFIG%EAXI = CONFIG%SAXI-1_INT64 + CONFIG%NA * AXI_COLS
    CONFIG%IWORK_SIZE = CONFIG%EAXI
    ! model batch indices
    CONFIG%SMXI = 1_INT64 + CONFIG%IWORK_SIZE
    CONFIG%EMXI = CONFIG%SMXI-1_INT64 + CONFIG%NM * XI_COLS
    CONFIG%IWORK_SIZE = CONFIG%EMXI
    ! sizes for batch
    CONFIG%SSB = 1_INT64 + CONFIG%IWORK_SIZE
    CONFIG%ESB = CONFIG%SSB-1_INT64 + CONFIG%NM
    CONFIG%IWORK_SIZE = CONFIG%ESB
    ! A order (for orthogonalization)
    CONFIG%SAO = 1_INT64 + CONFIG%IWORK_SIZE
    CONFIG%EAO = CONFIG%SAO-1_INT64 + CONFIG%ADS * CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EAO
    ! M order (for orthogonalization)
    CONFIG%SMO = 1_INT64 + CONFIG%IWORK_SIZE
    CONFIG%EMO = CONFIG%SMO-1_INT64 + CONFIG%MDS * CONFIG%NUM_THREADS
    CONFIG%IWORK_SIZE = CONFIG%EMO
  END SUBROUTINE NEW_FIT_CONFIG

  ! Initialize the weights for a model, optionally provide a random seed.
  SUBROUTINE INIT_MODEL(CONFIG, MODEL, SEED, INITIAL_SHIFT_RANGE, INITIAL_OUTPUT_SCALE)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    INTEGER, INTENT(IN), OPTIONAL :: SEED
    REAL(KIND=RT), INTENT(IN), OPTIONAL :: INITIAL_SHIFT_RANGE, INITIAL_OUTPUT_SCALE
    REAL(KIND=RT) :: SHIFT_RANGE, OUTPUT_SCALE
    !  Storage for seeding the random number generator (for repeatability). LOCAL ALLOCATION
    INTEGER, DIMENSION(:), ALLOCATABLE :: SEED_ARRAY
    ! Local iterator.
    INTEGER :: I
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
    IF (PRESENT(INITIAL_OUTPUT_SCALE)) THEN
       OUTPUT_SCALE = INITIAL_OUTPUT_SCALE
    ELSE
       OUTPUT_SCALE = 0.1_RT
    END IF
    ! Set a random seed, if one was provided (otherwise leave default).
    IF (PRESENT(SEED)) THEN
       CALL RANDOM_SEED(SIZE=I)
       ALLOCATE(SEED_ARRAY(I))
       SEED_ARRAY(:) = SEED
       CALL RANDOM_SEED(PUT=SEED_ARRAY(:))
       DEALLOCATE(SEED_ARRAY)
    END IF
    ! Initialize the fixed model.
    CALL INIT_SUBMODEL(&
         CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, &
         CONFIG%MDSO, CONFIG%MDO, CONFIG%MDE, CONFIG%MNE, &
         MODEL(CONFIG%MSIV:CONFIG%MEIV), &
         MODEL(CONFIG%MSIS:CONFIG%MEIS), &
         MODEL(CONFIG%MSSV:CONFIG%MESV), &
         MODEL(CONFIG%MSSS:CONFIG%MESS), &
         MODEL(CONFIG%MSOV:CONFIG%MEOV), &
         MODEL(CONFIG%MSEV:CONFIG%MEEV))
    ! 
    ! Initialize the aggregator model.
    CALL INIT_SUBMODEL(&
         CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, &
         CONFIG%ADSO, CONFIG%ADO+1, CONFIG%ADE, CONFIG%ANE, &
         MODEL(CONFIG%ASIV:CONFIG%AEIV), &
         MODEL(CONFIG%ASIS:CONFIG%AEIS), &
         MODEL(CONFIG%ASSV:CONFIG%AESV), &
         MODEL(CONFIG%ASSS:CONFIG%AESS), &
         MODEL(CONFIG%ASOV:CONFIG%AEOV), &
         MODEL(CONFIG%ASEV:CONFIG%AEEV))
    ! Set the normalization shifts to zero and multipliers to the identity.
    !   aggregator input shift,
    MODEL(CONFIG%AISS:CONFIG%AISE) = 0.0_RT    
    !   aggregator input multiplier,
    MODEL(CONFIG%AIMS:CONFIG%AIME) = 0.0_RT
    MODEL(CONFIG%AIMS:CONFIG%AIME:CONFIG%ADN+1) = 1.0_RT
    !   input shift,
    MODEL(CONFIG%MISS:CONFIG%MISE) = 0.0_RT    
    !   input multiplier,
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
    CONFIG%CINT = CONFIG%CINT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)


  CONTAINS
    ! Initialize the model after unpacking it into its constituent parts.
    SUBROUTINE INIT_SUBMODEL(MDI, MDS, MNS, MDSO, MDO, MDE, MNE, &
         INPUT_VECS, INPUT_SHIFT, STATE_VECS, STATE_SHIFT, &
         OUTPUT_VECS, EMBEDDINGS)
      INTEGER, INTENT(IN) :: MDI, MDS, MNS, MDSO, MDO, MDE, MNE
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDI, MDS) :: INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS) :: INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS, MDS, MAX(0_INT64,MNS-1_INT64)) :: STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDS, MAX(0_INT64,MNS-1_INT64)) :: STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDSO, MDO) :: OUTPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDE, MNE) :: EMBEDDINGS
      ! Local holder for "origin" at each layer.
      REAL(KIND=RT) :: R, D
      REAL(KIND=RT), DIMENSION(MDS) :: ORIGIN ! LOCAL ALLOCATION
      INTEGER(KIND=INT64), DIMENSION(MDS) :: ORDER  ! LOCAL ALLOCATION
      INTEGER(KIND=INT64) :: I, J
      ! Generate well spaced random unit-length vectors (no scaling biases)
      ! for all initial variables in the input, internal, output, and embedings.
      CALL RANDOM_UNIT_VECTORS(INPUT_VECS(:,:))
      DO I = 1, MNS-1
         CALL RANDOM_UNIT_VECTORS(STATE_VECS(:,:,I))
      END DO
      CALL RANDOM_UNIT_VECTORS(OUTPUT_VECS(:,:))
      CALL RANDOM_UNIT_VECTORS(EMBEDDINGS(:,:))
      ! Multiply the embeddings by random lengths to make them better spaced,
      !  specifically, try to make them uniformly distributed in the ball.
      D = 1.0_RT / REAL(MDE, RT)
      DO I = 1, MNE
         CALL RANDOM_NUMBER(R)
         EMBEDDINGS(:,I) = EMBEDDINGS(:,I) * R**D
      END DO
      ! Make the output vectors have very small magnitude initially.
      OUTPUT_VECS(:,:) = OUTPUT_VECS(:,:) * OUTPUT_SCALE
      ! Generate deterministic equally spaced shifts for inputs and internal layers, 
      !  zero shift for the output layer (first two will be rescaled).
      DO I = 1, MDS
         INPUT_SHIFT(I) = 2.0_RT * SHIFT_RANGE * &             ! 2 * shift *
              (REAL(I-1,RT) / MAX(1.0_RT, REAL(MDS-1, RT))) &  ! range [0, 1]
              - SHIFT_RANGE                                    ! - shift
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
    END SUBROUTINE INIT_SUBMODEL
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
    IF (SIZE(MODEL,KIND=INT64) .NE. CONFIG%TOTAL_SIZE) THEN
       INFO = 1 ! Model size does not match model configuration.
    ELSE IF (SIZE(X,2,INT64) .NE. SIZE(Y,2,INT64)) THEN
       INFO = 2 ! Input arrays do not match in size.
    ELSE IF (SIZE(X,1,INT64) .NE. CONFIG%MDN) THEN
       INFO = 3 ! X input dimension is bad.
    ELSE IF ((CONFIG%MDO .GT. 0) .AND. (SIZE(Y,1,INT64) .NE. CONFIG%MDO)) THEN
       INFO = 4 ! Output dimension is bad.
    ELSE IF ((CONFIG%MDO .EQ. 0) .AND. (SIZE(Y,1,INT64) .NE. CONFIG%ADO)) THEN
       INFO = 5 ! Output dimension is bad.
    ELSE IF ((CONFIG%MNE .GT. 0) .AND. (SIZE(XI,2,INT64) .NE. SIZE(X,2,INT64))) THEN
       INFO = 6 ! Input integer XI size does not match X.
    ELSE IF ((MINVAL(XI) .LT. 0) .OR. (MAXVAL(XI) .GT. CONFIG%MNE)) THEN
       INFO = 7 ! Input integer X index out of range.
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (SIZE(SIZES) .NE. SIZE(Y,2,INT64))) THEN
       INFO = 8 ! SIZES has wrong size.
    ELSE IF (SIZE(AX,2,INT64) .NE. SUM(SIZES)) THEN
       INFO = 9 ! AX and SUM(SIZES) do not match.
    ELSE IF (SIZE(AX,1,INT64) .NE. CONFIG%ADN) THEN
       INFO = 10 ! AX input dimension is bad.
    ELSE IF (SIZE(AXI,2,INT64) .NE. SIZE(AX,2,INT64)) THEN
       INFO = 11 ! Input integer AXI size does not match AX.
    ELSE IF ((MINVAL(AXI) .LT. 0) .OR. (MAXVAL(AXI) .GT. CONFIG%ANE)) THEN
       INFO = 12 ! Input integer AX index out of range.
    END IF
  END SUBROUTINE CHECK_SHAPE

 
  ! Given a number of batches, compute the batch start and ends for
  !  the aggregator and positional inputs. Store in (2,_) arrays.
  SUBROUTINE COMPUTE_BATCHES(CONFIG, NA, NM, SIZES, BATCHA_STARTS, &
       BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS, JOINT, INFO)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    INTEGER(KIND=INT64), INTENT(IN) :: NA, NM
    INTEGER, INTENT(IN),  DIMENSION(:) :: SIZES
    INTEGER(KIND=INT64), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: BATCHA_STARTS, BATCHA_ENDS
    INTEGER(KIND=INT64), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: AGG_STARTS
    INTEGER(KIND=INT64), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: BATCHM_STARTS, BATCHM_ENDS
    LOGICAL, INTENT(IN) :: JOINT
    INTEGER, INTENT(INOUT) :: INFO
    ! Local variables.
    INTEGER(KIND=INT64) :: BATCH, BN, BE, BS, I, NUM_A_BATCHES, NUM_M_BATCHES
    ! Check for errors.
    IF (CONFIG%NUM_THREADS .LT. 1) THEN
       WRITE (*,*) 'ERROR (COMPUTE_BATCHES): Number of threads (NUM_THREADS) is not positive.', CONFIG%NUM_THREADS
       INFO = -1 ! Number of threads (NUM_THREADS) is not positive (is 0 or negative).
       RETURN
    ELSE IF (CONFIG%BATCH_SIZE .LT. 1) THEN
       WRITE (*,*) 'ERROR (COMPUTE_BATCHES): Batch size (BATCH_SIZE) is not positive.', CONFIG%BATCH_SIZE
       INFO = -2 ! Number of points per batch (BATCH_SIZE) is not positive (is 0 or negative).
       RETURN
    ELSE IF (NM .EQ. 0) THEN
       WRITE (*,*) 'ERROR (COMPUTE_BATCHES): Number of points not positive.', NM
       INFO = -3 ! Number of points is not positive (is 0 or negative).
       RETURN
    END IF
    ! Compute batch sizes and allocate space.
    IF ((.NOT. JOINT) .OR. (NA .EQ. 0)) THEN
       ! Batches are computed independently.
       NUM_A_BATCHES = MAX(MIN(NA, CONFIG%NUM_THREADS), &
            NA / CONFIG%BATCH_SIZE)
       NUM_M_BATCHES = MAX(MIN(NM, CONFIG%NUM_THREADS), &
            NM / CONFIG%BATCH_SIZE)
    ELSE
       ! Batches are computed jointly, determining batch sizes by
       !  iterating over data (in order) and ending a batch when
       !  either NA or NM limits are hit.
       NUM_M_BATCHES = 1_INT64
       BS = 1_INT64
       BE = 0_INT64
       DO I = 1_INT64, SIZE(SIZES)
          BE = BE + SIZES(I)
          ! Transition batches based on size of next iterate.
          IF (I .LT. SIZE(SIZES)) THEN
             IF (SIZES(I+1_INT64) + BE - BS + 1_INT64 .GT. CONFIG%BATCH_SIZE) THEN
                BS = BE + 1_INT64
                NUM_M_BATCHES = NUM_M_BATCHES + 1_INT64
             END IF
          END IF
       END DO
       NUM_A_BATCHES = NUM_M_BATCHES
       ! ^ joint batching is needed for higher level parallelization
       !   in the training loop, which will allow for increased throughput.
    END IF
    ! Deallocate arrays if they were already allocated (should not be).
    IF (ALLOCATED(BATCHA_STARTS)) DEALLOCATE(BATCHA_STARTS)
    IF (ALLOCATED(BATCHA_ENDS)) DEALLOCATE(BATCHA_ENDS)
    IF (ALLOCATED(AGG_STARTS)) DEALLOCATE(AGG_STARTS)
    IF (ALLOCATED(BATCHM_STARTS)) DEALLOCATE(BATCHM_STARTS)
    IF (ALLOCATED(BATCHM_ENDS)) DEALLOCATE(BATCHM_ENDS)
    ! Allocate new arrays for batching.
    ALLOCATE( &
         BATCHM_STARTS(1:NUM_M_BATCHES), &
         BATCHM_ENDS(1:NUM_M_BATCHES), &
         BATCHA_STARTS(1:NUM_A_BATCHES), &
         BATCHA_ENDS(1:NUM_A_BATCHES), &
         AGG_STARTS(1:NM))
    ! Construct batches for data sets with aggregator inputs.
    IF (NA .GT. 0_INT64) THEN
       ! Compute the location of the first index in each aggregate set.
       AGG_STARTS(1) = 1_INT64
       DO I = 2_INT64, NM
          AGG_STARTS(I) = AGG_STARTS(I-1_INT64) + SIZES(I-1_INT64)
       END DO
       ! Handle number of batches.
       IF (NUM_A_BATCHES .EQ. 1_INT64) THEN
          BATCHA_STARTS(1) = 1_INT64
          BATCHA_ENDS(1) = NA
          BATCHM_STARTS(1) = 1_INT64
          BATCHM_ENDS(1) = NM
       ELSE
          IF (.NOT. JOINT) THEN
             ! Construct aggregate batches.
             BN = (NA + NUM_A_BATCHES - 1_INT64) / NUM_A_BATCHES ! = CEIL(NA / NUM_BATCHES)
             DO BATCH = 1, NUM_A_BATCHES
                BATCHA_STARTS(BATCH) = MIN(NA+1_INT64, BN*(BATCH-1_INT64) + 1_INT64)
                BATCHA_ENDS(BATCH) = MIN(NA, BN*BATCH)
             END DO
             ! Construct positional batches.
             BN = (NM + NUM_M_BATCHES - 1_INT64) / NUM_M_BATCHES ! = CEIL(NM / NUM_BATCHES)
             DO BATCH = 1, NUM_M_BATCHES
                BATCHM_STARTS(BATCH) = BN*(BATCH-1_INT64) + 1_INT64
                BATCHM_ENDS(BATCH) = MIN(NM, BN*BATCH)
             END DO
          ELSE
             ! Compute the joint batches over the data, with end-to-end parallelization
             !  and more jobs instead of granular parallelization (with more barriers).
             BATCH = 1
             BATCHM_STARTS(BATCH) = 1_INT64
             BS = 1_INT64
             BE = 0_INT64
             DO I = 1_INT64, SIZE(SIZES)-1_INT64
                BE = BE + SIZES(I)
                ! Transition batches based on size of next iterate.
                IF (SIZES(I+1_INT64) + BE - BS + 1_INT64 .GT. CONFIG%BATCH_SIZE) THEN
                   BATCHA_STARTS(BATCH) = BS
                   BATCHA_ENDS(BATCH) = BE
                   BS = BE + 1_INT64
                   BATCHM_ENDS(BATCH) = I
                   BATCH = BATCH + 1_INT64
                   BATCHM_STARTS(BATCH) = I+1
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
       BATCHA_STARTS(:) = 1
       BATCHA_ENDS(:) = 0
       AGG_STARTS(:) = 1
       IF (NUM_M_BATCHES .EQ. 1) THEN
          BATCHM_STARTS(1) = 1
          BATCHM_ENDS(1) = NM
       ELSE
          ! Construct positional batches.
          BN = (NM + NUM_M_BATCHES - 1_INT64) / NUM_M_BATCHES ! = CEIL(NM / NUM_BATCHES)
          DO BATCH = 1, NUM_M_BATCHES
             BATCHM_STARTS(BATCH) = BN*(BATCH-1_INT64) + 1_INT64
             BATCHM_ENDS(BATCH) = MIN(NM, BN*BATCH)
          END DO
       END IF
    END IF
  END SUBROUTINE COMPUTE_BATCHES


  ! Get the next index in the model point iterator.
  FUNCTION GET_NEXT_INDEX(CONFIG) RESULT(NEXT_I)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    INTEGER(KIND=INT64) :: NEXT_I
    ! If the NEXT_I is not valid, cycle it until it is.
    DO WHILE (CONFIG%I_NEXT .GE. CONFIG%NM)
       CONFIG%I_NEXT = MOD(CONFIG%I_NEXT * CONFIG%I_MULT + CONFIG%I_STEP, CONFIG%I_MOD)
    END DO
    ! Store the "NEXT_I" that is currently set.
    NEXT_I = CONFIG%I_NEXT + 1_INT64
    ! Cycle I_NEXT to the next value in the sequence.
    CONFIG%I_NEXT = MOD(CONFIG%I_NEXT * CONFIG%I_MULT + CONFIG%I_STEP, CONFIG%I_MOD)
  END FUNCTION GET_NEXT_INDEX


  ! Give the raw input data, fetch a new set of data that fits in memory.
  SUBROUTINE FETCH_DATA(CONFIG, MODEL, &
       AX_IN, AX, AXI_IN, AXI, SIZES_IN, SIZES, &
       X_IN, X, XI_IN, XI, Y_IN, Y, YW_IN, YW, NA )
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(IN), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AX_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER, INTENT(IN), DIMENSION(:,:) :: AXI_IN
    INTEGER, INTENT(INOUT), DIMENSION(:,:) :: AXI
    INTEGER, INTENT(IN), DIMENSION(:) :: SIZES_IN
    INTEGER, INTENT(INOUT), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: X_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER, INTENT(IN), DIMENSION(:,:) :: XI_IN
    INTEGER, INTENT(INOUT), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: Y_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: YW_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW
    INTEGER(KIND=INT64), INTENT(OUT) :: NA
    ! TODO: Add a flag PAIRWISE that can be pasesd in to enable
    !       pairwise comparisons between aggregate inputs.
    ! 
    ! Local variables.
    INTEGER(KIND=INT64) :: I, J, MAX_AGG, NEXTRA, AS, AN, AINS, GENDEX, START_VALUE
    REAL(KIND=RT) :: NREMAIN, CURRENT_TOTAL
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:) :: RSIZES
    INTEGER(KIND=INT64), ALLOCATABLE, DIMENSION(:) :: SORTED_ORDER, AGG_STARTS
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Allocate storage space if it will be used.
    IF (SIZE(SIZES_IN) .GT. 0) THEN
       ALLOCATE(RSIZES(1:CONFIG%NM), SORTED_ORDER(1:CONFIG%NM))
    END IF
    ! Pack the regular inputs into the working space, storing sizes.
    ! TODO: Instead of iterating over NM points, we need to use a linear generator here.
    START_VALUE = CONFIG%I_NEXT
    DO I = 1, CONFIG%NM
       GENDEX = GET_NEXT_INDEX(CONFIG)
       ! Store the size.
       IF (SIZE(SIZES_IN) .GT. 0) THEN
          RSIZES(I) = REAL(SIZES_IN(GENDEX), KIND=RT)
       END IF
       X(:CONFIG%MDN,I) = X_IN(:,GENDEX)
       XI(:,I) = XI_IN(:,GENDEX)
       Y(:,I) = Y_IN(:,GENDEX)
       YW(:,I) = YW_IN(:,GENDEX)
    END DO
    ! If there are aggregate inputs ...
    IF (SIZE(SIZES_IN) .GT. 0) THEN
       ! Determine a maximum allowed size that will permit the most
       !  points to have the most possible aggregate inputs.
       CALL ARGSORT(RSIZES(:), SORTED_ORDER(:))
       NREMAIN = REAL(CONFIG%NA,RT)
       compute_max_agg : DO I = 1, CONFIG%NM
          ! Assuming all elements had the size of the current element, would they fit?
          CURRENT_TOTAL = RSIZES(SORTED_ORDER(I)) * REAL(CONFIG%NM - I + 1_INT64, RT)
          ! If they do not fit ...
          IF (CURRENT_TOTAL .GT. NREMAIN) THEN
             ! Determine the maximum size that WOULD fit for the current (and all remaining) elements.
             MAX_AGG = INT(NREMAIN, INT64) / (CONFIG%NM - I + 1_INT64)
             ! Count the remainder, the number of aggregate sets that can have 1 extra.
             NEXTRA = INT(NREMAIN, INT64) - MAX_AGG * (CONFIG%NM - I + 1_INT64)
             ! Set the size for those inputs that get the MAX + 1.
             DO J = I, I+NEXTRA-1_INT64
                SIZES(SORTED_ORDER(J)) = MAX_AGG + 1_INT64
             END DO
             ! Set the size for those inputs that get the MAX permissible aggregate elements.
             DO J = I+NEXTRA, CONFIG%NM
                SIZES(SORTED_ORDER(J)) = MAX_AGG
             END DO
             NREMAIN = 0_RT
             EXIT compute_max_agg
          END IF
          SIZES(SORTED_ORDER(I)) = INT(RSIZES(SORTED_ORDER(I)))
          NREMAIN = NREMAIN - RSIZES(SORTED_ORDER(I))
       END DO compute_max_agg
       ! Deallocate memory that is no longer needed.
       DEALLOCATE(RSIZES, SORTED_ORDER)
       ! Compute the start indices for the different aggregate sets.
       ALLOCATE(AGG_STARTS(SIZE(SIZES_IN)))
       AGG_STARTS(1) = 1
       DO I = 1, SIZE(SIZES_IN)-1
          AGG_STARTS(I+1) = AGG_STARTS(I) + SIZES_IN(I)
       END DO
       ! Pack in the aggregate inputs.
       AS = 1_INT64
       CONFIG%I_NEXT = START_VALUE
       DO I = 1, CONFIG%NM
          GENDEX = GET_NEXT_INDEX(CONFIG)          
          ! Get the permitted size.
          AN = SIZES(I) - 1_INT64
          ! Pack in those inputs.
          ! TODO: this needs to be a linear generator from AX_IN.
          AX(:CONFIG%ADN,AS:AS+AN) = AX_IN(:,AGG_STARTS(GENDEX):AGG_STARTS(GENDEX)+AN)
          AXI(:,AS:AS+AN) = AXI_IN(:,AGG_STARTS(GENDEX):AGG_STARTS(GENDEX)+AN)
          ! Update A Start.
          AS = AS + AN + 1_INT64
       END DO
       ! Deallocate memory for identifying start of aggregate sets.
       DEALLOCATE(AGG_STARTS)
       ! Set the total number of aggregate inputs that were added.
       NA = CONFIG%NA - INT(NREMAIN, INT64)
    ELSE
       NA = 0_INT64
    END IF
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WGEN = CONFIG%WGEN + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CGEN = CONFIG%CGEN + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)
  END SUBROUTINE FETCH_DATA


  ! Given a model and mixed real and integer inputs, embed the integer
  !  inputs into their appropriate real-value-only formats.
  ! 
  ! TODO: Make this convert negative integers in AXI into pairs.
  SUBROUTINE EMBED(CONFIG, MODEL, AXI, XI, AX, X)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(IN),  DIMENSION(:) :: MODEL
    INTEGER,       INTENT(IN),  DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN),  DIMENSION(:,:) :: XI
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
            XI(:,:), X(CONFIG%MDN+1:CONFIG%MDN+CONFIG%MDE,:))
    END IF
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WEMB = CONFIG%WEMB + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CEMB = CONFIG%CEMB + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)

  CONTAINS
    ! Given integer inputs and embedding vectors, put embeddings in
    !  place of integer inputs inside of a real matrix.
    SUBROUTINE UNPACK_EMBEDDINGS(MDE, MNE, EMBEDDINGS, INT_INPUTS, EMBEDDED)
      INTEGER, INTENT(IN) :: MDE, MNE
      REAL(KIND=RT), INTENT(IN), DIMENSION(MDE, MNE) :: EMBEDDINGS
      INTEGER, INTENT(IN), DIMENSION(:,:) :: INT_INPUTS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: EMBEDDED
      INTEGER :: N, D, E
      REAL(KIND=RT) :: RD
      RD = REAL(SIZE(INT_INPUTS,1,INT64),RT)
      ! Add together appropriate embedding vectors based on integer inputs.
      EMBEDDED(:,:) = 0.0_RT
      !$OMP PARALLEL DO NUM_THREADS(CONFIG%NUM_THREADS) PRIVATE(N, D, E)
      DO N = 1, SIZE(INT_INPUTS,2,KIND=INT64)
         DO D = 1, SIZE(INT_INPUTS,1,KIND=INT64)
            E = INT_INPUTS(D,N)
            IF (E .GT. 0) THEN
               EMBEDDED(:,N) = EMBEDDED(:,N) + EMBEDDINGS(:,E)
            ELSE IF (E .LT. 0) THEN
               EMBEDDED(:,N) = EMBEDDED(:,N) &
                    + EMBEDDINGS(:, 1 + (-E-1) / MNE) &   ! 111222333
                    - EMBEDDINGS(:, 1 + MOD((-E-1), MNE)) ! 123123123
            END IF
         END DO
         IF (SIZE(INT_INPUTS,1,KIND=INT64) > 1) THEN
            EMBEDDED(:,N) = EMBEDDED(:,N) / RD
         END IF
      END DO
    END SUBROUTINE UNPACK_EMBEDDINGS
  END SUBROUTINE EMBED


  ! Evaluate the piecewise linear regression model, assume already-embedded inputs.
  SUBROUTINE EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, Y, A_STATES, M_STATES, INFO)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), TARGET, INTENT(IN),  DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:) :: AY
    INTEGER,       INTENT(IN),    DIMENSION(:)   :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:,:) :: A_STATES ! SIZE(AX,2), ADS, (ANS|2)
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:,:) :: M_STATES ! SIZE(X, 2), MDS, (MNS|2)
    INTEGER, INTENT(INOUT) :: INFO
    ! Internal values.
    INTEGER(KIND=INT64) :: I, BATCH, BN, BS, BE, BT, GS, GE, NB, NT, E
    REAL(KIND=RT), POINTER, DIMENSION(:) :: X_SHIFT, AX_SHIFT, AY_SHIFT, Y_SHIFT
    REAL(KIND=RT), POINTER, DIMENSION(:,:) :: X_RESCALE, AX_RESCALE, Y_RESCALE
    ! LOCAL ALLOCATION
    INTEGER(KIND=INT64), DIMENSION(:), ALLOCATABLE :: &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS
    ! Timing.
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! If there are no points to evaluate, then immediately return.
    IF (SIZE(Y,2,KIND=INT64) .EQ. 0) RETURN
    ! Set the pointers to the internal normalization values.
    IF (CONFIG%NORMALIZE) THEN
       AX_SHIFT(1:CONFIG%ADN) => MODEL(CONFIG%AISS:CONFIG%AISE)
       AX_RESCALE(1:CONFIG%ADN,1:CONFIG%ADN) => MODEL(CONFIG%AIMS:CONFIG%AIME)
       X_SHIFT(1:CONFIG%MDN) => MODEL(CONFIG%MISS:CONFIG%MISE)
       X_RESCALE(1:CONFIG%MDN,1:CONFIG%MDN) => MODEL(CONFIG%MIMS:CONFIG%MIME)
       Y_SHIFT(1:CONFIG%DO) => MODEL(CONFIG%MOSS:CONFIG%MOSE)
       Y_RESCALE(1:CONFIG%DO,1:CONFIG%DO) => MODEL(CONFIG%MOMS:CONFIG%MOME)
    END IF
    AY_SHIFT(1:CONFIG%ADO) => MODEL(CONFIG%AOSS:CONFIG%AOSE)
    ! Compute the batch start and end indices.
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.FALSE., INFO=INFO)
    IF (INFO .NE. 0) RETURN
    ! Compute the number of threads based on number of points.
    NT = MIN(SIZE(Y,2,KIND=INT64), CONFIG%NUM_THREADS)
    ! 
    ! Aggregator (set) model evaluation.
    ! 
    IF (CONFIG%ADI .GT. 0) THEN
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       aggregator_evaluation : DO BATCH = 1, SIZE(BATCHA_STARTS, KIND=INT64)
          BS = BATCHA_STARTS(BATCH)
          BE = BATCHA_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .LE. 0) CYCLE aggregator_evaluation
          ! Normalize the data.
          IF ((CONFIG%NORMALIZE) .AND. (CONFIG%ADN .GT. 0)) THEN
             ! Apply shift terms to aggregator inputs.
             DO I = BS, BE
                AX(:CONFIG%ADN,I) = AX(:CONFIG%ADN,I) + AX_SHIFT(:)
             END DO
             ! Remove any NaN or Inf values from the data.
             WHERE (IS_NAN(AX(:CONFIG%ADN,BS:BE)) .OR. (.NOT. IS_FINITE(AX(:CONFIG%ADN,BS:BE))))
                AX(:CONFIG%ADN,BS:BE) = 0.0_RT
             END WHERE
             ! Apply multiplier.
             IF (CONFIG%NEEDS_SCALING) THEN
                AX(:CONFIG%ADN,BS:BE) = MATMUL(TRANSPOSE(AX_RESCALE(:,:)), AX(:CONFIG%ADN,BS:BE))
             END IF
          END IF
          ! Evaluate the aggregator model.
          CALL UNPACKED_EVALUATE(INT(BT), &
               CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADSO, CONFIG%ADO+1, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASIS:CONFIG%AEIS), &
               MODEL(CONFIG%ASSV:CONFIG%AESV), &
               MODEL(CONFIG%ASSS:CONFIG%AESS), &
               MODEL(CONFIG%ASOV:CONFIG%AEOV), &
               AX(:,BS:BE), AY(BS:BE,:), A_STATES(BS:BE,:,:), YTRANS=.TRUE.)
          ! Apply zero-mean shift terms to aggregator model outputs.
          IF (CONFIG%MDO .GT. 0) THEN
             DO I = 1, CONFIG%ADO
                AY(BS:BE,I) = AY(BS:BE,I) + AY_SHIFT(I)
             END DO
          END IF
       END DO aggregator_evaluation
       ! 
       ! Aggregate the output of the set model.
       ! 
       E = CONFIG%MDN+CONFIG%MDE+1 ! <- start of aggregator output
       IF (CONFIG%MDO .GT. 0) THEN
          !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, GS, GE) IF(NT > 1)
          set_aggregation_to_x : DO I = 1, SIZE(Y,2,KIND=INT64)
             IF (SIZES(I) .EQ. 0) THEN
                X(E:,I) = 0.0_RT
             ELSE
                ! Take the mean of all outputs from the aggregator model, store
                !   as input to the model that proceeds this aggregation.
                GS = AGG_STARTS(I)
                GE = AGG_STARTS(I) + SIZES(I)-1
                X(E:,I) = SUM(AY(GS:GE,:CONFIG%ADO), 1) / REAL(SIZES(I),RT) 
             END IF
          END DO set_aggregation_to_x
       ELSE
          ! If there is no model after this, place results directly in Y.
          !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, GS, GE) IF(NT > 1)
          set_aggregation_to_y : DO I = 1, SIZE(Y,2,KIND=INT64)
             IF (SIZES(I) .EQ. 0) THEN
                Y(:,I) = 0.0_RT
             ELSE
                GS = AGG_STARTS(I)
                GE = AGG_STARTS(I) + SIZES(I)-1
                Y(:,I) = SUM(AY(GS:GE,:CONFIG%ADO), 1) / REAL(SIZES(I),RT) 
             END IF
          END DO set_aggregation_to_y
       END IF
    END IF
    ! 
    ! Positional model evaluation.
    ! 
    IF (CONFIG%MDO .GT. 0) THEN
       ! Get the number of batches.
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       model_evaluation : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          ! Update "BS", "BE", and "BT" to coincide with the model.
          BS = BATCHM_STARTS(BATCH)
          BE = BATCHM_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .LE. 0) CYCLE model_evaluation
          ! Apply shift terms to numeric model inputs.
          IF ((CONFIG%NORMALIZE) .AND. (CONFIG%MDN .GT. 0)) THEN
             DO I = BS, BE
                X(:CONFIG%MDN,I) = X(:CONFIG%MDN,I) + X_SHIFT(:)
             END DO
             ! Remove any NaN or Inf values from the data.
             WHERE (IS_NAN(X(:CONFIG%MDN,BS:BE)) .OR. (.NOT. IS_FINITE(X(:CONFIG%MDN,BS:BE))))
                X(:CONFIG%MDN,BS:BE) = 0.0_RT
             END WHERE
             ! Apply multiplier.
             IF (CONFIG%NEEDS_SCALING) THEN
                X(:CONFIG%MDN,BS:BE) = MATMUL(TRANSPOSE(X_RESCALE(:,:)), X(:CONFIG%MDN,BS:BE))
             END IF
          END IF
          ! Run the positional model.
          CALL UNPACKED_EVALUATE(INT(BT), &
               CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDSO, CONFIG%MDO, &
               MODEL(CONFIG%MSIV:CONFIG%MEIV), &
               MODEL(CONFIG%MSIS:CONFIG%MEIS), &
               MODEL(CONFIG%MSSV:CONFIG%MESV), &
               MODEL(CONFIG%MSSS:CONFIG%MESS), &
               MODEL(CONFIG%MSOV:CONFIG%MEOV), &
               X(:,BS:BE), Y(:,BS:BE), M_STATES(BS:BE,:,:), YTRANS=.FALSE.)
       END DO model_evaluation
    END IF
    ! Apply shift terms to final outputs.
    IF (CONFIG%NORMALIZE) THEN
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(I, BS, BE, BT) IF(NT > 1)
       output_normalization : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          ! Update "BS", "BE", and "BT" to coincide with the model.
          BS = BATCHM_STARTS(BATCH)
          BE = BATCHM_ENDS(BATCH)
          BT = BE-BS+1
          IF (BT .LE. 0) CYCLE output_normalization
          ! Apply scaling multiplier.
          IF (CONFIG%NEEDS_SCALING) THEN
             Y(:,BS:BE) = MATMUL(TRANSPOSE(Y_RESCALE(:,:)), Y(:,BS:BE))
          END IF
          ! Apply shift.
          DO I = BS, BE
             Y(:,I) = Y(:,I) - Y_SHIFT(:)
          END DO
       END DO output_normalization
    END IF
    ! Deallocate batch sizes.
    DEALLOCATE(BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS)
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WEVL = CONFIG%WEVL + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CEVL = CONFIG%CEVL + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)

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


  ! Compute the gradient with respect to embeddings given the input
  !  gradient by aggregating over the repeated occurrences of the embedding.
  SUBROUTINE EMBEDDING_GRADIENT(MDE, MNE, INT_INPUTS, GRAD, &
       EMBEDDING_GRAD, TEMP_GRAD, COUNTS)
    INTEGER, INTENT(IN) :: MDE, MNE
    INTEGER, INTENT(IN), DIMENSION(:,:) :: INT_INPUTS
    REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: GRAD
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(MDE,MNE) :: EMBEDDING_GRAD
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: TEMP_GRAD
    REAL(KIND=RT), DIMENSION(:) :: COUNTS
    ! Local variables.
    INTEGER :: N, D, E, E1, E2
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
          ELSE IF (E .LT. 0) THEN
             E1 = 1 + (-E-1) / MNE     ! 111222333
             E2 = 1 + MOD((-E-1), MNE) ! 123123123
             COUNTS(E1) = COUNTS(E1) + 1.0_RT
             COUNTS(E2) = COUNTS(E2) + 1.0_RT
             TEMP_GRAD(:,E1) = TEMP_GRAD(:,E1) + GRAD(:,N) / RD
             TEMP_GRAD(:,E2) = TEMP_GRAD(:,E2) - GRAD(:,N) / RD
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


  ! Given the values at all internal states in the model and an output
  !  gradient, propogate the output gradient through the model and
  !  return the gradient of all basis functions.
  SUBROUTINE BASIS_GRADIENT(CONFIG, MODEL, Y, X, AX, SIZES, &
       M_STATES, A_STATES, AY, GRAD, NT, BATCHA_STARTS, &
       BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS)
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: M_STATES ! SIZE(X, 2), MDS, MNS+1
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_STATES ! SIZE(AX,2), ADS, ANS+1
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY ! SIZE(AX,2), ADO+1
    REAL(KIND=RT), INTENT(OUT),   DIMENSION(:,:) :: GRAD ! SIZE(MODEL), NUM_THREADS
    INTEGER, INTENT(IN) :: NT
    INTEGER(KIND=INT64), DIMENSION(:), INTENT(IN) :: &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS
    ! Set the dimension of the X gradient that should be calculated.
    REAL(KIND=RT) :: YSUM
    INTEGER(KIND=INT64) :: I, J, GS, GE, XDG, BATCH, TN
    ! Propogate the gradient through the positional model.
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
          CALL UNPACKED_BASIS_GRADIENT( CONFIG, Y(:,GS:GE), M_STATES(GS:GE,:,:), X(:,GS:GE), &
               CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDSO, CONFIG%MDO, INT(XDG), &
               MODEL(CONFIG%MSIV:CONFIG%MEIV), &
               MODEL(CONFIG%MSIS:CONFIG%MEIS), &
               MODEL(CONFIG%MSSV:CONFIG%MESV), &
               MODEL(CONFIG%MSSS:CONFIG%MESS), &
               MODEL(CONFIG%MSOV:CONFIG%MEOV), &
               GRAD(CONFIG%MSIV:CONFIG%MEIV,TN), &
               GRAD(CONFIG%MSIS:CONFIG%MEIS,TN), &
               GRAD(CONFIG%MSSV:CONFIG%MESV,TN), &
               GRAD(CONFIG%MSSS:CONFIG%MESS,TN), &
               GRAD(CONFIG%MSOV:CONFIG%MEOV,TN), &
               YTRANS=.FALSE.) ! Y is in COLUMN vector format.
       END DO
    END IF
    ! Propogate the gradient from X into the aggregate outputs.
    IF (CONFIG%ADI .GT. 0) THEN
       ! Propogate gradient from the input to the positional model.
       IF (CONFIG%MDO .GT. 0) THEN
          XDG = SIZE(X,1) - CONFIG%ADO + 1  ! <- the first X column for aggregated values
          !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(GS, GE, I, J, YSUM) IF(NT > 1)
          DO I = 1, SIZE(SIZES, KIND=INT64)
             GS = AGG_STARTS(I)
             GE = AGG_STARTS(I) + SIZES(I)-1
             YSUM = SUM(ABS(Y(:,I)))
             DO J = GS, GE
                AY(J,:CONFIG%ADO) = X(XDG:,I) / REAL(SIZES(I),RT)
                ! Compute the gradient for the last column of AY to be sum
                !  of componentwise squared loss values for all outputs.
                AY(J,CONFIG%ADO+1) = AY(J,CONFIG%ADO+1) - YSUM
             END DO
          END DO
       ! Propogate gradient direction from the aggregate output.
       ELSE
          !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(GS, GE, I, J, YSUM) IF(NT > 1)
          DO I = 1, SIZE(SIZES, KIND=INT64)
             GS = AGG_STARTS(I)
             GE = AGG_STARTS(I) + SIZES(I)-1
             YSUM = SUM(ABS(Y(:,I)))
             DO J = GS, GE
                AY(J,:CONFIG%ADO) = Y(:,I) / REAL(SIZES(I),RT)
                ! Compute the gradient for the last column of AY to be sum
                !  of componentwise squared loss values for all outputs.
                AY(J,CONFIG%ADO+1) = AY(J,CONFIG%ADO+1) - YSUM
             END DO
          END DO
       END IF
       !$OMP PARALLEL DO NUM_THREADS(NT) PRIVATE(GS, GE, TN) IF(NT > 1)
       DO BATCH = 1, SIZE(BATCHA_STARTS, KIND=INT64)
          GS = BATCHA_STARTS(BATCH)
          GE = BATCHA_ENDS(BATCH)
          IF (GS .GT. GE) CYCLE
          TN = OMP_GET_THREAD_NUM() + 1
          ! Do the backward gradient calculation assuming "AY" contains output gradient.
          CALL UNPACKED_BASIS_GRADIENT( CONFIG, AY(GS:GE,:), A_STATES(GS:GE,:,:), AX(:,GS:GE), &
               CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADSO, CONFIG%ADO+1, CONFIG%ADE, &
               MODEL(CONFIG%ASIV:CONFIG%AEIV), &
               MODEL(CONFIG%ASIS:CONFIG%AEIS), &
               MODEL(CONFIG%ASSV:CONFIG%AESV), &
               MODEL(CONFIG%ASSS:CONFIG%AESS), &
               MODEL(CONFIG%ASOV:CONFIG%AEOV), &
               GRAD(CONFIG%ASIV:CONFIG%AEIV,TN), &
               GRAD(CONFIG%ASIS:CONFIG%AEIS,TN), &
               GRAD(CONFIG%ASSV:CONFIG%AESV,TN), &
               GRAD(CONFIG%ASSS:CONFIG%AESS,TN), &
               GRAD(CONFIG%ASOV:CONFIG%AEOV,TN), &
               YTRANS=.TRUE.) ! AY is in ROW vector format.
       END DO
    END IF

  CONTAINS

    ! Compute the model gradient.
    SUBROUTINE UNPACKED_BASIS_GRADIENT( CONFIG, Y, STATES, X, &
         MDI, MDS, MNS, MDSO, MDO, MDE, &
         INPUT_VECS, INPUT_SHIFT, &
         STATE_VECS, STATE_SHIFT, OUTPUT_VECS, &
         INPUT_VECS_GRADIENT, INPUT_SHIFT_GRADIENT, &
         STATE_VECS_GRADIENT, STATE_SHIFT_GRADIENT, &
         OUTPUT_VECS_GRADIENT, YTRANS )
    TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
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


  ! Compute the gradient of the sum of squared error of this regression
  ! model with respect to its variables given input and output pairs.
  SUBROUTINE MODEL_GRADIENT(CONFIG, MODEL, &
       AX, AXI, SIZES, X, XI, Y, YW, &
       SUM_SQUARED_GRADIENT, MODEL_GRAD, INFO, &
       AY_GRADIENT, Y_GRADIENT, A_GRADS, M_GRADS, &
       A_EMB_TEMP, M_EMB_TEMP, A_EMB_COUNTS, M_EMB_COUNTS)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: AXI
    INTEGER,       INTENT(IN),    DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: YW
    ! Sum (over all data) squared error (summed over dimensions).
    REAL(KIND=RT), INTENT(INOUT) :: SUM_SQUARED_GRADIENT
    ! Gradient of the model variables.
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: MODEL_GRAD
    ! Output and optional inputs.
    INTEGER, INTENT(INOUT) :: INFO
    ! Work space.
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AY_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: M_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_EMB_TEMP
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: M_EMB_TEMP
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_EMB_COUNTS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: M_EMB_COUNTS
    INTEGER(KIND=INT64) :: L, D, NT, TN, BATCH, SS, SE, MS, ME
    ! LOCAL ALLOCATION
    INTEGER(KIND=INT64), DIMENSION(:), ALLOCATABLE :: BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS
    ! Timing.
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Compute the batch start and end indices.
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.FALSE., INFO=INFO)
    IF (INFO .NE. 0) RETURN
    ! Compute the number of threads based on number of points.
    NT = MIN(SIZE(Y,2,KIND=INT64), CONFIG%NUM_THREADS)
    ! Set gradients to zero initially.
    MODEL_GRAD(:,:) = 0.0_RT
    SUM_SQUARED_GRADIENT = 0.0_RT
    !$OMP PARALLEL DO NUM_THREADS(NT) &
    !$OMP& FIRSTPRIVATE(MS, ME, D) &
    !$OMP& REDUCTION(+: SUM_SQUARED_GRADIENT) IF(NT > 1)
    error_gradient : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
       ! Set batch start and end indices. Exit early if there is no data.
       MS = BATCHM_STARTS(BATCH)
       ME = BATCHM_ENDS(BATCH)
       IF (MS .GE. ME) CYCLE
       ! Compute the gradient of the model outputs, overwriting "Y_GRADIENT"
       Y_GRADIENT(:,MS:ME) = Y_GRADIENT(:,MS:ME) - Y(:,MS:ME) ! squared error gradient
       ! Apply weights to the computed gradients (if they were provided.
       IF (SIZE(YW,1) .EQ. SIZE(Y,1)) THEN
          ! Handle 1 weight per component of each point.
          WHERE (YW(:,MS:ME) .GT. 0.0_RT)
             Y_GRADIENT(:,MS:ME) = Y_GRADIENT(:,MS:ME) * YW(:,MS:ME)
          ELSEWHERE (YW(:,MS:ME) .LT. 0.0_RT)
             ! TODO: Check to see if this is even useful, because it is "costly".
             Y_GRADIENT(:,MS:ME) = YW(:,MS:ME) * SIGN( &
                  (1.0_RT / (1.0_RT + ABS(Y_GRADIENT(:,MS:ME)))), &
                  Y_GRADIENT(:,MS:ME))
          END WHERE
       ELSE IF (SIZE(YW,1) .EQ. 1) THEN
          ! Handle 1 weight per point.
          DO D = 1, SIZE(Y,1)
             WHERE (YW(1,MS:ME) .GT. 0.0_RT)
                Y_GRADIENT(D,MS:ME) = Y_GRADIENT(D,MS:ME) * YW(1,MS:ME)
             ELSEWHERE (YW(1,MS:ME) .LT. 0.0_RT)
             ! TODO: Check to see if this is even useful, because it is "costly".
                Y_GRADIENT(D,MS:ME) = YW(1,MS:ME) * SIGN( &
                     (1.0_RT / (1.0_RT + ABS(Y_GRADIENT(D,MS:ME)))), &
                     Y_GRADIENT(D,MS:ME))
             END WHERE
          END DO
       END IF
       ! Compute the total squared gradient.
       SUM_SQUARED_GRADIENT = SUM_SQUARED_GRADIENT + SUM(Y_GRADIENT(:,MS:ME)**2)
    END DO error_gradient
    ! Adjust the batches to be defined based on inputs (aggregate sets kept together).
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.TRUE., INFO=INFO)
    ! Compute the gradient with respect to the model basis functions (does its own parallelism).
    CALL BASIS_GRADIENT(CONFIG, MODEL, Y_GRADIENT(:,:), X(:,:), AX(:,:), &
         SIZES(:), M_GRADS(:,:,:), A_GRADS(:,:,:), AY_GRADIENT(:,:), &
         MODEL_GRAD(:,:), INT(NT), BATCHA_STARTS(:), BATCHA_ENDS(:), AGG_STARTS(:), &
         BATCHM_STARTS(:), BATCHM_ENDS(:))
    ! Readjust the batches back to being equally dispersed.
    CALL COMPUTE_BATCHES(CONFIG, SIZE(AX,2,KIND=INT64), SIZE(X,2,KIND=INT64), SIZES, &
         BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS, &
         JOINT=.FALSE., INFO=INFO)
    IF (CONFIG%MDE .GT. 0) THEN
       !$OMP PARALLEL DO NUM_THREADS(NT) &
       !$OMP& PRIVATE(MS, ME, TN) IF(NT > 1)
       m_embeddings_gradient : DO BATCH = 1, SIZE(BATCHM_STARTS, KIND=INT64)
          ! Set batch start and end indices. Exit early if there is no data.
          MS = BATCHM_STARTS(BATCH)
          ME = BATCHM_ENDS(BATCH)
          IF (MS .GT. ME) CYCLE
          TN = OMP_GET_THREAD_NUM() + 1
          ! Convert the computed input gradients into average gradients for each embedding.
          CALL EMBEDDING_GRADIENT(CONFIG%MDE, CONFIG%MNE, &
               XI(:,MS:ME), X(CONFIG%MDI-CONFIG%ADO-CONFIG%MDE+1:CONFIG%MDI-CONFIG%ADO,MS:ME), &
               MODEL_GRAD(CONFIG%MSEV:CONFIG%MEEV,TN), M_EMB_TEMP(:,:,TN), M_EMB_COUNTS(:,TN))
       END DO m_embeddings_gradient
    END IF
    IF (CONFIG%ADE .GT. 0) THEN
       !$OMP PARALLEL DO NUM_THREADS(NT) &
       !$OMP& PRIVATE(SS, SE, TN) IF(NT > 1)
       a_embeddings_gradient : DO BATCH = 1, SIZE(BATCHA_STARTS, KIND=INT64)
          SS = BATCHA_STARTS(BATCH)
          SE = BATCHA_ENDS(BATCH)
          IF (SS .GT. SE) CYCLE
          TN = OMP_GET_THREAD_NUM() + 1          
          CALL EMBEDDING_GRADIENT(CONFIG%ADE, CONFIG%ANE, &
               AXI(:,SS:SE), AX(CONFIG%ADI-CONFIG%ADE+1:CONFIG%ADI,SS:SE), &
               MODEL_GRAD(CONFIG%ASEV:CONFIG%AEEV,TN), A_EMB_TEMP(:,:,TN), A_EMB_COUNTS(:,TN))
       END DO a_embeddings_gradient
    END IF
    ! Free memory devoted to batces.
    DEALLOCATE(BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS)
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WGRD = CONFIG%WGRD + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CGRD = CONFIG%CGRD + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)
  END SUBROUTINE MODEL_GRADIENT

  
  ! Make inputs and outputs radially symmetric (to make initialization
  !  more well spaced and lower the curvature of the error gradient).
  SUBROUTINE NORMALIZE_DATA(CONFIG, MODEL, &
       AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YW_IN, &
       AX, AXI, SIZES, X, XI, Y, YW, &
       AX_SHIFT, AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_SHIFT, &
       X_SHIFT, X_RESCALE, XI_SHIFT, XI_RESCALE, Y_SHIFT, Y_RESCALE, &
       A_EMB_VECS, M_EMB_VECS, A_OUT_VECS, A_STATES, AY, INFO)
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX_IN
    INTEGER, INTENT(IN), DIMENSION(:,:) :: AXI_IN
    INTEGER, INTENT(IN), DIMENSION(:) :: SIZES_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X_IN
    INTEGER, INTENT(IN), DIMENSION(:,:) :: XI_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX
    INTEGER, INTENT(INOUT), DIMENSION(:,:) :: AXI
    INTEGER, INTENT(INOUT), DIMENSION(:) :: SIZES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER, INTENT(INOUT), DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AX_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AXI_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AXI_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: AY_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: X_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: XI_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: XI_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: Y_SHIFT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_RESCALE
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_EMB_VECS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: M_EMB_VECS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_OUT_VECS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:,:) :: A_STATES
    REAL(KIND=RT), INTENT(OUT), DIMENSION(:,:) :: AY
    INTEGER, INTENT(INOUT) :: INFO
    ! WARNING: Local variables for processing aggreagte values (and checking outputs).
    !          These will be sized as large as reasonably possible given memory limits.
    LOGICAL, ALLOCATABLE, DIMENSION(:,:) :: YW_MASK ! LOCAL ALLOCATION
    REAL(KIND=RT), ALLOCATABLE, DIMENSION(:,:) :: Y_SCALE ! LOCAL ALLOCATION
    INTEGER(KIND=INT64) :: D, E, NA
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    ! Allocate local variables.
    ALLOCATE( &
         YW_MASK(SIZE(YW,1), SIZE(YW,2)), &
         Y_SCALE(SIZE(Y_RESCALE,1), SIZE(Y_RESCALE,2)) &
    )
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Get some data to use in the normalization process.
    CALL FETCH_DATA(CONFIG, MODEL, &
         AX_IN, AX, AXI_IN, AXI, SIZES_IN, SIZES, &
         X_IN, X, XI_IN, XI, Y_IN, Y, YW_IN, YW, NA )
    ! Encode embeddings if the are provided.
    IF ((CONFIG%MDE + CONFIG%ADE .GT. 0) .AND. (&
         (.NOT. CONFIG%XI_NORMALIZED) .OR. (.NOT. CONFIG%AXI_NORMALIZED))) THEN
       CALL EMBED(CONFIG, MODEL, AXI, XI, AX, X)
    END IF
    ! AX
    IF ((.NOT. CONFIG%AX_NORMALIZED) .AND. (CONFIG%ADN .GT. 0)) THEN
       CALL RADIALIZE(AX(:CONFIG%ADN,:NA), AX_SHIFT(:), AX_RESCALE(:,:), &
            FLATTEN=LOGICAL(CONFIG%RESCALE_AX), MAXBOUND=.TRUE.)
       !$OMP PARALLEL DO
       DO D = 1, CONFIG%ADN
          AX_IN(D,:) = AX_IN(D,:) + AX_SHIFT(D)
       END DO
       !$OMP PARALLEL DO
       DO D = 1, SIZE(AX_IN,2,INT64)
          AX_IN(:,D) = MATMUL(AX_IN(:,D), AX_RESCALE(:,:))
       END DO
       CONFIG%AX_NORMALIZED = .TRUE.
    ELSE IF (CONFIG%ADN .GT. 0) THEN
       AX_SHIFT(:) = 0.0_RT
       AX_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:CONFIG%ADN) AX_RESCALE(D,D) = 1.0_RT
    END IF
    ! AXI
    IF ((.NOT. CONFIG%AXI_NORMALIZED) .AND. (CONFIG%ADE .GT. 0)) THEN
       CALL RADIALIZE(AX(CONFIG%ADN+1:CONFIG%ADN+CONFIG%ADE,:NA), AXI_SHIFT(:), AXI_RESCALE(:,:))
       ! Apply the shift to the source embeddings.
       !$OMP PARALLEL DO
       DO D = 1, CONFIG%ADE
          A_EMB_VECS(D,:) = A_EMB_VECS(D,:) + AXI_SHIFT(D)
       END DO
       ! Apply the transformation to the source embeddings.
       !$OMP PARALLEL DO
       DO D = 1, SIZE(A_EMB_VECS,2,INT64)
          A_EMB_VECS(:,D) = MATMUL(A_EMB_VECS(:,D), AXI_RESCALE(:,:))
       END DO
       CONFIG%AXI_NORMALIZED = .TRUE.
    END IF
    ! X
    IF ((.NOT. CONFIG%X_NORMALIZED) .AND. (CONFIG%MDN .GT. 0)) THEN
       CALL RADIALIZE(X(:CONFIG%MDN,:), X_SHIFT(:), X_RESCALE(:,:), &
            FLATTEN=LOGICAL(CONFIG%RESCALE_X), MAXBOUND=.TRUE.)
       !$OMP PARALLEL DO
       DO D = 1, CONFIG%MDN
          X_IN(D,:) = X_IN(D,:) + X_SHIFT(D)
       END DO
       !$OMP PARALLEL DO
       DO D = 1, SIZE(X_IN,2,INT64)
          X_IN(:,D) = MATMUL(X_IN(:,D), X_RESCALE(:,:))
       END DO
       CONFIG%X_NORMALIZED = .TRUE.
    ELSE IF (CONFIG%MDN .GT. 0) THEN
       X_SHIFT(:) = 0.0_RT
       X_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:CONFIG%MDN) X_RESCALE(D,D) = 1.0_RT
    END IF
    ! XI
    IF ((.NOT. CONFIG%XI_NORMALIZED) .AND. (CONFIG%MDE .GT. 0)) THEN
       CALL RADIALIZE(X(CONFIG%MDN+1:CONFIG%MDN+CONFIG%MDE,:), XI_SHIFT(:), XI_RESCALE(:,:))
       ! Apply the shift to the source embeddings.
       !$OMP PARALLEL DO
       DO D = 1, CONFIG%MDE
          M_EMB_VECS(D,:) = M_EMB_VECS(D,:) + XI_SHIFT(D)
       END DO
       ! Apply the transformation to the source embeddings.
       !$OMP PARALLEL DO
       DO D = 1, SIZE(M_EMB_VECS,2,INT64)
          M_EMB_VECS(:,D) = MATMUL(M_EMB_VECS(:,D), XI_RESCALE(:,:))
       END DO
       CONFIG%XI_NORMALIZED = .TRUE.
    END IF
    ! Y
    IF (.NOT. CONFIG%Y_NORMALIZED) THEN
       CALL RADIALIZE(Y(:,:), Y_SHIFT(:), Y_SCALE(:,:), &
            INVERSE=Y_RESCALE(:,:), FLATTEN=LOGICAL(CONFIG%RESCALE_Y))
       !$OMP PARALLEL DO
       DO D = 1, CONFIG%DO
          Y_IN(D,:) = Y_IN(D,:) + Y_SHIFT(D)
       END DO
       !$OMP PARALLEL DO
       DO D = 1, SIZE(Y_IN,2,INT64)
          Y_IN(:,D) = MATMUL(Y_IN(:,D), Y_SCALE(:,:))
       END DO
       CONFIG%Y_NORMALIZED = .TRUE.
    ELSE
       Y_SHIFT(:) = 0.0_RT
       Y_RESCALE(:,:) = 0.0_RT
       FORALL (D=1:SIZE(Y,1)) Y_RESCALE(D,D) = 1.0_RT
    END IF
    ! YW
    IF (SIZE(YW_IN) .GT. 0) THEN
       ! Divide by the average YW to make its mean 1 (separately for negative and positive YW).
       YW_MASK(:,:) = (YW_IN .GE. 0.0_RT)
       ! TODO: Use YW instead of directly operating on YW_IN (to conserve memory).
       WHERE (YW_MASK(:,:))
          ! TODO: Handle when the COUNT is 0 correctly.
          YW_IN(:,:) = YW_IN(:,:) / (SUM(YW_IN(:,:), MASK=YW_MASK(:,:)) / REAL(COUNT(YW_MASK(:,:)),RT))
       ELSEWHERE
          ! TODO: Handle when the COUNT is 0 correctly.
          YW_IN(:,:) = YW_IN(:,:) / (SUM(YW_IN(:,:), MASK=.NOT. YW_MASK(:,:)) / REAL(COUNT(.NOT. YW_MASK(:,:)),RT))
       END WHERE
    END IF
    ! 
    ! Normalize AY (AX must already be normalized, EVALUATE contains parallelization).
    IF ((.NOT. CONFIG%AY_NORMALIZED) .AND. (CONFIG%ADO .GT. 0)) THEN
       AY_SHIFT(:) = 0.0_RT
       ! Only apply the normalization to AY if there is a model afterwards.
       IF (CONFIG%MDO .GT. 0) THEN
          E = CONFIG%MDN + CONFIG%MDE + 1 ! <- beginning of ADO storage in X
          ! Disable "model" evaluation for this forward pass.
          !   (Give "A_STATES" for the "M_STATES" argument, since it will not be used.)
          D = CONFIG%MDO ; CONFIG%MDO = 0
          CALL EVALUATE(CONFIG, MODEL, AX(:,:NA), AY(:NA,:), SIZES(:), X, X(E:,:), &
               A_STATES(:NA,:,:), A_STATES(:NA,:,:), INFO)
          CONFIG%MDO = D
          ! Compute AY shift as the mean of mean-outputs, apply it.
          AY_SHIFT(:) = -SUM(X(E:,:),2) / REAL(SIZE(X,2),RT)
          !$OMP PARALLEL DO
          DO D = 0, CONFIG%ADO-1
             X(E+D,:) = X(E+D,:) + AY_SHIFT(D+1)
          END DO
          ! Compute the AY scale as the standard deviation of mean-outputs.
          X(E:,1) = SUM(X(E:,:)**2,2) / REAL(SIZE(X,2),RT)
          ! This storage space is overwritten on each embedding, so we can use it
          !  here as a scratch space for holding the intermediate normalization values.
          WHERE (X(E:,1) .GT. 0.0_RT)
             X(E:,1) = SQRT(X(E:,1))
          ELSEWHERE
             X(E:,1) = 1.0_RT
          END WHERE
          ! Apply the factor to the output vectors (and the shift values).
          DO D = 1, CONFIG%ADO
             A_OUT_VECS(:,D) = A_OUT_VECS(:,D) / X(E+D-1,1)
             AY_SHIFT(D) = AY_SHIFT(D) / X(E+D-1,1)
          END DO
       END IF
       CONFIG%AY_NORMALIZED = .TRUE.
    END IF
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WNRM = CONFIG%WNRM + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CNRM = CONFIG%CNRM + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)
    ! Deallocate local variables.
    DEALLOCATE(YW_MASK, Y_SCALE)
  END SUBROUTINE NORMALIZE_DATA

  
  ! Performing conditioning related operations on this model 
  !  (ensure that mean squared error is effectively reduced).
  SUBROUTINE CONDITION_MODEL(CONFIG, MODEL, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, &
       AX, AXI, AY, AY_GRADIENT, X, XI, Y, Y_GRADIENT, &
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
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: AXI
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: AY
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: AY_GRADIENT
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: XI
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y
    REAL(KIND=RT), INTENT(IN),    DIMENSION(:,:) :: Y_GRADIENT
    ! Configuration.
    INTEGER(KIND=INT64), INTENT(IN) :: NUM_THREADS, FIT_STEP
    ! States, gradients, lengths, temporary storage, and order (of ranks).
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_STATES, M_STATES
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:,:) :: A_GRADS, M_GRADS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_LENGTHS, M_LENGTHS
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: A_STATE_TEMP, M_STATE_TEMP
    INTEGER, INTENT(INOUT), DIMENSION(:,:) :: A_ORDER, M_ORDER
    INTEGER :: TOTAL_EVAL_RANK, TOTAL_GRAD_RANK
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! 
    ! Maintain a constant max-norm across the magnitue of input and internal vectors.
    ! 
    CALL UNIT_MAX_NORM(CONFIG, INT(NUM_THREADS), &
         MODEL(CONFIG%ASIV:CONFIG%AEIV), & ! A input vecs
         MODEL(CONFIG%ASIS:CONFIG%AEIS), & ! A input shift
         MODEL(CONFIG%ASSV:CONFIG%AESV), & ! A state vecs
         MODEL(CONFIG%ASSS:CONFIG%AESS), & ! A state shift
         MODEL(CONFIG%ASOV:CONFIG%AEOV), & ! A out vecs
         MODEL(CONFIG%AOSS:CONFIG%AOSE), & ! AY shift
         AY(:,:), & ! AY values (to update shift)
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
            CONFIG%ADI, CONFIG%ADS, CONFIG%ANS, CONFIG%ADSO, CONFIG%ADO, INT(NUM_THREADS), &
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
       ! Update for the positional model.
       CALL CHECK_MODEL_RANK( &
            CONFIG%MDI, CONFIG%MDS, CONFIG%MNS, CONFIG%MDSO, CONFIG%MDO, INT(NUM_THREADS), &
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
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WCON = CONFIG%WCON + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CCON = CONFIG%CCON + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)

  CONTAINS

    ! Make max length vector in each weight matrix have unit length.
    SUBROUTINE UNIT_MAX_NORM(CONFIG, NUM_THREADS, &
         A_INPUT_VECS, A_INPUT_SHIFT, A_STATE_VECS, A_STATE_SHIFT, A_OUTPUT_VECS, AY_SHIFT, AY, &
         M_INPUT_VECS, M_INPUT_SHIFT, M_STATE_VECS, M_STATE_SHIFT, M_OUTPUT_VECS)
      TYPE(MODEL_CONFIG), INTENT(IN) :: CONFIG
      INTEGER, INTENT(IN) :: NUM_THREADS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADI, CONFIG%ADS) :: A_INPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS) :: A_INPUT_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS, CONFIG%ADS, MAX(0,CONFIG%ANS-1)) :: A_STATE_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADS, MAX(0,CONFIG%ANS-1)) :: A_STATE_SHIFT
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADSO, CONFIG%ADO+1) :: A_OUTPUT_VECS
      REAL(KIND=RT), INTENT(INOUT), DIMENSION(CONFIG%ADO) :: AY_SHIFT
      REAL(KIND=RT), INTENT(IN), DIMENSION(:,:) :: AY
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
                    - CONFIG%STEP_AY_CHANGE  * (SUM(AY(:,:CONFIG%ADO),1) / REAL(SIZE(AY,1),RT))
            END IF
         END IF
      END DO
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
      !  - Set new shift term such that the sum of the gradient is maximized?
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
      ! 
      ! ! Find the first zero-valued (unused) basis function (after orthogonalization).
      ! FORALL (RANK = 1 :SIZE(ORDER(:))) ORDER(RANK) = RANK
      ! VALUES(:) = -REAL(USAGE,RT)
      ! CALL ARGSORT(VALUES(:), ORDER(:))
      ! DO RANK = 1, SIZE(ORDER(:))
      !    IF (USAGE(ORDER(RANK)) .EQ. 0) EXIT
      ! END DO
      ! IF (RANK .GT. SIZE(ORDER)) RETURN
      ! 
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
  SUBROUTINE FIT_MODEL(CONFIG, MODEL, RWORK, IWORK, &
       AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YW_IN, &
       STEPS, RECORD, SUM_SQUARED_ERROR, INFO)
    ! TODO: Take an output file name (STDERR and STDOUT are handled).
    TYPE(MODEL_CONFIG), INTENT(INOUT) :: CONFIG
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: MODEL
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:) :: RWORK
    INTEGER,       INTENT(INOUT), DIMENSION(:) :: IWORK
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: AX_IN
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: AXI_IN
    INTEGER,       INTENT(IN),    DIMENSION(:) :: SIZES_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: X_IN
    INTEGER,       INTENT(IN),    DIMENSION(:,:) :: XI_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: Y_IN
    REAL(KIND=RT), INTENT(INOUT), DIMENSION(:,:) :: YW_IN
    INTEGER,       INTENT(IN) :: STEPS
    REAL(KIND=RT), INTENT(OUT), DIMENSION(6,STEPS), OPTIONAL :: RECORD
    REAL(KIND=RT), INTENT(OUT) :: SUM_SQUARED_ERROR
    INTEGER,       INTENT(OUT) :: INFO
    ! Local variables.
    !    "backspace" character array for printing to the same line repeatedly
    CHARACTER(LEN=*), PARAMETER :: RESET_LINE = REPEAT(CHAR(8),27)
    !    temporary holders for overwritten CONFIG attributes
    LOGICAL :: NORMALIZE
    INTEGER :: NUM_THREADS
    !    miscellaneous (hard to concisely categorize)
    LOGICAL :: DID_PRINT
    INTEGER(KIND=INT64) :: STEP, BATCH, NS, SS, SE, MIN_TO_UPDATE, D, VS, VE, T, NT
    INTEGER :: TOTAL_RANK, TOTAL_EVAL_RANK, TOTAL_GRAD_RANK
    INTEGER(KIND=INT64) :: CURRENT_TIME, LAST_PRINT_TIME, WAIT_TIME
    REAL(KIND=RT) :: MSE, PREV_MSE, BEST_MSE
    REAL(KIND=RT) :: STEP_MEAN_REMAIN, STEP_CURV_REMAIN
    REAL :: CPU_TIME_START, CPU_TIME_END
    INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
    CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
    CALL CPU_TIME(CPU_TIME_START)
    ! Check for a valid data shape given the model.
    INFO = 0
    ! Check the shape of all inputs (to make sure they match this model).
    CALL CHECK_SHAPE(CONFIG, MODEL, AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, INFO)
    ! Do shape checks on the work space provided.
    IF (SIZE(RWORK,KIND=INT64) .LT. CONFIG%RWORK_SIZE) THEN
       INFO = 13 ! Provided RWORK is not large enough.
    ELSE IF (SIZE(IWORK,KIND=INT64) .LT. CONFIG%IWORK_SIZE) THEN
       INFO = 14 ! Provided IWORK is not large enough.
    ELSE IF ((CONFIG%ADI .GT. 0) .AND. (CONFIG%NA .LT. 1)) THEN
       INFO = 15 ! Aggregate batch size is zero with nonzero expected aggregate input.
    ELSE IF ((CONFIG%MDI .GT. 0) .AND. (CONFIG%NM .LT. 1)) THEN
       INFO = 16 ! Model batch size is zero with nonzero expected model input.
    END IF
    ! Do shape checks on the YW (weights for Y's) provided.
    IF (SIZE(YW_IN,2) .NE. SIZE(Y_IN,2)) THEN
       INFO = 17 ! Bad YW number of points.
    ELSE IF ((SIZE(YW_IN,1) .NE. 0) & ! some weights provided
         .AND. (SIZE(YW_IN,1) .NE. 1) & ! not one weight per point
         .AND. (SIZE(YW_IN,1) .NE. SIZE(Y_IN,1))) THEN ! one weight per output component
       INFO = 18 ! Bad YW dimension (either 1 per point or commensurate with Y).
    ELSE IF (MINVAL(YW_IN(:,:)) .LT. 0.0_RT) THEN
       INFO = 19 ! Bad YW values (negative numbers are not allowed).
    END IF
    IF (INFO .NE. 0) RETURN
    ! Set the local number of threads (in case there are fewer data points than threads).
    NT = MIN(CONFIG%NUM_THREADS, SIZE(Y_IN,2,INT64))
    ! Initialize RWORK to be zeros because it holds lots of gradients.
    RWORK(:) = 0.0_RT
    ! 
    ! TODO: The AX model needs to produce some output that predicts
    !       the error of an approximation that includes that element.
    ! 
    ! TODO: For deciding which points to keep when doing batching:
    !        track the trailing average error CHANGE for all points (E^.5)
    !        track the trailing average error CHANGE for each point (^.5)
    !        keep the largest X% of (point's error) * (-error change)
    !        introduce new points with the trailing average error change
    !       
    ! 
    ! Unpack all of the work storage into the expected shapes.
    CALL UNPACKED_FIT_MODEL(&
         ! Data batch holders.
         IWORK(CONFIG%SAXI : CONFIG%EAXI), & ! AXI
         RWORK(CONFIG%SAXB : CONFIG%EAXB), & ! AX
         IWORK(CONFIG%SSB : CONFIG%ESB), & ! SIZES
         IWORK(CONFIG%SMXI : CONFIG%EMXI), & ! XI
         RWORK(CONFIG%SMXB : CONFIG%EMXB), & ! X
         RWORK(CONFIG%SMYB : CONFIG%EMYB), & ! Y
         ! Model components.
         MODEL(CONFIG%ASIV : CONFIG%AEIV), & ! AGGREGATOR_INPUT_VECS
         MODEL(CONFIG%ASOV : CONFIG%AEOV), & ! AGGREGATOR_OUTPUT_VECS
         MODEL(CONFIG%MSIV : CONFIG%MEIV), & ! MODEL_INPUT_VECS
         MODEL(CONFIG%MSOV : CONFIG%MEOV), & ! MODEL_OUTPUT_VECS
         MODEL(CONFIG%ASEV : CONFIG%AEEV), & ! AGGREGATOR_EMBEDDING_VECS
         MODEL(CONFIG%MSEV : CONFIG%MEEV), & ! MODEL_EMBEDDING_VECS
         ! States and gradients for model optimization.
         RWORK(CONFIG%SMG : CONFIG%EMG), & ! MODEL_GRAD(NUM_VARS,NUM_THREADS)
         RWORK(CONFIG%SMGM : CONFIG%EMGM), & ! MODEL_GRAD_MEAN(NUM_VARS)
         RWORK(CONFIG%SMGC : CONFIG%EMGC), & ! MODEL_GRAD_CURV(NUM_VARS)
         RWORK(CONFIG%SBM : CONFIG%EBM), & ! BEST_MODEL(NUM_VARS)
         ! States and gradients in parellel regions (work space).
         RWORK(CONFIG%SYG : CONFIG%EYG), & ! Y_GRADIENT(MDO,NM)
         RWORK(CONFIG%SMXG : CONFIG%EMXG), & ! M_GRADS(NM,MDS,MNS+1)
         RWORK(CONFIG%SMXS : CONFIG%EMXS), & ! M_STATES(NM,MDS,MNS+1)
         RWORK(CONFIG%SAYG : CONFIG%EAYG), & ! AY_GRADIENT(NA,ADO+1)
         RWORK(CONFIG%SAY : CONFIG%EAY), & ! AY(NA,ADO+1)
         RWORK(CONFIG%SAXG : CONFIG%EAXG), & ! A_GRADS(NA,ADS,ANS+1)
         RWORK(CONFIG%SAXS : CONFIG%EAXS), & ! A_STATES(NA,ADS,ANS+1)
         ! Data scaling and normalization.
         MODEL(CONFIG%AISS:CONFIG%AISE), & ! AX_SHIFT(ADN)
         MODEL(CONFIG%AIMS:CONFIG%AIME), & ! AX_RESCALE(ADN,ADN)
         RWORK(CONFIG%SAXIS : CONFIG%EAXIS), & ! AXI_SHIFT(ADE)
         RWORK(CONFIG%SAXIR : CONFIG%EAXIR), & ! AXI_RESCALE(ADE,ADE)
         MODEL(CONFIG%AOSS : CONFIG%AOSE), & ! AY_SHIFT(ADO)
         MODEL(CONFIG%MISS:CONFIG%MISE), & ! X_SHIFT(MDN)
         MODEL(CONFIG%MIMS:CONFIG%MIME), & ! X_RESCALE(MDN,MDN)
         RWORK(CONFIG%SMXIS : CONFIG%EMXIS), & ! XI_SHIFT(MDE)
         RWORK(CONFIG%SMXIR : CONFIG%EMXIR), & ! XI_RESCALE(MDE,MDE)
         MODEL(CONFIG%MOSS:CONFIG%MOSE), & ! Y_SHIFT(DO)
         MODEL(CONFIG%MOMS:CONFIG%MOME), & ! Y_RESCALE(DO,DO)
         ! Work space for orthogonalization (conditioning).
         RWORK(CONFIG%SAL : CONFIG%EAL), & ! A_LENGTHS
         RWORK(CONFIG%SML : CONFIG%EML), & ! M_LENGTHS
         RWORK(CONFIG%SAST : CONFIG%EAST), & ! A_STATE_TEMP
         RWORK(CONFIG%SMST : CONFIG%EMST), & ! M_STATE_TEMP
         RWORK(CONFIG%SAET : CONFIG%EAET), & ! A_EMB_TEMP
         RWORK(CONFIG%SMET : CONFIG%EMET), & ! M_EMB_TEMP
         RWORK(CONFIG%SAEC : CONFIG%EAEC), & ! A_EMB_COUNTS
         RWORK(CONFIG%SMEC : CONFIG%EMEC), & ! M_EMB_COUNTS
         ! Rank evaluation (when conditioning model).
         IWORK(CONFIG%SAO : CONFIG%EAO), & ! A_ORDER
         IWORK(CONFIG%SMO : CONFIG%EMO) & ! M_ORDER
         )
    ! Record the end of the total time.
    CALL CPU_TIME(CPU_TIME_END)
    CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
    CONFIG%WFIT = CONFIG%WFIT + (WALL_TIME_END - WALL_TIME_START)
    CONFIG%CFIT = CONFIG%CFIT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)

  CONTAINS

    ! Unpack the work arrays into the proper shapes.
    SUBROUTINE UNPACKED_FIT_MODEL(&
         AXI, AX, SIZES, XI, X, Y, &
         A_IN_VECS, A_OUT_VECS, M_IN_VECS, M_OUT_VECS, A_EMB_VECS, M_EMB_VECS, &
         MODEL_GRAD, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, BEST_MODEL, &
         Y_GRADIENT, M_GRADS, M_STATES, AY_GRADIENT, AY, A_GRADS, A_STATES, &
         AX_SHIFT, AX_RESCALE, AXI_SHIFT, AXI_RESCALE, AY_SHIFT, &
         X_SHIFT, X_RESCALE, XI_SHIFT, XI_RESCALE, Y_SHIFT, Y_RESCALE, &
         A_LENGTHS, M_LENGTHS, A_STATE_TEMP, M_STATE_TEMP, &
         A_EMB_TEMP, M_EMB_TEMP, A_EMB_COUNTS, M_EMB_COUNTS, A_ORDER, M_ORDER)
      ! Definition of unpacked work storage.
      INTEGER, DIMENSION(SIZE(AXI_IN,1), CONFIG%NA) :: AXI
      REAL(KIND=RT), DIMENSION(CONFIG%ADI, CONFIG%NA) :: AX
      INTEGER, DIMENSION(CONFIG%NM) :: SIZES
      INTEGER, DIMENSION(SIZE(XI_IN,1), CONFIG%NM) :: XI
      REAL(KIND=RT), DIMENSION(CONFIG%MDI, CONFIG%NM) :: X
      REAL(KIND=RT), DIMENSION(CONFIG%DO, CONFIG%NM) :: Y
      REAL(KIND=RT), DIMENSION(CONFIG%ADI, CONFIG%ADS) :: A_IN_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%ADSO, CONFIG%ADO) :: A_OUT_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDI, CONFIG%MDS) :: M_IN_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDSO, CONFIG%MDO) :: M_OUT_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%ADE, CONFIG%ANE) :: A_EMB_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%MDE, CONFIG%MNE) :: M_EMB_VECS
      REAL(KIND=RT), DIMENSION(CONFIG%NUM_VARS,CONFIG%NUM_THREADS) :: MODEL_GRAD
      REAL(KIND=RT), DIMENSION(CONFIG%NUM_VARS) :: MODEL_GRAD_MEAN, MODEL_GRAD_CURV, BEST_MODEL
      REAL(KIND=RT), DIMENSION(CONFIG%DO, CONFIG%NM) :: Y_GRADIENT
      REAL(KIND=RT), DIMENSION(CONFIG%NM, CONFIG%MDS, CONFIG%MNS+1) :: M_GRADS, M_STATES
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADO+1_INT64) :: AY_GRADIENT, AY
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADS, CONFIG%ANS+1) :: A_GRADS, A_STATES
      REAL(KIND=RT), DIMENSION(CONFIG%ADN) :: AX_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%ADN, CONFIG%ADN) :: AX_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADE) :: AXI_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%ADE, CONFIG%ADE) :: AXI_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADO) :: AY_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%MDN) :: X_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%MDN, CONFIG%MDN) :: X_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%MDE) :: XI_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%MDE, CONFIG%MDE) :: XI_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%DO) :: Y_SHIFT
      REAL(KIND=RT), DIMENSION(CONFIG%DO, CONFIG%DO) :: Y_RESCALE
      REAL(KIND=RT), DIMENSION(CONFIG%ADS, CONFIG%NUM_THREADS) :: A_LENGTHS
      REAL(KIND=RT), DIMENSION(CONFIG%MDS, CONFIG%NUM_THREADS) :: M_LENGTHS
      REAL(KIND=RT), DIMENSION(CONFIG%NA, CONFIG%ADS) :: A_STATE_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%NM, CONFIG%MDS) :: M_STATE_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%ADE, CONFIG%ANE, CONFIG%NUM_THREADS) :: A_EMB_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%MDE, CONFIG%MNE, CONFIG%NUM_THREADS) :: M_EMB_TEMP
      REAL(KIND=RT), DIMENSION(CONFIG%ANE, CONFIG%NUM_THREADS) :: A_EMB_COUNTS
      REAL(KIND=RT), DIMENSION(CONFIG%MNE, CONFIG%NUM_THREADS) :: M_EMB_COUNTS
      INTEGER, DIMENSION(CONFIG%ADS, CONFIG%NUM_THREADS) :: A_ORDER
      INTEGER, DIMENSION(CONFIG%MDS, CONFIG%NUM_THREADS) :: M_ORDER
      INTEGER(KIND=INT64) :: NA, NM
      REAL :: CPU_TIME_START, CPU_TIME_END
      INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
      ! WARNING: Local allocations (because of INT64 type).
      INTEGER(KIND=INT64), DIMENSION(CONFIG%NUM_VARS) :: UPDATE_INDICES ! LOCAL ALLOCATION
      ! WARNING: Locall allocation (because of unknown YW first dimension at fit config).
      REAL(KIND=RT), DIMENSION(SIZE(YW_IN,1), CONFIG%NM) :: YW ! LOCAL ALLOCATION
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
      CONFIG%NUM_TO_UPDATE = MAX(1_INT64, MIN(CONFIG%NUM_TO_UPDATE, CONFIG%NUM_VARS))
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
      NORMALIZE = CONFIG%NORMALIZE
      CONFIG%NORMALIZE = .FALSE.
      ! 
      ! TODO: Set up validation data (separate from training data).
      !       Add swap scratch space to the memory somewhere.
      !       Put the validation data at the end of the array of data.
      !       Make sure the "real" data is at the front.
      !       Whichever set of data is smaller should be generated by well-spacedness.
      ! 
      ! TODO: Apply normalizations separately to the validation data.
      ! 
      ! TODO: Parameter updating policies should be determined by training MSE change.
      !       Model saving and early stopping should be determined by validation MSE.
      ! 
      ! 
      ! Normalize the data (with parallelization enabled for evaluating AY).
      CALL NORMALIZE_DATA(CONFIG, MODEL, &
           AX_IN, AXI_IN, SIZES_IN, X_IN, XI_IN, Y_IN, YW_IN, &
           AX, AXI, SIZES, X, XI, Y, YW, &
           AX_SHIFT, AX_RESCALE, &
           AXI_SHIFT, AXI_RESCALE, &
           AY_SHIFT, &
           X_SHIFT, X_RESCALE, &
           XI_SHIFT, XI_RESCALE, &
           Y_SHIFT, Y_RESCALE, &
           A_EMB_VECS, M_EMB_VECS, &
           A_OUT_VECS, A_STATES, AY, INFO)
      IF (INFO .NE. 0) RETURN
      ! 
      ! TODO: Compute batches once, reuse for all of training.
      !       ! Set the num batches (NT).
      !       NT = MIN(CONFIG%NUM_THREADS, SIZE(Y,2))
      !       CALL COMPUTE_BATCHES(NT, CONFIG%NA, CONFIG%NM, SIZES, BATCHA_STARTS, &
      !            BATCHA_ENDS, AGG_STARTS, BATCHM_STARTS, BATCHM_ENDS, JOINT=.FALSE., INFO=INFO)
      !       IF (INFO .NE. 0) THEN
      !          Y(:,:) = 0.0_RT
      !          RETURN
      !       END IF
      ! 
      ! ----------------------------------------------------------------
      !                    Minimizing mean squared error
      ! 
      ! Iterate, taking steps with the average gradient over all data.
      fit_loop : DO STEP = 1, STEPS
         ! TODO: Consider wrapping the embed, evaluate, model gradient code in
         !       a higher level thread block to include parallelization over
         !       larger scopes. Will have to be done after the batch is constructed.
         ! TODO: The EMBED and MODEL_GRADIENT functions need to accept AXI as input
         !       and they need to map integers above the literal range into the space
         !       of differences of pairs of aggregate embeddings.
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !                 Pack data into the input work space. 
         ! Pack the data into the work space for a single batch forward pass operation.
         ! If pairwise aggregation is enabled, this also computes the appropriate differences.
         ! 
         CALL FETCH_DATA(CONFIG, MODEL, &
              AX_IN, AX, AXI_IN, AXI, SIZES_IN, SIZES, &
              X_IN, X, XI_IN, XI, Y_IN, Y, YW_IN, YW, NA )
         ! 
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !             Evaluate the model at all points, storing states.
         ! Embed all integer inputs into real vector inputs.
         CALL EMBED(CONFIG, MODEL, AXI(:,:NA), XI, AX(:,:NA), X)
         ! Evaluate the model, storing internal states (for gradient calculation).
         ! If we are checking rank, we need to store evaluations and gradients separately.
         IF ((CONFIG%RANK_CHECK_FREQUENCY .GT. 0) .AND. &
              (MOD(STEP-1,CONFIG%RANK_CHECK_FREQUENCY) .EQ. 0)) THEN
            CALL EVALUATE(CONFIG, MODEL, AX(:,:NA), AY(:NA,:), SIZES, &
                 X, Y_GRADIENT, A_STATES, M_STATES, INFO)
            ! Copy the state values into holders for the gradients.
            A_GRADS(:NA,:,:) = A_STATES(:NA,:,:)
            M_GRADS(:,:,:) = M_STATES(:,:,:)
            AY_GRADIENT(:NA,:) = AY(:NA,:)
         ! Here we can reuse the same memory from evaluation for gradient computation.
         ELSE
            CALL EVALUATE(CONFIG, MODEL, AX(:,:NA), AY_GRADIENT(:NA,:), SIZES, &
                 X, Y_GRADIENT, A_GRADS(:NA,:,:), M_GRADS, INFO)
         END IF
         IF (INFO .NE. 0) RETURN
         ! 
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !                       Compute model gradient 
         ! 
         ! Sum the gradient over all data. If a rank check will be
         !  performed then store the states separate from the gradients.
         !  Otherwise, only compute the gradients and reuse that memory space.
         CALL MODEL_GRADIENT(CONFIG, MODEL(:), &
              AX(:,:NA), AXI(:,:NA), SIZES(:), X(:,:), XI(:,:), Y(:,:), YW(:,:), &
              SUM_SQUARED_ERROR, MODEL_GRAD(:,:), INFO, AY_GRADIENT(:,:),  &
              Y_GRADIENT(:,:), A_GRADS(:NA,:,:), M_GRADS(:,:,:), &
              A_EMB_TEMP(:,:,:), M_EMB_TEMP(:,:,:), A_EMB_COUNTS(:,:), M_EMB_COUNTS(:,:))
         IF (INFO .NE. 0) RETURN
         ! 
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !           Update the step factors, early stop if appropaite.
         ! 
         CALL ADJUST_RATES(BEST_MODEL)
         IF (INFO .NE. 0) RETURN
         IF (NS .EQ. HUGE(NS)) EXIT fit_loop
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !              Modify the model variables (take step).
         ! 
         CALL STEP_VARIABLES(MODEL_GRAD(:,1:NT), MODEL_GRAD_MEAN(:), MODEL_GRAD_CURV(:), UPDATE_INDICES(:))
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
              CONFIG%NUM_THREADS, CONFIG%STEPS_TAKEN, & ! Configuration for conditioning.
              A_STATES(:,:,:), M_STATES(:,:,:), & ! State values at basis functions.
              A_GRADS(:,:,:), M_GRADS(:,:,:), & ! Gradient values at basis functions.
              A_LENGTHS(:,:), M_LENGTHS(:,:), & ! Work space for orthogonalization.
              A_STATE_TEMP(:,:), M_STATE_TEMP(:,:), & ! Work space for state values.
              A_ORDER(:,:), M_ORDER(:,:), &
              TOTAL_EVAL_RANK, TOTAL_GRAD_RANK)
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         ! Record statistics about the model and write an update about 
         ! step and convergence to the command line.
         CALL RECORD_STATS(MODEL_GRAD)
      END DO fit_loop
      CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
      CALL CPU_TIME(CPU_TIME_START)
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
      IF (CONFIG%ENCODE_SCALING) THEN
         IF (CONFIG%ADN .GT. 0) THEN
            IF (CONFIG%ANS .GT. 0) THEN
               A_IN_VECS(:CONFIG%ADN,:) = MATMUL(AX_RESCALE(:,:), A_IN_VECS(:CONFIG%ADN,:))
            ELSE
               A_OUT_VECS(:CONFIG%ADN,:) = MATMUL(AX_RESCALE(:,:), A_OUT_VECS(:CONFIG%ADN,:))
            END IF
            AX_RESCALE(:,:) = 0.0_RT
            DO D = 1, SIZE(AX_RESCALE,1)
               AX_RESCALE(D,D) = 1.0_RT
            END DO
         END IF
         IF (CONFIG%MDN .GT. 0) THEN
            IF (CONFIG%MNS .GT. 0) THEN
               M_IN_VECS(:CONFIG%MDN,:) = MATMUL(X_RESCALE(:,:), M_IN_VECS(:CONFIG%MDN,:))
            ELSE
               M_OUT_VECS(:CONFIG%MDN,:) = MATMUL(X_RESCALE(:,:), M_OUT_VECS(:CONFIG%MDN,:))
            END IF
            X_RESCALE(:,:) = 0.0_RT
            DO D = 1, SIZE(X_RESCALE,1)
               X_RESCALE(D,D) = 1.0_RT
            END DO
         END IF
         ! Apply the output rescale to whichever part of the model produces output.
         IF (CONFIG%MDO .GT. 0) THEN
            M_OUT_VECS(:,:) = MATMUL(M_OUT_VECS(:,:), Y_RESCALE(:,:))
         ELSE
            A_OUT_VECS(:,:) = MATMUL(A_OUT_VECS(:,:), Y_RESCALE(:,:))
         END IF
         Y_RESCALE(:,:) = 0.0_RT
         DO D = 1, SIZE(Y_RESCALE,1)
            Y_RESCALE(D,D) = 1.0_RT
         END DO
         ! Store the fact that scaling has already been encoded into the model.
         CONFIG%NEEDS_SCALING = .FALSE.
      END IF
      ! 
      ! Erase the printed message if one was produced.
      IF (DID_PRINT) WRITE (*,'(A)',ADVANCE='NO') RESET_LINE
      ! 
      ! Reset configuration settings that were modified.
      CONFIG%NORMALIZE = NORMALIZE
      CALL CPU_TIME(CPU_TIME_END)
      CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
      CONFIG%WENC = CONFIG%WENC + (WALL_TIME_END - WALL_TIME_START)
      CONFIG%CENC = CONFIG%CENC + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)
    END SUBROUTINE UNPACKED_FIT_MODEL

    
    ! Adjust the rates of the model optimization parameters.
    SUBROUTINE ADJUST_RATES(BEST_MODEL)
      REAL(KIND=RT), DIMENSION(:) :: BEST_MODEL
      REAL :: CPU_TIME_START, CPU_TIME_END
      INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
      CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
      CALL CPU_TIME(CPU_TIME_START)
      ! Convert the sum of squared errors into the mean squared error.
      MSE = SUM_SQUARED_ERROR / REAL(CONFIG%NM * CONFIG%DO, RT) ! RNY * SIZE(Y,1)
      IF (IS_NAN(MSE) .OR. (.NOT. IS_FINITE(MSE))) THEN
         INFO = 20 ! Encountered NaN or Inf mean squared error during training.
         RETURN
      END IF
      ! Adjust exponential sliding windows based on change in error.
      IF (MSE .LE. PREV_MSE) THEN
         CONFIG%STEP_FACTOR = MIN(CONFIG%STEP_FACTOR * CONFIG%FASTER_RATE, CONFIG%MAX_STEP_FACTOR)
         CONFIG%STEP_MEAN_CHANGE = CONFIG%STEP_MEAN_CHANGE * CONFIG%SLOWER_RATE
         STEP_MEAN_REMAIN = 1.0_RT - CONFIG%STEP_MEAN_CHANGE
         CONFIG%STEP_CURV_CHANGE = CONFIG%STEP_CURV_CHANGE * CONFIG%SLOWER_RATE
         STEP_CURV_REMAIN = 1.0_RT - CONFIG%STEP_CURV_CHANGE
         CONFIG%NUM_TO_UPDATE = CONFIG%NUM_TO_UPDATE + &
              INT(CONFIG%UPDATE_RATIO_STEP * REAL(CONFIG%NUM_VARS,RT))
      ELSE
         CONFIG%STEP_FACTOR = CONFIG%STEP_FACTOR * CONFIG%SLOWER_RATE
         CONFIG%STEP_MEAN_CHANGE = CONFIG%STEP_MEAN_CHANGE * CONFIG%FASTER_RATE
         STEP_MEAN_REMAIN = 1.0_RT - CONFIG%STEP_MEAN_CHANGE
         CONFIG%STEP_CURV_CHANGE = CONFIG%STEP_CURV_CHANGE * CONFIG%FASTER_RATE
         STEP_CURV_REMAIN = 1.0_RT - CONFIG%STEP_CURV_CHANGE
         CONFIG%NUM_TO_UPDATE = CONFIG%NUM_TO_UPDATE - &
              INT(CONFIG%UPDATE_RATIO_STEP * REAL(CONFIG%NUM_VARS,RT))
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
         NS = HUGE(NS)
      END IF
      ! Record the end of the total time.
      CALL CPU_TIME(CPU_TIME_END)
      CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
      CONFIG%WRAT = CONFIG%WRAT + (WALL_TIME_END - WALL_TIME_START)
      CONFIG%CRAT = CONFIG%CRAT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)
    END SUBROUTINE ADJUST_RATES

    
    ! Step the model variables.
    SUBROUTINE STEP_VARIABLES(MODEL_GRAD, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, UPDATE_INDICES)
      REAL(KIND=RT), DIMENSION(:,:) :: MODEL_GRAD
      REAL(KIND=RT), DIMENSION(:) :: MODEL_GRAD_MEAN, MODEL_GRAD_CURV
      INTEGER(KIND=INT64), DIMENSION(:) :: UPDATE_INDICES
      INTEGER(KIND=INT64) :: I, NT, NP, MS, S, E
      REAL :: CPU_TIME_START, CPU_TIME_END
      INTEGER(KIND=INT64) :: WALL_TIME_START, WALL_TIME_END
      CALL SYSTEM_CLOCK(WALL_TIME_START, CLOCK_RATE, CLOCK_MAX)
      CALL CPU_TIME(CPU_TIME_START)
      NT = MIN(CONFIG%NUM_VARS, CONFIG%NUM_THREADS)
      NP = MAX(1_INT64, CONFIG%NUM_VARS / NT)
      ! TODO: Parallelize the most expensive parts of the update.
      !       - the argselect for model parameters.
      ! 
      !$OMP PARALLEL DO PRIVATE(S, E)
      DO I = 0, NT-1
         S = 1_INT64+I*NP
         E = MIN((I+1_INT64)*NP, CONFIG%NUM_VARS)
         ! Aggregate over computed batches and compute average gradient.
         MODEL_GRAD(S:E,1) = SUM(MODEL_GRAD(S:E,:),2) / REAL(SIZE(MODEL_GRAD,2),RT)
         MODEL_GRAD_MEAN(S:E) = STEP_MEAN_REMAIN * MODEL_GRAD_MEAN(S:E) &
              + CONFIG%STEP_MEAN_CHANGE * MODEL_GRAD(S:E,1)
         MODEL_GRAD_CURV(S:E) = STEP_CURV_REMAIN * MODEL_GRAD_CURV(S:E) &
              + CONFIG%STEP_CURV_CHANGE * (MODEL_GRAD_MEAN(S:E) - MODEL_GRAD(S:E,1))**2
         MODEL_GRAD_CURV(S:E) = MAX(MODEL_GRAD_CURV(S:E), EPSILON(CONFIG%STEP_FACTOR))
         ! Set the step as the mean direction (over the past few steps).
         MODEL_GRAD(S:E,1) = MODEL_GRAD_MEAN(S:E)
         ! Start scaling by step magnitude by curvature once enough data is collected.
         IF (STEP .GE. CONFIG%MIN_STEPS_TO_STABILITY) THEN
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
         MODEL(UPDATE_INDICES(1:CONFIG%NUM_TO_UPDATE)) = &
              MODEL(UPDATE_INDICES(1:CONFIG%NUM_TO_UPDATE)) &
              - MODEL_GRAD(UPDATE_INDICES(1:CONFIG%NUM_TO_UPDATE),1) &
              * CONFIG%STEP_FACTOR
      END IF
      CONFIG%STEPS_TAKEN = CONFIG%STEPS_TAKEN + 1
      ! Record the end of the total time.
      CALL CPU_TIME(CPU_TIME_END)
      CALL SYSTEM_CLOCK(WALL_TIME_END, CLOCK_RATE, CLOCK_MAX)
      CONFIG%WOPT = CONFIG%WOPT + (WALL_TIME_END - WALL_TIME_START)
      CONFIG%COPT = CONFIG%COPT + INT(REAL(CPU_TIME_END - CPU_TIME_START, RT) * CLOCK_RATE, INT64)
    END SUBROUTINE STEP_VARIABLES


    ! Record various statistics that are currently of interest (for research).
    SUBROUTINE RECORD_STATS(MODEL_GRAD)
      REAL(KIND=RT), DIMENSION(:,:) :: MODEL_GRAD
      IF (PRESENT(RECORD)) THEN
         ! Store the mean squared error at this iteration.
         RECORD(1,STEP) = MSE
         ! Store the current multiplier on the step.
         RECORD(2,STEP) = CONFIG%STEP_FACTOR
         ! Store the norm of the step that was taken (intermittently).
         IF ((CONFIG%LOG_GRAD_NORM_FREQUENCY .GT. 0) .AND. &
              (MOD(CONFIG%STEPS_TAKEN-1,CONFIG%LOG_GRAD_NORM_FREQUENCY) .EQ. 0)) THEN
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
      ! Write the status update to the command line.
      CALL SYSTEM_CLOCK(CURRENT_TIME, CLOCK_RATE, CLOCK_MAX)
      IF (CURRENT_TIME - LAST_PRINT_TIME .GT. WAIT_TIME) THEN
         IF (DID_PRINT) WRITE (*,'(A)',ADVANCE='NO') RESET_LINE
         WRITE (*,'(I6,"  (",F6.4,") [",F6.4,"]")', ADVANCE='NO') STEP, MSE, BEST_MSE
         LAST_PRINT_TIME = CURRENT_TIME
         DID_PRINT = .TRUE.
      END IF
    END SUBROUTINE RECORD_STATS


  END SUBROUTINE FIT_MODEL


END MODULE AXY



!2022-11-27 12:17:55
!
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! ! ! Batching.                                                      !
         ! ! IWORK(CONFIG%SUI : CONFIG%EUI), & ! UPDATE_INDICES(NUM_VARS)     !
         ! ! IWORK(CONFIG%SBAS : CONFIG%EBAS), & ! BATCHA_STARTS(NUM_THREADS) !
         ! ! IWORK(CONFIG%SBAE : CONFIG%EBAE), & ! BATCHA_ENDS(NUM_THREADS)   !
         ! ! IWORK(CONFIG%SAS : CONFIG%EAS), & ! AGG_STARTS(NUM_THREADS)      !
         ! ! IWORK(CONFIG%SBMS : CONFIG%EBMS), & ! BATCHM_STARTS(NUM_THREADS) !
         ! ! IWORK(CONFIG%SBME : CONFIG%EBME), & ! BATCHM_ENDS(NUM_THREADS)   !
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2022-11-27 12:36:32
!
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! ! UPDATE_INDICES, BATCHA_STARTS, BATCHA_ENDS, AGG_STARTS, & !
         ! ! BATCHM_STARTS, BATCHM_ENDS, &                             !
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2022-11-27 12:59:06
!
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     ! INTEGER(KIND=INT64) :: SBAS, EBAS ! BATCHA_STARTS(NUM_THREADS) !
     ! INTEGER(KIND=INT64) :: SBAE, EBAE ! BATCHA_ENDS(NUM_THREADS)   !
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2022-11-27 12:59:08
!
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     ! INTEGER(KIND=INT64) :: SBMS, EBMS ! BATCHM_STARTS(NUM_THREADS) !
     ! INTEGER(KIND=INT64) :: SBME, EBME ! BATCHM_ENDS(NUM_THREADS)   !
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2022-11-27 13:00:22
!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! ! aggregator batch starts                              !
    ! CONFIG%SBAS = 1_INT64 + CONFIG%IWORK_SIZE              !
    ! CONFIG%EBAS = CONFIG%SBAS-1_INT64 + CONFIG%NUM_THREADS !
    ! CONFIG%IWORK_SIZE = CONFIG%EBAS                        !
    ! ! aggregator batch ends                                !
    ! CONFIG%SBAE = 1_INT64 + CONFIG%IWORK_SIZE              !
    ! CONFIG%EBAE = CONFIG%SBAE-1_INT64 + CONFIG%NUM_THREADS !
    ! CONFIG%IWORK_SIZE = CONFIG%EBAE                        !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2022-11-27 13:00:25
!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! ! model batch starts                                   !
    ! CONFIG%SBMS = 1_INT64 + CONFIG%IWORK_SIZE              !
    ! CONFIG%EBMS = CONFIG%SBMS-1_INT64 + CONFIG%NUM_THREADS !
    ! CONFIG%IWORK_SIZE = CONFIG%EBMS                        !
    ! ! model batch ends                                     !
    ! CONFIG%SBME = 1_INT64 + CONFIG%IWORK_SIZE              !
    ! CONFIG%EBME = CONFIG%SBME-1_INT64 + CONFIG%NUM_THREADS !
    ! CONFIG%IWORK_SIZE = CONFIG%EBME                        !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2023-01-25 20:56:36
!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! ! ! ------------------------------------------------------------------          !
    ! ! !                DEVELOPING BATCHED EVALUATION CODE                           !
    ! ! !    measured gradient contribution of all input points                       !
    ! ! REAL(KIND=RT), DIMENSION(SIZE(Y,2)) :: Y_ERROR                                !
    ! ! REAL(KIND=RT), DIMENSION(SIZE(Y,2)) :: Y_REDUCTION                            !
    ! ! INTEGER, DIMENSION(SIZE(Y,2)) :: Y_CONSECUTIVE_STEPS                          !
    ! ! INTEGER, DIMENSION(SIZE(Y,2)) :: Y_UNUSED_STEPS                               !
    ! ! !    count of how many steps have been taken since last usage                 !
    ! ! INTEGER, DIMENSION(SIZE(AX,2)) :: AX_UNUSED_STEPS                             !
    ! ! INTEGER, DIMENSION(SIZE(X,2)) :: X_UNUSED_STEPS                               !
    ! ! !    indices (used for sorting and selecting points for gradient computation) !
    ! ! INTEGER, DIMENSION(SIZE(AX,2)) :: AX_INDICES                                  !
    ! ! INTEGER, DIMENSION(SIZE(X,2)) :: X_INDICES                                    !
    ! ! ! ------------------------------------------------------------------          !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2023-01-25 21:03:01
!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! !       It should be a function of the AX and MX embeddings that !
    ! !       predicts the error of the model. A linear function would !
    ! !       suffice, since the nonlinearities in the aggregator and  !
    ! !       model can capture any shape of this error function.      !
    ! !                                                                !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2023-01-25 21:03:10
!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! !       Then the elements that have lower predicted error can be   !
    ! !       kept, and points that have higher predicted error can be   !
    ! !       filtered out (this is a differentiable feature selection   !
    ! !       method that allows nonlinear feature selection functions). !
    ! !                                                                  !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2023-02-18 12:42:27
!
          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          ! ! Unapply shift terms to aggregator inputs.                                            !
          ! IF ((CONFIG%NORMALIZE) .AND. (CONFIG%ADN .GT. 0)) THEN                                 !
          !    DO I = BS, BE                                                                       !
          !       AX(:CONFIG%ADN,I) = AX(:CONFIG%ADN,I) - AX_SHIFT(:)                              !
          !    END DO                                                                              !
          !    ! TODO: Unapply the multiplier.                                                     !
          !    ! AX(:CONFIG%ADN,BS:BE) = MATMUL(TRANSPOSE(AX_RESCALE(:,:)), AX(:CONFIG%ADN,BS:BE)) !
          ! END IF                                                                                 !
          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!2023-02-18 12:42:37
!
          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          ! ! Unapply the X shifts.                                                             !
          ! IF ((CONFIG%NORMALIZE) .AND. (CONFIG%MDN .GT. 0)) THEN                              !
          !    DO I = BS, BE                                                                    !
          !       X(:CONFIG%MDN,I) = X(:CONFIG%MDN,I) - X_SHIFT(:)                              !
          !    END DO                                                                           !
          !    ! TODO: Unapply the multiplier.                                                  !
          !    ! X(:CONFIG%MDN,BS:BE) = MATMUL(TRANSPOSE(X_RESCALE(:,:)), X(:CONFIG%MDN,BS:BE)) !
          ! END IF                                                                              !
          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
