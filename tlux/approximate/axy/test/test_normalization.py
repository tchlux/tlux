# 
# 
# OpenMP compatibility.
#   FUNCTION OMP_GET_MAX_THREADS()
#   FUNCTION OMP_GET_THREAD_NUM()
# 
# Profiling and development.
#   FUNCTION PROFILE(SUBROUTINE_NAME)
# 
# Model initialization.
#   SUBROUTINE NEW_MODEL_CONFIG(ADN, ADE, ANE, ADS, ANS, ADO, ...
#   SUBROUTINE NEW_FIT_CONFIG(NM, NA, NMT, NAT, ADI, MDI, ODI, SEED, CONFIG)
#   SUBROUTINE INIT_MODEL(CONFIG, MODEL, SEED, INITIAL_SHIFT_RANGE, INITIAL_OUTPUT_SCALE)
#   SUBROUTINE INIT_EMBEDDINGS(DE, NE, EMBEDDINGS, EMBEDDINGS_MEAN, RANDOM_LENGTHS)
#   SUBROUTINE INIT_SUBMODEL(MDI, MDS, MNS, MDSO, MDO, ...
# 
# Correct usage checks.
#   SUBROUTINE CHECK_SHAPE(CONFIG, MODEL, AX, AXI, SIZES, X, XI, Y, YI, INFO, FOR_EVALUATION)
#   SUBROUTINE FIT_CHECK(CONFIG, MODEL, RWORK, IWORK, LWORK, ...
# 
# Fitting model.
# 
#   Data handling.
#     SUBROUTINE COMPUTE_BATCHES(CONFIG, NA, NM, SIZES, BATCHA_STARTS, ...
#     SUBROUTINE FETCH_DATA(CONFIG, AGG_ITERATORS_IN, ...
#     SUBROUTINE EMBED(CONFIG, MODEL, AXI, XI, AX, X)
#       SUBROUTINE UNPACK_EMBEDDINGS(MDE, MNE, EMBEDDINGS, INT_INPUTS, EMBEDDED)
# 
#   Normalization of data.
#     SUBROUTINE NORMALIZE_DATA(CONFIG, MODEL, AGG_ITERATORS, &
#     SUBROUTINE NORMALIZE_INPUTS(CONFIG, MODEL, AX, SIZES, X, INFO)
# 
#   Model evaluation.
#     SUBROUTINE EVALUATE(CONFIG, MODEL, AX, AY, SIZES, X, Y, A_STATES, M_STATES, INFO)
#       SUBROUTINE UNPACKED_EVALUATE(N, MDI, MDS, MNS, MDSO, MDO, INPUT_VECS, &
# 
#   Gradient calculation.
#     SUBROUTINE MODEL_GRADIENT(CONFIG, MODEL, ...
#       SUBROUTINE EMBEDDING_GRADIENT(MDE, MNE, PAIRWISE, INT_INPUTS, GRAD, ...
#       SUBROUTINE BASIS_GRADIENT(CONFIG, MODEL, Y, X, AX, SIZES, ...
#         SUBROUTINE UNPACKED_BASIS_GRADIENT( CONFIG, Y, STATES, X, ...
#       SUBROUTINE OUTPUT_GRADIENT(CONFIG, Y_GRADIENT, Y, YI, YW, O_EMB_VECS, O_EMB_GRAD, ...
# 
#   Normalization of model.
#     SUBROUTINE CONDITION_MODEL(CONFIG, MODEL, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, &
#       SUBROUTINE UNIT_MAX_NORM(CONFIG, NUM_THREADS, &
#       SUBROUTINE CHECK_MODEL_RANK(DI, DS, NS, DSO, DO, NUM_THREADS, X, Y_GRADIENT, &
#       SUBROUTINE REPLACE_BASIS_FUNCTIONS(USAGE, &
# 
#   Orchestration.
#     SUBROUTINE FIT_MODEL(CONFIG, MODEL, RWORK, IWORK, LWORK, ...
#       SUBROUTINE UNPACKED_FIT_MODEL( ...
#       SUBROUTINE ADJUST_RATES(BEST_MODEL, MODEL_GRAD_MEAN, MODEL_GRAD_CURV)
#       SUBROUTINE STEP_VARIABLES(MODEL_GRAD, MODEL_GRAD_MEAN, MODEL_GRAD_CURV, UPDATE_INDICES, NB)
#       SUBROUTINE RECORD_STATS(MODEL_GRAD)
# 
# 


# Scope out where in the AXY.f90 code I will need to support dynamic normalization.
# Create tests where I can check how the gradient changes when the normalization is applied.
# Create data that changes through iterations.


if __name__ == "__main__":

    # # Get all subroutine and function names.
    # import os
    # path = lambda p: os.path.expanduser(p)
    # with open(path("../axy.f90")) as f:
    #     lines = f.read().split("\n")
    # for l in lines:
    #     l = l.strip().upper()
    #     if (l.startswith("SUBROUTINE") or l.startswith("FUNCTION")):
    #         print(l)

    # Check how gradient changes when model variables are multiplied by a number.
    
    pass
