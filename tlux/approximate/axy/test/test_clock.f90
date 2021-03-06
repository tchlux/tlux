PROGRAM TEST
  USE ISO_FORTRAN_ENV, ONLY: IT => INT64, RT => REAL32
  INTEGER(KIND=IT) :: C1, C2, COUNT_RATE, COUNT_MAX
  REAL :: S1, S2
  INTEGER :: I
  CALL CPU_TIME(S1)
  CALL SYSTEM_CLOCK(C1, COUNT_RATE, COUNT_MAX)
  !$OMP PARALLEL DO NUM_THREADS(10)
  DO I = 1, 10
     CALL SLEEP(1)     
  END DO
  !$OMP END PARALLEL DO
  CALL SYSTEM_CLOCK(C2, COUNT_RATE, COUNT_MAX)
  CALL CPU_TIME(S2)
  PRINT *, 'ELAPSED:', C2 - C1
  PRINT *, 'RATE:   ', COUNT_RATE
  PRINT *, 'SECONDS:', REAL(C2-C1,RT) / REAL(COUNT_RATE,RT)
  PRINT *, 'CPU:    ', S2 - S1

END PROGRAM TEST
