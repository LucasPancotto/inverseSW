! External forcing of passive/active scalar.
! The scalar can be passive (e.g., in the PHD solver, in which case 
! the forcing represents a source of passive scalar), or active (as 
! in the shallow-water solvers, where the function here represents 
! the bottom topography).
! This file contains the expression used for the external forcing in 
! the equation for the scalar in many solvers. You can use temporary 
! real arrays R1 of size (n,jsta:jend), and temporary complex arrays 
! C1, C2, C12 and C13 of size (n,ista:iend) to do intermediate 
! computations. The variable c0 should control the global amplitude 
! of the forcing, and variables cparam0-9 can be used to control the 
! amplitudes of individual terms. At the end, the result in spectral 
! space should be stored in the array fs.

! Null forcing / flat bottom topography

      DO j = jsta,jend
         DO i = 1,n
            R1(i,j) = sparam0 !ho cparam1
            DO ki = skdn, skup
               R1(i,j) = R1(i,j) + sparam1*COS(2*pi*ki*real(i,kind=GP)/real(n,kind=GP))*COS(2*pi*ki*real(j,kind=GP)/real(n,kind=GP))

            END DO
         END DO
      END DO

      WRITE(ext , fmtext) 0
      CALL io_write(1,odir,'fs',ext,planio,R1)

      CALL fftp2d_real_to_complex(planrc,R1,fs,MPI_COMM_WORLD)


