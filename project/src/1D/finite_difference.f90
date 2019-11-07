module FDiff

! machine eps
real*8, parameter :: float32_eps = 1.1920928955078125e-07
real*8, parameter :: float64_eps = 2.22044604925031308084726e-16

contains

! given a discretized linear operator L,
! get steady state distribution using Forward-Euler in time
subroutine get_steady_ft( &
    n, dt, L, check_step, p_initial, p_final &
    )
    integer*8, intent(in) :: n
    real*8, intent(in) :: dt
    real*8, dimension(n,n), intent(in) :: L
    integer*8, intent(in) :: check_step
    real*8, dimension(n), intent(inout) :: p_initial
    real*8, dimension(n), intent(inout) :: p_final

    real*8, dimension(:), allocatable :: p_now, p_last_ref

    ! continue condition
    logical cc
    integer*8 step_counter ! counting steps
    real*8 tot_var_dist ! total variation distance

    allocate(p_now(n), p_last_ref(n))

    ! initialize based on initial data
    p_now = p_initial


    ! initialize reference array
    p_last_ref = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    do while (cc)

        ! update probabilities using a BLAS routine
        call dgemv('n', n, n, real(1.0,kind=8), dt*L, n, p_now, 1, real(1.0,kind=8), p_now, 1)

        ! bail at the first sign of trouble
        if (abs(sum(p_now) - 1.0) .ge. float32_eps) stop "Normalization broken!"

        if (step_counter .EQ. check_step) then
            tot_var_dist = 0.5*sum(abs(p_last_ref-p_now))

            print *, tot_var_dist

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0
                p_last_ref = p_now
            end if
        end if
        step_counter = step_counter + 1
    end do

    p_final = 0.0; p_final = p_now

end subroutine get_steady_ft

end module FDiff