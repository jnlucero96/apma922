module FDiff

! machine eps
real*8, parameter :: float32_eps = 1.1920928955078125e-07
! real*8, parameter :: float64_eps = 2.22044604925031308084726e-16
real*8, parameter :: float64_eps = 8.22044604925031308084726e-12

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

! given a discretized linear operator L,
! get steady state distribution using 10th-order implicit RK Gauss Legendre in time
subroutine get_steady_gl10( &
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

        ! update probabilities using a Gauss-Legendre scheme
        call gl10(n, L, p_now, dt)

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

end subroutine get_steady_gl10

! 10th order implicit Gauss-Legendre integrator
subroutine gl10(n, L, p, dt)
    integer*8, parameter :: s = 5
    integer*8, intent(in) :: n
    real*8, dimension(n,n), intent(in) :: L
    real*8, dimension(n), intent(inout) :: p
    real*8, intent(in) :: dt
    real*8, dimension(:,:), allocatable :: g
    integer*8 i, k

    ! Butcher tableau for 10th order Gauss-Legendre method
    real*8, parameter :: a(s,s) = reshape((/ &
                0.5923172126404727187856601017997934066Q-1, -1.9570364359076037492643214050884060018Q-2, &
                1.1254400818642955552716244215090748773Q-2, -0.5593793660812184876817721964475928216Q-2, &
                1.5881129678659985393652424705934162371Q-3,  1.2815100567004528349616684832951382219Q-1, &
                1.1965716762484161701032287870890954823Q-1, -2.4592114619642200389318251686004016630Q-2, &
                1.0318280670683357408953945056355839486Q-2, -2.7689943987696030442826307588795957613Q-3, &
                1.1377628800422460252874127381536557686Q-1,  2.6000465168064151859240589518757397939Q-1, &
                1.4222222222222222222222222222222222222Q-1, -2.0690316430958284571760137769754882933Q-2, &
                4.6871545238699412283907465445931044619Q-3,  1.2123243692686414680141465111883827708Q-1, &
                2.2899605457899987661169181236146325697Q-1,  3.0903655906408664483376269613044846112Q-1, &
                1.1965716762484161701032287870890954823Q-1, -0.9687563141950739739034827969555140871Q-2, &
                1.1687532956022854521776677788936526508Q-1,  2.4490812891049541889746347938229502468Q-1, &
                2.7319004362580148889172820022935369566Q-1,  2.5888469960875927151328897146870315648Q-1, &
                0.5923172126404727187856601017997934066Q-1/), (/s,s/))
    real*8, parameter ::   b(s) = [ &
                1.1846344252809454375713202035995868132Q-1,  2.3931433524968323402064575741781909646Q-1, &
                2.8444444444444444444444444444444444444Q-1,  2.3931433524968323402064575741781909646Q-1, &
                1.1846344252809454375713202035995868132Q-1]

    allocate(g(n,s))

    ! iterate trial steps
    g = 0.0; do k = 1,16
            g = matmul(g,a)
            do i = 1,s
                    call dgemv('n', n, n, real(1.0,kind=8), L, n, p+g(:,i)*dt, 1, real(0.0,kind=8), g(:,i), 1)
            end do
    end do

    ! update the solution
    p = p + matmul(g,b)*dt
end subroutine gl10

end module FDiff