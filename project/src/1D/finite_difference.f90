module FDiff

! machine eps
real*8, parameter :: float32_eps = 1.1920928955078125e-07
! real*8, parameter :: float64_eps = 2.22044604925031308084726e-16
real*8, parameter :: float64_eps = 8.22044604925031308084726e-12

contains

subroutine get_steady_ft_1D(n, dt, check_step, D, dx, drift_at_pos, p_initial, p_final)
    integer*8, intent(in) :: n
    real*8, intent(in) :: dt
    integer*8, intent(in) :: check_step
    real*8, intent(in) :: D
    real*8, intent(in) :: dx
    real*8, dimension(n) :: drift_at_pos
    real*8, dimension(n), intent(in) :: p_initial
    real*8, dimension(n), intent(inout) :: p_final

    real*8, dimension(:), allocatable :: p_now, p_last, p_last_ref, dp

    ! continue condition
    logical cc
    integer*8 step_counter ! counting steps
    real*8 tot_var_dist ! total variation distance

    allocate(p_now(n), p_last(n), p_last_ref(n), dp(n))

    ! initialize based on initial data
    p_last = p_initial

    ! initialize reference array
    p_last_ref = 0.0; p_now = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    do while (cc)

        ! call update_probability_1D(n, dx, p_now, p_last, drift_at_pos, D, dt)
        call spatial_derivs_FD_1D(n, dx, dp, p_last, drift_at_pos, D, dt)

        p_now = p_last + dp
        
        ! bail at the first sign of trouble
        if (abs(sum(p_now) - 1.0) .ge. float32_eps) stop "Normalization broken!"

        if (step_counter .EQ. check_step) then
            tot_var_dist = 0.5*sum(abs(p_last_ref-p_now))

            ! print *, tot_var_dist

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0
                p_last_ref = p_now
            end if
        end if

        ! cycle the variables
        p_last = p_now; p_now = 0.0
        step_counter = step_counter + 1
    end do

    p_final = 0.0; p_final = p_last

end subroutine get_steady_ft_1D

subroutine get_steady_ft_2D(n, dt, check_step, D, dx, drift1_at_pos, drift2_at_pos, p_initial, p_final)
    integer*8, intent(in) :: n
    real*8, intent(in) :: dt
    integer*8, intent(in) :: check_step
    real*8, intent(in) :: D
    real*8, intent(in) :: dx
    real*8, dimension(n,n), intent(in) :: drift1_at_pos, drift2_at_pos
    real*8, dimension(n,n), intent(in) :: p_initial
    real*8, dimension(n,n), intent(inout) :: p_final

    real*8, dimension(:,:), allocatable :: p_now, p_last, p_last_ref, dp

    ! continue condition
    logical cc
    integer*8 step_counter ! counting steps
    real*8 tot_var_dist ! total variation distance

    allocate(p_now(n,n), p_last(n,n), p_last_ref(n,n), dp(n,n))

    ! initialize based on initial data
    p_last = p_initial

    ! initialize reference array
    p_last_ref = 0.0; p_now = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    do while (cc)

        call spatial_derivs_FD_2D(n, dx, dp, p_last, drift1_at_pos, drift2_at_pos, D, dt)

        p_now = p_last + dp

        ! bail at the first sign of trouble
        if (abs(sum(p_now) - 1.0) .ge. float32_eps) stop "Normalization broken!"

        if (step_counter .EQ. check_step) then
            tot_var_dist = 0.5*sum(abs(p_last_ref-p_now))

            ! print *, tot_var_dist

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0
                p_last_ref = p_now
            end if
        end if

        ! cycle the variables
        p_last = p_now; p_now = 0.0
        step_counter = step_counter + 1
    end do

    p_final = 0.0; p_final = p_last

end subroutine get_steady_ft_2D

! get steady state distribution using 10th-order implicit RK Gauss Legendre in time
subroutine get_steady_gl10_1D( &
    n, dt, check_step, D, dx, drift_at_pos, p_initial, p_final &
    )
    integer*8, intent(in) :: n
    real*8, intent(in) :: dt
    integer*8, intent(in) :: check_step
    real*8, intent(in) :: D, dx
    real*8, dimension(n), intent(in) :: drift_at_pos 
    real*8, dimension(n), intent(inout) :: p_initial
    real*8, dimension(n), intent(inout) :: p_final

    real*8, dimension(:), allocatable :: p_now, p_last_ref, dp

    ! continue condition
    logical cc
    integer*8 step_counter ! counting steps
    real*8 tot_var_dist ! total variation distance

    allocate(p_now(n), p_last_ref(n), dp(n))

    ! initialize based on initial data
    p_now = p_initial

    ! initialize reference array
    p_last_ref = 0.0; dp = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    do while (cc)

        ! update probabilities using a Gauss-Legendre scheme
        call gl10_1D(n, dx, drift_at_pos, p_now, dp, D, dt)

        p_now = p_now + dp

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

end subroutine get_steady_gl10_1D

! get steady state distribution using 10th-order implicit RK Gauss Legendre in time
subroutine get_steady_gl10_2D( &
    n, dt, check_step, D, dx, drift1_at_pos, drift2_at_pos, p_initial, p_final &
    )
    integer*8, intent(in) :: n
    real*8, intent(in) :: dt
    integer*8, intent(in) :: check_step
    real*8, intent(in) :: D, dx
    real*8, dimension(n,n), intent(in) :: drift1_at_pos, drift2_at_pos 
    real*8, dimension(n,n), intent(inout) :: p_initial
    real*8, dimension(n,n), intent(inout) :: p_final

    real*8, dimension(:,:), allocatable :: p_now, p_last_ref, dp

    ! continue condition
    logical cc
    integer*8 step_counter ! counting steps
    real*8 tot_var_dist ! total variation distance

    allocate(p_now(n,n), p_last_ref(n,n), dp(n,n))

    ! initialize based on initial data
    p_now = p_initial

    ! initialize reference array
    p_last_ref = 0.0; dp = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    do while (cc)

        ! update probabilities using a Gauss-Legendre scheme
        call gl10_2D(n, dx, drift1_at_pos, drift2_at_pos, p_now, dp, D, dt)

        p_now = p_now + dp

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

end subroutine get_steady_gl10_2D

! 10th order implicit Gauss-Legendre integrator
subroutine gl10_1D(n, dx, drift_at_pos, p, dp, D, dt)
    integer*8, parameter :: s = 5
    integer*8, intent(in) :: n
    real*8, intent(in) :: dx
    real*8, dimension(n), intent(in) :: drift_at_pos 
    real*8, dimension(n), intent(inout) :: p, dp
    real*8, intent(in) :: D, dt
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
                    call spatial_derivs_FD_1D(n, dx, g(:,i), p+g(:,i)*dt, drift_at_pos, D, dt)
            end do
    end do

    ! update the solution
    dp = matmul(g,b)*dt
end subroutine gl10_1D

! 10th order implicit Gauss-Legendre integrator
subroutine gl10_2D(n, dx, drift1_at_pos, drift2_at_pos, p, dp, D, dt)
    integer*8, parameter :: s = 5
    integer*8, intent(in) :: n
    real*8, intent(in) :: dx
    real*8, dimension(n), intent(in) :: drift1_at_pos, drift2_at_pos
    real*8, dimension(n), intent(inout) :: p, dp
    real*8, intent(in) :: D, dt
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
                    call spatial_derivs_FD_2D(n, dx, g(:,i), p+g(:,i)*dt, drift1_at_pos, drift2_at_pos, D, dt)
            end do
    end do

    ! update the solution
    dp = matmul(g,b)*dt
end subroutine gl10_2D

subroutine spatial_derivs_FD_1D(n, dx, ddx, p_last, drift_at_pos, D, dt)
    integer*8, intent(in) :: n
    real*8, intent(in) :: dx
    real*8, dimension(n), intent(inout) :: ddx
    real*8, dimension(n), intent(in) :: p_last
    real*8, dimension(n), intent(in) :: drift_at_pos
    real*8, intent(in) :: D, dt
    integer*8 i ! iterator variable

    !! Periodic boundary conditions:
    !! Explicity update FPE for the end points 
    ddx(1) = D*dt*(-(drift_at_pos(2)*p_last(2)-drift_at_pos(n)*p_last(n))/(2.0*dx) &
        + (p_last(2)-2.0*p_last(1)+p_last(n))/(dx*dx)) 
    ddx(n) = D*dt*(-(drift_at_pos(1)*p_last(1)-drift_at_pos(n-1)*p_last(n-1))/(2.0*dx) &
        + (p_last(1)-2.0*p_last(n)+p_last(n-1))/(dx*dx)) 

    ! iterate through all the coordinates,not on the corners,for both variables
    do i=2,n-1
        ddx(i) = D*dt*(-(drift_at_pos(i+1)*p_last(i+1)-drift_at_pos(i-1)*p_last(i-1))/(2.0*dx) &
            + (p_last(i+1)-2.0*p_last(i)+p_last(i-1))/(dx*dx)) 
    end do
end subroutine spatial_derivs_FD_1D

subroutine spatial_derivs_FD_2D(n, dx, ddx, p_last, drift1_at_pos, drift2_at_pos, D, dt)
    integer*8, intent(in) :: n
    real*8, intent(in) :: dx
    real*8, dimension(n,n), intent(inout) :: ddx
    real*8, dimension(n,n), intent(in) :: p_last
    real*8, dimension(n,n), intent(in) :: drift1_at_pos, drift2_at_pos
    real*8, intent(in) :: D, dt
    integer*8 i,j ! iterator variable

    !! Periodic boundary conditions:
    !! Explicity update FPE for the corners
    ddx(1,1) = D*dt*(-(drift1_at_pos(2,1)*p_last(2,1)-drift1_at_pos(n,1)*p_last(n,1))/(2.0*dx) &
        + (p_last(2,1)-2.0*p_last(1,1)+p_last(n,1))/(dx*dx) &
        - (drift2_at_pos(1,2)*p_last(1,2)-drift2_at_pos(1,n)*p_last(1,n))/(2.0*dx) &
        + (p_last(1,2)-2.0*p_last(1,1)+p_last(1,n))/(dx*dx)) 
    ddx(1,n) = D*dt*(-(drift1_at_pos(2,n)*p_last(2,n)-drift1_at_pos(n,n)*p_last(n,n))/(2.0*dx) &
        + (p_last(2,n)-2.0*p_last(1,n)+p_last(n,n))/(dx*dx) &
        - (drift2_at_pos(1,1)*p_last(1,1)-drift2_at_pos(1,n-1)*p_last(1,n-1))/(2.0*dx) &
        + (p_last(1,1)-2.0*p_last(1,n)+p_last(1,n-1))/(dx*dx)) 
    ddx(n,1) = D*dt*(-(drift1_at_pos(1,1)*p_last(1,1)-drift1_at_pos(n-1,1)*p_last(n-1,1))/(2.0*dx) &
        + (p_last(1,1)-2.0*p_last(n,1)+p_last(n-1,1))/(dx*dx) &
        - (drift2_at_pos(n,2)*p_last(n,2)-drift2_at_pos(n,n)*p_last(n,n))/(2.0*dx) &
        + (p_last(n,2)-2.0*p_last(n,1)+p_last(n,n))/(dx*dx)) 
    ddx(n,n) = D*dt*(-(drift1_at_pos(1,n)*p_last(1,n)-drift1_at_pos(n-1,n)*p_last(n-1,n))/(2.0*dx) &
        + (p_last(1,n)-2.0*p_last(n,n)+p_last(n-1,n))/(dx*dx) &
        - (drift2_at_pos(n,1)*p_last(n,1)-drift2_at_pos(n,n-1)*p_last(n,n-1))/(2.0*dx) &
        + (p_last(n,1)-2.0*p_last(n,n)+p_last(n,n-1))/(dx*dx))

    ! iterate through all the coordinates,not on the corners,for both variables
    !$omp parallel do
    do i=2,n-1
        !! Periodic boundary conditions:
        !! Explicitly update FPE for edges not corners
        ddx(1,i) = D*dt*(-(drift1_at_pos(2,i)*p_last(2,i)-drift1_at_pos(n,i)*p_last(n,i))/(2.0*dx) &
            + (p_last(2,i)-2*p_last(1,i)+p_last(n,i))/(dx*dx) &
            - (drift2_at_pos(1,i+1)*p_last(1,i+1)-drift2_at_pos(1,i-1)*p_last(1,i-1))/(2.0*dx) &
            + (p_last(1,i+1)-2*p_last(1,i)+p_last(1,i-1))/(dx*dx))
        ddx(i,1) = D*dt*(-(drift1_at_pos(i+1,1)*p_last(i+1,1)-drift1_at_pos(i-1,1)*p_last(i-1,1))/(2.0*dx) &
            + (p_last(i+1,1)-2*p_last(i,1)+p_last(i-1,1))/(dx*dx) &
            - (drift2_at_pos(i,2)*p_last(i,2)-drift2_at_pos(i,n)*p_last(i,n))/(2.0*dx) &
            + (p_last(i,2)-2*p_last(i,1)+p_last(i,n))/(dx*dx))

        !! all points with well defined neighbours go like so:
        do j=2,n-1
            ddx(i,j) = D*dt*(-(drift1_at_pos(i+1,j)*p_last(i+1,j)-drift1_at_pos(i-1,j)*p_last(i-1,j))/(2.0*dx) &
                + (p_last(i+1,j)-2.0*p_last(i,j)+p_last(i-1,j))/(dx*dx) &
                - (drift2_at_pos(i,j+1)*p_last(i,j+1)-drift2_at_pos(i,j-1)*p_last(i,j-1))/(2.0*dx) &
                + (p_last(i,j+1)-2.0*p_last(i,j)+p_last(i,j-1))/(dx*dx)) 
        end do

        !! Explicitly update FPE for rest of edges not corners
        ddx(n,i) = D*dt*(-(drift1_at_pos(1,i)*p_last(1,i)-drift1_at_pos(n-1,i)*p_last(n-1,i))/(2.0*dx) &
            + (p_last(1,i)-2.0*p_last(n,i)+p_last(n-1,i))/(dx*dx) &
            - (drift2_at_pos(n,i+1)*p_last(n,i+1)-drift2_at_pos(n,i-1)*p_last(n,i-1))/(2.0*dx) &
            + (p_last(n,i+1)-2.0*p_last(n,i)+p_last(n,i-1))/(dx*dx))
        ddx(i,n) = D*dt*(-(drift1_at_pos(i+1,n)*p_last(i+1,n)-drift1_at_pos(i-1,n)*p_last(i-1,n))/(2.0*dx) &
            + (p_last(i+1,n)-2.0*p_last(i,n)+p_last(i-1,n))/(dx*dx) &
            - (drift2_at_pos(i,1)*p_last(i,1)-drift2_at_pos(i,n-1)*p_last(i,n-1))/(2.0*dx) &
            + (p_last(i,1)-2.0*p_last(i,n)+p_last(i,n-1))/(dx*dx)) 
    end do
    !$omp end parallel do
end subroutine spatial_derivs_FD_2D

end module FDiff