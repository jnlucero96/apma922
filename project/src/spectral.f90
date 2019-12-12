module fft_solve
implicit none

include "fftw3.f"

real(8), parameter :: float32_eps = 1.1920928955078125e-07
real(8), parameter :: float64_eps = 8.22044604925031308084726e-16
real(8), parameter :: pi = 4.0*atan(1.0)

contains

! propagate initial distribution until it reaches steady-state
subroutine get_spectral_steady( &
    n, m, dt, scheme, check_step, D, dx, dy, &
    mu1, dmu1, mu2, dmu2, p_initial, p_final, refarray &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: scheme
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2
    real(8), dimension(n,m), intent(in) :: p_initial
    real(8), dimension(n,m), intent(inout) :: p_final
    real(8), dimension(1), intent(inout) :: refarray

    ! continue condition
    logical cc
    integer(8) step_counter ! counting steps
    real(8) tot_var_dist ! total variation distance

    complex(8), dimension(n,m) :: pr, phat_mat, p_penultimate, px_now
    complex(8), dimension(n*m+1) :: p_now
    real(8), dimension(n,m) :: p_last_ref
    integer(8) plan0, plan1, plan2

    integer(8) i, cpts
    real(8) num_checks
    real(8), dimension(:,:), allocatable :: kx, ky
    real(8), dimension(:), allocatable :: EE, EE2, L, Q, f1, f2, f3
    real(8), dimension(:,:), allocatable :: kkx, kky
    complex(8), dimension(:), allocatable :: r
    complex(8), dimension(:,:), allocatable :: LR

    allocate( kx(n,m), ky(n,m) )
    allocate( EE(n*m+1), EE2(n*m+1), L(n*m+1), r(16), LR(n*m+1, 16), Q(n*m+1), &
        f1(n*m+1), f2(n*m+1), f3(n*m+1), kkx(n,m), kky(n,m))

    pr = p_initial

    ! planning is good
    call dfftw_plan_dft_2d(plan0,n,m,pr,phat_mat,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat_mat,px_now,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan2,n,m,phat_mat,p_penultimate,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! take the distribution into k-space
    call dfftw_execute_dft(plan0,pr,phat_mat)

    call dfftw_destroy_plan(plan0)

    ! initialize based on fourier transform of initial data
    p_now(:n*m) = pack(phat_mat, .TRUE.)
    p_now(n*m+1) = 0.0 ! set time = 0

    phat_mat = 0.0 ! reset the phat_mat matrix

    ! initialize reference array
    p_last_ref = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    ! get the frequencies ready
    kx = spread(fftfreq(n,dx),2,m)
    ky = spread(fftfreq(m,dy),1,n)

    ! special set up step for etd method
    if (scheme .eq. 6) then

        kkx = kx; kky = ky
        if (modulo(n,2) .eq. 0) kkx(n/2+1,:) = 0.0
        if (modulo(m,2) .eq. 0) kky(:,m/2+1) = 0.0

        cpts = 32

        L(:n*m) = pack(-D*(kkx**2 + kky**2), .TRUE.); L(n*m+1) = 1.0
        EE = exp(dt*L); EE(n*m+1) = 1.0; EE2 = exp(0.5*dt*L); EE2(n*m+1) = 1.0
        r = exp((0.0,1.0)*pi*([(i, i=1,cpts)]-0.5)/cpts)
        LR = spread(dt*L,2,cpts) + spread(r,1,n*m+1)

        Q = realpart(sum((exp(0.5*LR)-1)/LR, 2))/real(cpts,8)
        f1 = realpart(sum((-4.0-LR+exp(LR)*(4.0-3.0*LR+LR**2))/(LR**3),2))/real(cpts,8)
        f2 = realpart(sum((2.0+LR+exp(LR)*(-2.0+LR))/(LR**3),2))/real(cpts,8)
        f3 = realpart(sum((-4.0-3.0*LR-LR**2+exp(LR)*(4.0-LR))/(LR**3),2))/real(cpts,8)

        Q(n*m+1) = 1.0; f1(n*m+1) = 1.0; f2(n*m+1) = 1.0; f3(n*m+1) = 1.0
    else
        ! free memory if not needed
        deallocate( EE, EE2, L, r, LR, Q, f1, f2, f3 )
    end if

    num_checks = 0.0

    do while (cc)

        ! update probabilities using specified scheme
        select case (scheme)
            case(1); call forward_euler(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(2); call imex(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(3); call rk2(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(4); call rk4(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(5); call ifrk4(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(6); call etdrk4( &
                n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt, &
                EE, EE2, Q, f1, f2, f3 &
                )
            case(7); call gl04(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(8); call gl06(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(9); call gl08(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(10); call gl10(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
        end select

        if (step_counter .EQ. check_step) then

            phat_mat = reshape(p_now(:n*m), [n,m])

            call dfftw_execute(plan1, phat_mat, px_now)

            ! check normalization and non-negativity of distributions
            ! bail at first sign of trouble
            if (abs(sum(realpart(px_now)/(n*m)) - 1.0) .ge. float32_eps) stop "Normalization broken!"
            if (count(realpart(px_now)/(n*m) < -float64_eps) .ge. 1) stop "Negative probabilities!"

            ! compute total variation distance
            tot_var_dist = 0.5*sum(abs(p_last_ref-(realpart(px_now)/(n*m))))

            num_checks = num_checks + 1.0 ! increment counter

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0; phat_mat = 0.0
                p_last_ref = realpart(px_now)/(n*m)
            end if
        end if
        step_counter = step_counter + 1
    end do

    call dfftw_destroy_plan(plan1)

    phat_mat = reshape(p_now(:n*m), [n,m])

    p_penultimate = 0.0
    ! take the distribution back into real-space
    call dfftw_execute_dft(plan2,phat_mat,p_penultimate)
    call dfftw_destroy_plan(plan2)

    p_final = 0.0; p_final = realpart(p_penultimate)/(n*m)

    refarray(1) = num_checks ! record the number of checks

    ! final checks on distribution before exiting subroutine
    ! bail if checks fail
    if (abs(sum(p_final) - 1.0) .ge. float32_eps) stop "Normalization broken!"
    if (count(p_final < -float64_eps) .ge. 1) stop "Negative probabilities!"
end subroutine get_spectral_steady

! propogate distribution for a set time
subroutine get_spectral_fwd( &
    nsteps, ntrack, n, m, dt, scheme, check_step, D, dx, dy, &
    mu1, dmu1, mu2, dmu2, p_initial, p_trace &
    )
    integer(8), intent(in) :: nsteps, ntrack, n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: scheme
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2
    real(8), dimension(n,m), intent(in) :: p_initial
    real(8), dimension(n,m,ntrack), intent(inout) :: p_trace

    integer(8) step_counter ! counting steps

    complex(8), dimension(n,m) :: pr, phat_mat, px_now
    complex(8), dimension(n*m+1) :: p_now
    integer(8) plan0, plan1

    integer(8) i, counter, cpts
    real(8), dimension(:,:), allocatable :: kx, ky
    real(8), dimension(:), allocatable :: EE, EE2, L, Q, f1, f2, f3
    real(8), dimension(:,:), allocatable :: kkx, kky
    complex(8), dimension(:), allocatable :: r
    complex(8), dimension(:,:), allocatable :: LR

    allocate( kx(n,m), ky(n,m) )
    allocate( EE(n*m+1), EE2(n*m+1), L(n*m+1), r(16), LR(n*m+1, 16), Q(n*m+1), &
        f1(n*m+1), f2(n*m+1), f3(n*m+1), kkx(n,m), kky(n,m) )

    pr = p_initial

    ! planning is good
    call dfftw_plan_dft_2d(plan0,n,m,pr,phat_mat,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat_mat,px_now,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! take the distribution into k-space
    call dfftw_execute_dft(plan0,pr,phat_mat)

    call dfftw_destroy_plan(plan0)

    ! initialize based on fourier transform of initial data
    p_now(:n*m) = pack(phat_mat, .TRUE.)
    p_now(n*m+1) = 0.0 ! set time = 0

    phat_mat = 0.0 ! reset the phat_mat matrix

    ! get the frequencies ready
    kx = spread(fftfreq(n,dx),2,m)
    ky = spread(fftfreq(m,dy),1,n)

    if (scheme .eq. 6) then
        kkx = kx; kky = ky
        if (modulo(n,2) .eq. 0) kkx(n/2+1,:) = 0.0
        if (modulo(m,2) .eq. 0) kky(:,m/2+1) = 0.0

        cpts = 16

        L(:n*m) = pack(-D*(kkx**2 + kky**2), .TRUE.); L(n*m+1) = 1.0
        EE = exp(dt*L); EE(n*m+1) = 1.0; EE2 = exp(0.5*dt*L); EE2(n*m+1) = 1.0
        r = exp((0.0,1.0)*pi*([(i, i=1,cpts)]-0.5)/cpts)
        LR = spread(dt*L,2,cpts) + spread(r,1,n*m+1)

        Q = realpart(sum((exp(0.5*LR)-1)/LR, 2))/real(cpts,8)
        f1 = realpart(sum((-4.0-LR+exp(LR)*(4.0-3.0*LR+LR**2))/(LR**3),2))/real(cpts,8)
        f2 = realpart(sum((2.0+LR+exp(LR)*(-2.0+LR))/(LR**3),2))/real(cpts,8)
        f3 = realpart(sum((-4.0-3.0*LR-LR**2+exp(LR)*(4.0-LR))/(LR**3),2))/real(cpts,8)

        Q(n*m+1) = 1.0; f1(n*m+1) = 1.0; f2(n*m+1) = 1.0; f3(n*m+1) = 1.0
    else
        ! free memory if not needed
        deallocate( EE, EE2, L, r, LR, Q, f1, f2, f3 )
    end if

    counter = 2; step_counter = 0

    do i=1,nsteps

        ! update probabilities using specified scheme
        select case (scheme)
            case(1); call forward_euler(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(2); call imex(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(3); call rk2(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(4); call rk4(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(5); call ifrk4(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(6); call etdrk4( &
                n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt, &
                EE, EE2, Q, f1, f2, f3 &
                )
            case(7); call gl04(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(8); call gl06(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(9); call gl08(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
            case(10); call gl10(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)
        end select

        if (step_counter .EQ. check_step) then

            phat_mat = reshape(p_now(:n*m), [n,m])

            call dfftw_execute(plan1, phat_mat, px_now)

            ! bail at first sign of trouble
            if (abs(sum(realpart(px_now)/(n*m)) - 1.0) .ge. float32_eps) stop "Normalization broken!"
            if (count(realpart(px_now)/(n*m) < -float64_eps) .ge. 1) stop "Negative probabilities!"

            p_trace(:,:,counter) = realpart(px_now)/(n*m) ! store this iteration

            counter = counter + 1 ! increment
            step_counter = 0 ! reset the counter
        end if
        step_counter = step_counter + 1
    end do

    call dfftw_destroy_plan(plan1)
end subroutine get_spectral_fwd

! simple wrapper for taking a Forward Euler step
subroutine forward_euler(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2
    real(8), dimension(n,m), intent(in) :: kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update

    allocate( update(n*m+1) )

    call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p, update)

    ! update the solution
    p = p + dt*update
end subroutine forward_euler

! wrapper for Crank-Nicolson Forward Euler (CNFT) scheme
subroutine imex(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update

    allocate( update(n*m+1) )

    call evalRHS2(n, m, D, dt, mu1, mu2, dmu1, dmu2, kx, ky, p, update)

    p = update
end subroutine imex

! 2nd order Runge-Kutta integrator with fixed step size
! derivative of RK4 scheme drawn from "Numerical Recipes in Fortran" by
! Press et al.
subroutine rk2(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update, yt, dydx, dyt

    allocate( update(n*m+1), yt(n*m+1), dydx(n*m+1), dyt(n*m+1) )

    call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p, dydx)
    yt = p + dt*dydx

    call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, yt, dyt)

    update = dt*(dydx + dyt)/2.0

    ! update the solution
    p = p + update
end subroutine rk2

! 4th order Runge-Kutta integrator with fixed step size
! naming conventions drawn from Kassam and Trefethen (2005)
subroutine rk4(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2
    real(8), dimension(n,m), intent(in) :: kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update, yt
    complex(8), dimension(:), allocatable :: a, b, c, dd
    real(8) hdt

    allocate( &
        update(n*m+1), yt(n*m+1), &
        a(n*m+1), b(n*m+1), c(n*m+1), dd(n*m+1) &
        )

    hdt = 0.5*dt

    call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p, a)

    call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p + hdt*a, b)

    call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p + hdt*b, c)

    call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p + dt*c, dd)

    update = dt*(a + 2.0*(b+c) + dd)/6.0

    ! update the solution
    p = p + update
end subroutine rk4

! 4th order Runge-Kutta integrator using integrating factor
! drawn from p.27 of "Spectral Methods in MATLAB by Trefethen"
subroutine ifrk4(n, m, mu1, dmu1, mu2, dmu2, kx, ky, v, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2
    real(8), dimension(n,m), intent(in) :: kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: v
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update
    complex(8), dimension(:), allocatable :: a, b, c, dd, E, E2

    allocate( &
        update(n*m+1), &
        a(n*m+1), b(n*m+1), c(n*m+1), dd(n*m+1), &
        E(n*m+1), E2(n*m+1) &
        )

    ! define integrating factor
    E(:n*m) = pack(exp(-dt*D*(kx**2+ky**2)/2.0), .TRUE.); E(n*m+1) = 1.0
    E2 = E**2

    call evalRHS_nonlinear(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, v, a)

    call evalRHS_nonlinear( &
        n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, E*(v + a/2.0), b &
        )

    call evalRHS_nonlinear( &
        n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, (E*v + b/2.0), c &
        )

    call evalRHS_nonlinear( &
        n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, (E2*v + E*c), dd &
        )

    update = dt*(E2*a + 2.0*E*(b+c) + dd)/6.0

    ! uvdate the solution
    v = E2*v + update
end subroutine ifrk4

! 4th order Runge-Kutta integrator using exponential time differencing
! drawn from "kursiv.m" by Trefethen
subroutine etdrk4( &
    n, m, mu1, dmu1, mu2, dmu2, kx, ky, v, D, dt, &
    EE, EE2, Q, f1, f2, f3 &
    )
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2
    real(8), dimension(n,m), intent(in) :: kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: v
    real(8), intent(in) :: D, dt
    real(8), dimension(n*m+1), intent(in) :: EE, EE2
    real(8), dimension(n*m+1), intent(in) :: Q, f1, f2, f3
    complex(8), dimension(:), allocatable :: update, a, b, c
    complex(8), dimension(:), allocatable :: Nv, Na, Nb, Nc

    allocate( &
        update(n*m+1), a(n*m+1), b(n*m+1), c(n*m+1), &
        Nv(n*m+1), Na(n*m+1), Nb(n*m+1), Nc(n*m+1) &
        )

    call evalRHS_nonlinear(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, v, Nv)
    a = (EE2*v + dt*Q*Nv)

    call evalRHS_nonlinear(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, a, Na)
    b = (EE2*v + dt*Q*Na)

    call evalRHS_nonlinear(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, b, Nb)
    c = (EE2*a + dt*Q*(2.0*Nb-Nv))

    call evalRHS_nonlinear(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, c, Nc)

    update = dt*(Nv*f1 + 2.0*f2*(Na+Nb) + Nc*f3)
    update(n*m+1) = dt ! manually update the time

    ! update the solution
    v = EE*v + update
end subroutine etdrk4

! 2nd order implicit Gauss-Legendre integrator
subroutine gl02(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 1 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update
    complex(8), dimension(:,:), allocatable :: g
    integer(8) k

    ! Butcher tableau for 2nd order Gauss-Legendre method
    complex(8), parameter :: a = 0.5
    complex(8), parameter :: b = 1.0

    allocate( g(n*m+1,s), update(n*m+1) )

    ! iterate trial steps
    g = 0.0; do k = 1,16
            g(:,1) = g(:,1)*a
            call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p+g(:,1)*dt, g(:,1))
    end do

    update = g(:,1)*b*dt

    ! update the solution
    p = p + update
end subroutine gl02

! 4th order implicit Gauss-Legendre integrator
subroutine gl04(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 2 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update
    complex(8), dimension(:,:), allocatable :: g
    integer(8) i, k

    ! Butcher tableau for 4th order Gauss-Legendre method
    complex(8), parameter :: a(s,s) = reshape( &
        (/ 0.25, 0.25 - 0.5/sqrt(3.0), 0.25 + 0.5/sqrt(3.0), 0.25 /), (/s,s/))
    complex(8), parameter :: b(s) = (/ 0.5, 0.5 /)

    allocate( g(n*m+1,s), update(n*m+1) )

    ! iterate trial steps
    g = 0.0; do k = 1,16
        g = matmul(g,a)
        do i = 1,s
            call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl04

! 6th order implicit Gauss-Legendre integrator
subroutine gl06(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 3 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update
    complex(8), dimension(:,:), allocatable :: g
    integer(8) i, k

    ! Butcher tableau for 6th order Gauss-Legendre method
    complex(8), parameter :: a(s,s) = reshape((/ &
            5.0/36.0, 2.0/9.0 - 1.0/sqrt(15.0), 5.0/36.0 - 0.5/sqrt(15.0), &
            5.0/36.0 + sqrt(15.0)/24.0, 2.0/9.0, 5.0/36.0 - sqrt(15.0)/24.0, &
            5.0/36.0 + 0.5/sqrt(15.0), 2.0/9.0 + 1.0/sqrt(15.0), 5.0/36.0 /), (/s,s/))
    complex(8), parameter ::   b(s) = (/ 5.0/18.0, 4.0/9.0, 5.0/18.0/)

    allocate( g(n*m+1,s), update(n*m+1) )

    ! iterate trial steps
    g = 0.0; do k = 1,16
        g = matmul(g,a)
        do i = 1,s
            call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl06

! 8th order implicit Gauss-Legendre integrator
subroutine gl08(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 4 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update
    complex(8), dimension(:,:), allocatable :: g
    integer(8) i, k

    ! Butcher tableau for 8th order Gauss-Legendre method
    complex(8), parameter :: a(s,s) = reshape((/ &
                0.869637112843634643432659873054998518Q-1, -0.266041800849987933133851304769531093Q-1, &
                0.126274626894047245150568805746180936Q-1, -0.355514968579568315691098184956958860Q-2, &
                0.188118117499868071650685545087171160Q0,   0.163036288715636535656734012694500148Q0,  &
            -0.278804286024708952241511064189974107Q-1,  0.673550059453815551539866908570375889Q-2, &
                0.167191921974188773171133305525295945Q0,   0.353953006033743966537619131807997707Q0,  &
                0.163036288715636535656734012694500148Q0,  -0.141906949311411429641535704761714564Q-1, &
                0.177482572254522611843442956460569292Q0,   0.313445114741868346798411144814382203Q0,  &
                0.352676757516271864626853155865953406Q0,   0.869637112843634643432659873054998518Q-1 /), (/s,s/))
    complex(8), parameter ::   b(s) = (/ &
                0.173927422568726928686531974610999704Q0,   0.326072577431273071313468025389000296Q0,  &
                0.326072577431273071313468025389000296Q0,   0.173927422568726928686531974610999704Q0  /)

    allocate( g(n*m+1,s), update(n*m+1) )

    ! iterate trial steps
    g = 0.0; do k = 1,16
        g = matmul(g,a)
        do i = 1,s
            call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl08

! 10th order implicit Gauss-Legendre integrator
subroutine gl10(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 5 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update
    complex(8), dimension(:,:), allocatable :: g
    integer(8) i, k

    ! Butcher tableau for 10th order Gauss-Legendre method
    complex(8), parameter :: a(s,s) = reshape((/ &
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
    complex(8), parameter ::   b(s) = [ &
                1.1846344252809454375713202035995868132Q-1,  2.3931433524968323402064575741781909646Q-1, &
                2.8444444444444444444444444444444444444Q-1,  2.3931433524968323402064575741781909646Q-1, &
                1.1846344252809454375713202035995868132Q-1]

    allocate( g(n*m+1,s), update(n*m+1) )

    ! iterate trial steps
    g = 0.0;
    do k = 1,16
        g = matmul(g,a)
        do i = 1,s
            call evalRHS(n, m, D, mu1, mu2, dmu1, dmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl10

! represent entire RHS
subroutine evalRHS( &
    n, m, D, &
    mu1, mu2, dmu1, dmu2, kx, ky, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), dimension(n,m), intent(in) :: mu1, mu2, dmu1, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update

    ! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2, phat3
    complex(8), dimension(n,m) :: out0, out1, out2
    integer(8) plan0, plan1, plan2, plan3

    real(8) t

    ! process the input array
    pr = 0.0; pr = reshape(in_array(:n*m), [n,m])
    t = realpart(in_array(n*m+1))

    ! initialize real-space arrays
    out0 = 0.0; out1 = 0.0; out2 = 0.0
    ! initialize k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0; phat3 = 0.0

    ! plan all of the fft's that are to be done

    ! inverse fourier transforms back to real-space
    call dfftw_plan_dft_2d(plan0,n,m,pr,out0,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat1,out1,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan2,n,m,phat2,out2,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! forward fourier transforms to k-space
    call dfftw_plan_dft_2d(plan3,n,m,out0,phat0,FFTW_FORWARD,FFTW_ESTIMATE)

    phat1 = (0.0,1.0)*kx*pr;
    if (modulo(n,2) .eq. 0) phat1(n/2,:) = 0.0

    phat2 = (0.0,1.0)*ky*pr;
    if (modulo(m,2) .eq. 0) phat2(:,m/2) = 0.0

    phat3 = -D*( kx**2+ky**2 )*pr

    ! go back into real space
    call dfftw_execute_dft(plan0, pr, out0)
    call dfftw_execute_dft(plan1, phat1, out1)
    call dfftw_execute_dft(plan2, phat2, out2)

    ! do multiplications in real space
    out0 = -D*( &
        (dmu1+dmu2)*(realpart(out0)/(n*m)) &
        + mu1*(realpart(out1)/(n*m)) &
        + mu2*(realpart(out2)/(n*m)) &
        )

    ! reset the k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! go back into fourier space
    call dfftw_execute_dft(plan3, out0, phat0)

    call dfftw_destroy_plan(plan0)
    call dfftw_destroy_plan(plan1)
    call dfftw_destroy_plan(plan2)
    call dfftw_destroy_plan(plan3)

    update = 0.0
    update(:n*m) = pack(phat0 + phat3,.TRUE.)
    update(n*m+1) = 1.0
end subroutine evalRHS

! same as evalRHS but only evaluate the non-linear terms
! ie. don't evaluate the diffusion term
subroutine evalRHS_nonlinear( &
    n, m, D, &
    mu1, mu2, dmu1, dmu2, kx, ky, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), dimension(n,m), intent(in) :: mu1, mu2
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update

    ! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2
    complex(8), dimension(n,m) :: out0, out1, out2
    integer(8) plan0, plan1, plan2, plan3

    real(8) t

    ! process the input array
    pr = 0.0; pr = reshape(in_array(:n*m), [n,m])
    t = realpart(in_array(n*m+1))

    ! initialize real-space arrays
    out0 = 0.0; out1 = 0.0; out2 = 0.0
    ! initialize k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! plan all of the fft's that are to be done

    ! inverse fourier transforms back to real-space
    call dfftw_plan_dft_2d(plan0,n,m,pr,out0,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat1,out1,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan2,n,m,phat2,out2,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! forward fourier transforms to k-space
    call dfftw_plan_dft_2d(plan3,n,m,out0,phat0,FFTW_FORWARD,FFTW_ESTIMATE)

    phat1 = (0.0,1.0)*kx*pr;
    if (modulo(n,2) .eq. 0) phat1(n/2,:) = 0.0

    phat2 = (0.0,1.0)*ky*pr;
    if (modulo(m,2) .eq. 0) phat2(:,m/2) = 0.0

    ! go back into real space
    call dfftw_execute_dft(plan0, pr, out0)
    call dfftw_execute_dft(plan1, phat1, out1)
    call dfftw_execute_dft(plan2, phat2, out2)

    ! do multiplications in real space
    out0 = -D*( &
        (dmu1+dmu2)*(realpart(out0)/(n*m)) &
        + mu1*(realpart(out1)/(n*m)) &
        + mu2*(realpart(out2)/(n*m)) &
        )

    ! reset the k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! go back into fourier space
    call dfftw_execute_dft(plan3, out0, phat0)

    call dfftw_destroy_plan(plan0)
    call dfftw_destroy_plan(plan1)
    call dfftw_destroy_plan(plan2)
    call dfftw_destroy_plan(plan3)

    update = 0.0
    update(:n*m) = pack(phat0,.TRUE.)
    update(n*m+1) = 1.0
end subroutine evalRHS_nonlinear

! integrator for CNFT method
subroutine evalRHS2( &
    n, m, D, dt, &
    mu1, mu2, dmu1, dmu2, kx, ky, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), intent(in) :: dt
    real(8), dimension(n,m), intent(in) :: mu1, mu2, dmu1, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update

    ! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2, phat3
    complex(8), dimension(n,m) :: out0, out1, out2
    integer(8) plan0, plan1, plan2, plan3

    real(8) t

    ! process the input array
    pr = 0.0; pr = reshape(in_array(:n*m), [n,m])
    t = realpart(in_array(n*m+1))

    ! initialize real-space arrays
    out0 = 0.0; out1 = 0.0; out2 = 0.0
    ! initialize k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0; phat3 = 0.0

    ! plan all of the fft's that are to be done

    ! inverse fourier transforms back to real-space
    call dfftw_plan_dft_2d(plan0,n,m,pr,out0,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat1,out1,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan2,n,m,phat2,out2,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! forward fourier transforms to k-space
    call dfftw_plan_dft_2d(plan3,n,m,out0,phat0,FFTW_FORWARD,FFTW_ESTIMATE)

    phat1 = (0.0,1.0)*kx*pr
    if (modulo(n,2) .eq. 0) phat1(n/2+1,:) = 0.0

    phat2 = (0.0,1.0)*ky*pr
    if (modulo(m,2) .eq. 0) phat2(:,m/2+1) = 0.0

    phat3 = ((1.0/dt)-0.5*D*( kx**2+ky**2 ))*pr

    ! go back into real space
    call dfftw_execute_dft(plan0, pr, out0)
    call dfftw_execute_dft(plan1, phat1, out1)
    call dfftw_execute_dft(plan2, phat2, out2)

    ! do multiplications in real space
    out0 = -D*( &
        (dmu1+dmu2)*(realpart(out0)/(n*m)) &
        + mu1*(realpart(out1)/(n*m)) &
        + mu2*(realpart(out2)/(n*m)) &
        )

    ! reset the k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! go back into fourier space
    call dfftw_execute_dft(plan3, out0, phat0)

    call dfftw_destroy_plan(plan0)
    call dfftw_destroy_plan(plan1)
    call dfftw_destroy_plan(plan2)
    call dfftw_destroy_plan(plan3)

    update = 0.0
    update(:n*m) = pack((phat0 + phat3)/((1.0/dt)+0.5*D*( kx**2 + ky**2 )),.TRUE.)
    update(n*m+1) = 1.0
end subroutine evalRHS2

! get the frequencies in the correct order
! frequencies scaled appropriately to the length of the domain
function fftfreq(n, d)
    integer(8), intent(in) :: n
    real(8), intent(in) :: d
    real(8), dimension(n) :: fftfreq

    integer(8) i

    do i=1,n/2
        fftfreq(i) = (i-1)
        fftfreq(n+1-i) = -i
    end do

    fftfreq = (2.0*pi)*fftfreq/(d*n)
end function fftfreq

end module fft_solve