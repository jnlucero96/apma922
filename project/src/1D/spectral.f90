module fft_solve
implicit none

include "fftw3.f"

real(8), parameter :: float32_eps = 1.1920928955078125e-07
real(8), parameter :: float64_eps = 8.22044604925031308084726e-16
real(8), parameter :: pi = 4.0*atan(1.0)

contains

subroutine get_spectral_steady( &
    n, m, dt, scheme, check_step, D, dx, dy, &
    dmu1, ddmu1, dmu2, ddmu2, p_initial, p_final &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: scheme
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2
    real(8), dimension(n,m), intent(in) :: p_initial
    real(8), dimension(n,m), intent(inout) :: p_final

    ! continue condition
    logical cc
    integer(8) step_counter ! counting steps
    real(8) tot_var_dist ! total variation distance

    complex(8), dimension(n,m) :: pr, phat_mat, p_penultimate, px_now
    complex(8), dimension(n*m+1) :: p_now
    real(8), dimension(n,m) :: p_last_ref
    integer(8) plan0, plan1, plan2, status

    integer(8) i
    real(8), dimension(:,:), allocatable :: kx, ky
    real(8), dimension(:), allocatable :: EE, EE2, L, Q, f1, f2, f3
    real(8), dimension(:,:), allocatable :: kkx, kky
    complex(8), dimension(:), allocatable :: r
    complex(8), dimension(:,:), allocatable :: LR

    allocate( kx(n,m), ky(n,m) )
    allocate( EE(n*m+1), EE2(n*m+1), L(n*m+1), r(16), LR(n*m+1, 16), Q(n*m+1), & 
        f1(n*m+1), f2(n*m+1), f3(n*m+1), kkx(n,m), kky(n,m))

    pr = p_initial

    call dfftw_init_threads(status)
    call dfftw_plan_with_nthreads(2)

    ! planning is good
    call dfftw_plan_dft_2d(plan0,n,m,pr,phat_mat,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat_mat,px_now,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan2,n,m,phat_mat,p_penultimate,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! take the distribution into k-space
    call dfftw_execute_dft(plan0,pr,phat_mat)

    call dfftw_destroy_plan(plan0)

    ! initialize based on fourier transform of initial data
    p_now(:n*m) = pack(phat_mat,.TRUE.)
    p_now(n*m+1) = 0.0 ! set time = 0

    phat_mat = 0.0 ! reset the phat_mat matrix

    ! initialize reference array
    p_last_ref = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    ! get the frequencies ready
    do i=1,n
        kx(:,i) = fftfreq(n, dx)
        ky(i,:) = fftfreq(m, dy)
    end do

    if (scheme .eq. 6) then

        kkx = kx; kky = ky
        if (modulo(n,2) .eq. 0) kkx(n/2+1,:) = 0.0
        if (modulo(m,2) .eq. 0) kky(:,m/2+1) = 0.0

        L(:n*m) = pack(kkx**2 + kky**2, .TRUE.); L(n*m+1) = 1.0
        EE = exp(-dt*L); EE2 = exp(-0.5*dt*L)
        r = exp((0.0,1.0)*pi*([(i, i=1,16)]-0.5)/16)
        LR = spread(dt*L,2,16) + spread(r,1,n*m+1)

        do i=1,n*m+1
            Q(i) = realpart(mean(16, (exp(0.5*LR(i,:))-1)/LR(i,:)))
            f1(i) = realpart(mean(16, (-4.0-LR(i,:)+exp(LR(i,:))*(4.0-3.0*LR(i,:)+LR(i,:)**2))/(LR(i,:)**3)))
            f2(i) = realpart(mean(16, (2.0+LR(i,:)+exp(LR(i,:)*(-2.0+LR(i,:))))/(LR(i,:)**3)))
            f3(i) = realpart(mean(16, (-4.0-3.0*LR(i,:)-LR(i,:)**2+exp(LR(i,:))*(4.0-LR(i,:)))/(LR(i,:)**3)))
        end do
    else
        deallocate( EE, EE2, L, r, LR, Q, f1, f2, f3 )
    end if

    do while (cc)

        ! update probabilities using an given scheme
        select case (scheme)
            case(1); call forward_euler(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(2); call imex(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(3); call rk2(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(4); call rk4(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(5); call ifrk4(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(6); call etdrk4( &
                n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt, &
                EE, EE2, Q, f1, f2, f3 &
                )
            case(7); call gl04(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(8); call gl06(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(9); call gl08(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
            case(10); call gl10(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p_now, D, dt)
        end select

        print *, "Normalization:", realpart(p_now(1))

        if (step_counter .EQ. check_step) then

            phat_mat = reshape(p_now(:n*m), [n,m])

            call dfftw_execute(plan1, phat_mat, px_now)

            ! bail at first sign of trouble
            if (abs(sum(realpart(px_now)/(n*m)) - 1.0) .ge. float32_eps) stop "Normalization broken!"

            ! compute total variation distance
            tot_var_dist = 0.5*sum(abs(p_last_ref-(realpart(px_now)/(n*m))))

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0; phat_mat = 0.0
                p_last_ref = realpart(px_now)/(n*m)
            end if
        end if
        step_counter = step_counter + 1
    end do

    phat_mat = reshape(p_now(:n*m), [n,m])

    p_penultimate = 0.0
    ! take the distribution back into real-space
    call dfftw_execute_dft(plan2,phat_mat,p_penultimate)
    call dfftw_destroy_plan(plan2)

    call dfftw_cleanup_threads()

    p_final = 0.0; p_final = realpart(p_penultimate)/(n*m)

    if (abs(sum(p_final) - 1.0) .ge. float32_eps) stop "Normalization broken!"
end subroutine get_spectral_steady

subroutine forward_euler(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update

    allocate( update(n*m+1) )

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p, update)

    ! update the solution
    p = p + dt*update
end subroutine forward_euler

subroutine imex(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update

    allocate( update(n*m+1) )

    call evalf2(n, m, D, dt, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p, update)

    p = update
end subroutine imex

! 2nd order Runge-Kutta integrator with fixed step size
subroutine rk2(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update, yt, dydx, dyt

    allocate( update(n*m+1), yt(n*m+1), dydx(n*m+1), dyt(n*m+1) )

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p, dydx)
    yt = p + dt*dydx

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dyt)

    update = dt*(dydx + dyt)/2.0

    ! update the solution
    p = p + update
end subroutine rk2

! 4th order Runge-Kutta integrator with fixed step size
subroutine rk4(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update, yt, dydx, dyt, dym
    real(8) hh

    allocate( update(n*m+1), yt(n*m+1), dydx(n*m+1), dyt(n*m+1), dym(n*m+1) )

    hh = 0.5*dt

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p, dydx)
    yt = p + hh*dydx

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dyt)
    yt = p + hh*dyt

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dym)
    yt = p + dt*dym
    dym = dyt + dym

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dyt)

    update = dt*(dydx + dyt + 2.0*dym)/6.0

    ! update the solution
    p = p + update
end subroutine rk4

! 4th order Runge-Kutta integrator using integrating factor
subroutine ifrk4(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update, yt, dydx, dyt, dym, EE, EE2

    allocate( update(n*m+1), yt(n*m+1), dydx(n*m+1), dyt(n*m+1), dym(n*m+1), EE(n*m+1), EE2(n*m+1) )

    EE(:n*m) = pack(exp(-D*(kx**2+ky**2)*dt), .TRUE.); EE(n*m+1) = 1.0
    EE2 = EE**2

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p, dydx)
    yt = EE*(p + 0.5*dydx)

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dyt)
    yt = (EE*p + 0.5*dyt)

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dym)
    yt = (EE2*p + EE*dym)
    dym = dyt + dym

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dyt)

    update = dt*(EE2*dydx + dyt + 2.0*EE*dym)/6.0

    ! update the solution
    p = EE2*p + update
end subroutine ifrk4 

! 4th order Runge-Kutta integrator using exponential time differencing
subroutine etdrk4( &
    n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt, EE, EE2, Q, f1, f2, f3 &
    )
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    real(8), dimension(n*m+1), intent(in) :: EE, EE2, Q, f1, f2, f3
    complex(8), dimension(:), allocatable :: update, yta, yt, dydx, dyt, dym

    allocate( update(n*m+1), yta(n*m+1), yt(n*m+1), dydx(n*m+1), dyt(n*m+1), dym(n*m+1) )

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p, dydx)
    yta = (EE2*p + dt*Q*dydx)

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yta, dyt)
    yt = (EE2*p + dt*Q*dyt)

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dym)
    yt = (EE2*yta + dt*Q*(-dydx+2.0*dyt))
    dym = dyt + dym

    call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, yt, dyt)

    update = dt*(dydx*f1 + dyt*f3 + 2.0*f2*dym)

    ! update the solution
    p = EE*p + update
end subroutine etdrk4 

! 2nd order implicit Gauss-Legendre integrator
subroutine gl02(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 1 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
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
            call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p+g(:,1)*dt, g(:,1))
    end do

    update = g(:,1)*b*dt

    ! update the solution
    p = p + update
end subroutine gl02

! 4th order implicit Gauss-Legendre integrator
subroutine gl04(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 2 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
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
            call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl04

! 6th order implicit Gauss-Legendre integrator
subroutine gl06(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 3 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
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
            call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl06

! 8th order implicit Gauss-Legendre integrator
subroutine gl08(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 4 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
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
            call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl08

! 10th order implicit Gauss-Legendre integrator
subroutine gl10(n, m, dmu1, ddmu1, dmu2, ddmu2, kx, ky, p, D, dt)
    integer(8), parameter :: s = 5 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2, kx, ky
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
            call evalf(n, m, D, dmu1, dmu2, ddmu1, ddmu2, kx, ky, p+g(:,i)*dt, g(:,i))
        end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl10

! represent f entirely explicitly
subroutine evalf( &
    n, m, D, &
    dmu1, dmu2, ddmu1, ddmu2, kx, ky, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2, ddmu1, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update

    ! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2, phat3
    complex(8), dimension(n,m) :: out0, out1, out2
    integer(8) plan0, plan1, plan2, plan3, plan4, plan5

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
    call dfftw_plan_dft_2d(plan4,n,m,out1,phat1,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan5,n,m,out2,phat2,FFTW_FORWARD,FFTW_ESTIMATE)

    phat1 = (0.0,1.0)*kx*pr;
    if (modulo(n,2) .eq. 0) phat1(n/2+1,:) = 0.0

    phat2 = (0.0,1.0)*ky*pr;
    if (modulo(m,2) .eq. 0) phat2(:,m/2+1) = 0.0

    phat3 = -D*( kx**2+ky**2 )*pr

    ! go back into real space
    call dfftw_execute_dft(plan0, pr, out0)
    call dfftw_execute_dft(plan1, phat1, out1)
    call dfftw_execute_dft(plan2, phat2, out2)

    ! do multiplications in real space
    out0 = -D*(ddmu1+ddmu2)*(realpart(out0)/(n*m))
    out1 = -D*dmu1*(realpart(out1)/(n*m))
    out2 = -D*dmu2*(realpart(out2)/(n*m))

    ! reset the k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! go back into fourier space
    call dfftw_execute_dft(plan3, out0, phat0)
    call dfftw_execute_dft(plan4, out1, phat1)
    call dfftw_execute_dft(plan5, out2, phat2)

    call dfftw_destroy_plan(plan0)
    call dfftw_destroy_plan(plan1)
    call dfftw_destroy_plan(plan2)
    call dfftw_destroy_plan(plan3)
    call dfftw_destroy_plan(plan4)
    call dfftw_destroy_plan(plan5)

    update = 0.0
    update(:n*m) = pack(phat0 + phat1 + phat2 + phat3,.TRUE.)
    update(n*m+1) = 1.0
end subroutine evalf

! integrator for implicit-explicit methods
subroutine evalf2( &
    n, m, D, dt, &
    dmu1, dmu2, ddmu1, ddmu2, kx, ky, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), intent(in) :: dt
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2, ddmu1, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update

    ! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2, phat3
    complex(8), dimension(n,m) :: out0, out1, out2
    integer(8) plan0, plan1, plan2, plan3, plan4, plan5

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
    call dfftw_plan_dft_2d(plan4,n,m,out1,phat1,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan5,n,m,out2,phat2,FFTW_FORWARD,FFTW_ESTIMATE)

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
    out0 = -D*(ddmu1+ddmu2)*(realpart(out0)/(n*m))
    out1 = -D*dmu1*(realpart(out1)/(n*m))
    out2 = -D*dmu2*(realpart(out2)/(n*m))

    ! reset the k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! go back into fourier space
    call dfftw_execute_dft(plan3, out0, phat0)
    call dfftw_execute_dft(plan4, out1, phat1)
    call dfftw_execute_dft(plan5, out2, phat2)

    call dfftw_destroy_plan(plan0)
    call dfftw_destroy_plan(plan1)
    call dfftw_destroy_plan(plan2)
    call dfftw_destroy_plan(plan3)
    call dfftw_destroy_plan(plan4)
    call dfftw_destroy_plan(plan5)

    update = 0.0
    update(:n*m) = pack((phat0 + phat1 + phat2 + phat3)/((1.0/dt)+0.5*D*( kx**2 + ky**2 )),.TRUE.)
    update(n*m+1) = 1.0
end subroutine evalf2

! rhs for 2nd-order IMEX methods
subroutine evalf3( &
    n, m, D, dt, &
    dmu1, dmu2, ddmu1, ddmu2, kx, ky, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), intent(in) :: dt
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2, ddmu1, ddmu2, kx, ky
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update

    ! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2, phat3
    complex(8), dimension(n,m) :: out0, out1, out2
    integer(8) plan0, plan1, plan2, plan3, plan4, plan5

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
    call dfftw_plan_dft_2d(plan4,n,m,out1,phat1,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan5,n,m,out2,phat2,FFTW_FORWARD,FFTW_ESTIMATE)

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
    out0 = -D*(ddmu1+ddmu2)*(realpart(out0)/(n*m))
    out1 = -D*dmu1*(realpart(out1)/(n*m))
    out2 = -D*dmu2*(realpart(out2)/(n*m))

    ! reset the k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! go back into fourier space
    call dfftw_execute_dft(plan3, out0, phat0)
    call dfftw_execute_dft(plan4, out1, phat1)
    call dfftw_execute_dft(plan5, out2, phat2)

    call dfftw_destroy_plan(plan0)
    call dfftw_destroy_plan(plan1)
    call dfftw_destroy_plan(plan2)
    call dfftw_destroy_plan(plan3)
    call dfftw_destroy_plan(plan4)
    call dfftw_destroy_plan(plan5)

    update = 0.0
    update(:n*m) = pack((phat0 + phat1 + phat2 + phat3)/((1.0/dt)+0.5*D*( kx**2 + ky**2 )),.TRUE.)
    update(n*m+1) = 1.0
end subroutine evalf3

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

function mean(n, x) result(mu)
    integer(8), intent(in) :: n
    complex(8), dimension(n), intent(in) :: x
    complex(8) mu

    mu = sum(x)/real(n,8)
end function mean

end module fft_solve