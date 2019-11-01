module FDiff

! this is what you think it is
real*8, parameter :: pi = 3.14159265358979323846264338327950288419716939937510582
! machine eps
real*8, parameter :: float32_eps = 1.1920928955078125e-07
real*8, parameter :: float64_eps = 2.22044604925031308084726e-16

real*8, parameter :: D = 0.001 ! diffusion constant

contains


! given a discretized linear operator L, get steady state distribution
subroutine get_steady_state( &
    n, dx, theta, dt, num_minima, E, psi, check_step, p_final &
    )
    integer*8, intent(in) :: n
    real*8, intent(in) :: dx
    real*8, dimension(n), intent(in) :: theta
    real*8, intent(in) :: dt, num_minima, E, psi
    integer*8, intent(in) :: check_step
    real*8, dimension(n), intent(inout) :: p_final

    real*8, dimension(:), allocatable :: mu
    real*8, dimension(:,:), allocatable :: L
    real*8, dimension(:), allocatable :: p_now, p_last, p_last_ref

    ! continue condition
    logical cc
    integer*8 step_counter ! counting steps
    real*8 tot_var_dist ! total variation distance

    ! lapack consts.
    integer*8 mm,nn


    allocate(mu(n), p_now(n), p_last(n), p_last_ref(n), L(n,n))

    ! initialize based on initial data
    p_now = initialize_distribution(n, theta, num_minima, E)

    p_last = 0.0; p_last_ref = 0.0

    mu = calc_drift(n, theta, num_minima, E, psi)

    ! initialize the FPE operator
    L = initl(n, mu, dx)

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    do while (cc)

        ! update probabilities
        call dgemv('n', n, n, real(1.0,kind=8), dt*L, n, p_last, 1, real(1.0,kind=8), p_last, 1)

        ! bail at the first sign of trouble
        if (abs(sum(p_last) - 1.0) .ge. float32_eps) stop "Normalization broken!"

        if (step_counter .EQ. check_step) then
            tot_var_dist = 0.5*sum(abs(p_last_ref-p_last))

            print *, tot_var_dist

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0
                p_last_ref = p_last
            end if
        end if
        step_counter = step_counter + 1
    end do

    p_final = 0.0; p_final = p_last

end subroutine get_steady_state

subroutine calculate_L(n, theta, num_minima, E, psi, dx, L)
    integer*8, intent(in) :: n
    real*8, dimension(n), intent(in) :: theta
    real*8, intent(in) :: num_minima
    real*8, intent(in) :: E, psi, dx
    real*8, dimension(n,n) :: L
    real*8, dimension(:), allocatable :: mu

    allocate(mu(n))

    mu = calc_drift(n, theta, num_minima, E, psi)

    L = initl(n, mu, dx)

end subroutine calculate_L

function calc_drift(n, theta, num_minima, E, psi) result(mu)
    integer*8, intent(in) :: n
    real*8, dimension(n), intent(in) :: theta
    real*8, intent(in) :: num_minima, E, psi
real*8, dimension(:), allocatable :: mu

    allocate(mu(n))

    mu = -D*(0.5*num_minima*E*sin(num_minima*theta)-(psi))
end function calc_drift

function potential(n, num_minima, E, theta) result(U)
    integer*8, intent(in) :: n
    real*8, intent(in) :: num_minima, E
    real*8, dimension(n), intent(in) :: theta
    real*8, dimension(:), allocatable :: U

    allocate(U(n))

    U = 0.5*E*(1-cos(num_minima*theta))
end function potential

function initialize_distribution(n, theta, num_minima, E) result(p_eq)
    integer*8, intent(in) :: n
    real*8, dimension(n), intent(in) :: theta
    real*8, intent(in) :: num_minima, E
    real*8, dimension(:), allocatable :: U, p_eq

    allocate(U(n), p_eq(n))

    U = potential(n, num_minima, E, theta)
    p_eq = exp(-U)

    ! normalize the distribution
    p_eq = p_eq / sum(p_eq)
end function initialize_distribution

function initl(n, mu, dx) result(L)
    integer*8, intent(in) :: n
    real*8, dimension(n), intent(in) :: mu
    real*8, intent(in) :: dx
    real*8, dimension(:,:), allocatable :: L
    real*8, dimension(:,:), allocatable :: D1, D2
    integer*8 ii ! iteration indexes

    allocate(D1(n,n), D2(n,n))

    ! initialization
    D1 = 0.0; D2 = 0.0

    ! set up differentiation matrices
    do ii=1,n
        ! 1st derivative operator
        D1(modulo(ii-1,n)+1,modulo(ii-2,n)+1) = (-1.0)*mu(modulo(ii-2,n)+1) ! sub-diagonal
        D1(modulo(ii-1,n)+1,modulo(ii,n)+1) = mu(modulo(ii,n)+1) ! super-diagonal

        ! 2nd derivative operator
        D2(modulo(ii-1,n)+1,modulo(ii-2,n)+1) = 1.0 ! sub-diagonal
        D2(modulo(ii-1,n)+1,modulo(ii-1,n)+1) = -2.0 ! diagonal
        D2(modulo(ii-1,n)+1,modulo(ii,n)+1) = 1.0 ! super-diagonal
    end do

    D1 = (-1.0)*D1/(2.0*dx)
    D2 = D2/(dx*dx)

    ! define the linear fokker-planck operator
    L = D*(D1+D2)
end function initl

!========================= UTILITIES ================================

function linspace(start, stop, n)
    real*8, intent(in) :: start, stop
    integer*8, intent(in) :: n
    real*8, dimension(:), allocatable :: linspace
    real*8 delta
    integer*8 i

    delta = (stop-start)/(n-1)

    allocate(linspace(n))

    do i=1,n
        linspace(i) = start + i*delta
    end do
end function linspace

function kahan_sum(n, array)
    integer*8, intent(in) :: n
    real*8, dimension(n,n), intent(in) :: array
    real*8 kahan_sum, c, t
    integer*8 iii, jjj ! declare iterators

    ! initialization
    kahan_sum = 0.0; c = 0.0

    do jjj=1,n
        do iii=1,n
            t = kahan_sum + array(iii,jjj)
            if (abs(kahan_sum) .GE. abs(array(iii,jjj))) then
                c = c + (kahan_sum - t) + array(iii,jjj)
            else
                c = c + (array(iii,jjj)-t) + kahan_sum
            end if
            kahan_sum = t
        end do
    end do
end function kahan_sum

end module FDiff