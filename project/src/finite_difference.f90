module FDiff

! machine eps
real(8), parameter :: float32_eps = 1.1920928955078125e-07
real(8), parameter :: float64_eps = 8.22044604925031308084726e-16

contains

subroutine get_fd_steady( &
    n, m, dt, scheme, check_step, D, dx, dy, &
    dmu1, dmu2, p_initial, p_final, refarray &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: scheme
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2
    real(8), dimension(n,m), intent(in) :: p_initial
    real(8), dimension(n,m), intent(inout) :: p_final
    real(8), dimension(1), intent(inout) :: refarray

    real(8), dimension(:,:), allocatable :: p_now, p_last_ref

    real(8) num_checks

    ! continue condition
    logical cc
    integer(8) step_counter ! counting steps
    real(8) tot_var_dist ! total variation distance

    allocate( p_now(n,m), p_last_ref(n,m) )

    ! initialize based on initial data
    p_now = p_initial

    ! initialize reference array
    p_last_ref = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0; num_checks = 0.0

    do while (cc)

        select case (scheme)
            case(1); call forward_euler(n, m, dx, dy, dmu1, dmu2, p_now, D, dt)
            case(2); call rk2(n, m, dx, dy, dmu1, dmu2, p_now, D, dt)
        end select

        if (step_counter .EQ. check_step) then

            ! bail at the first sign of trouble
            if (abs(sum(p_now) - 1.0) .ge. float32_eps) stop "Normalization broken!"
            if (count(p_now < -float64_eps) .GE. 1) stop "Negative probabilities!"

            tot_var_dist = 0.5*sum(abs(p_last_ref-p_now))

            num_checks = num_checks + 1.0

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0
                p_last_ref = p_now
            end if
        end if
        ! cycle the variables
        step_counter = step_counter + 1
    end do

    refarray(1) = num_checks

    p_final = 0.0; p_final = p_now
end subroutine get_fd_steady

subroutine get_fd_fwd( &
    nsteps, ntrack, n, m, dt, scheme, check_step, D, dx, dy, &
    dmu1, dmu2, p_initial, p_trace &
    )
    integer(8), intent(in) :: nsteps, ntrack, n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: scheme
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2
    real(8), dimension(n,m), intent(in) :: p_initial
    real(8), dimension(n,m,ntrack), intent(inout) :: p_trace

    real(8), dimension(:,:), allocatable :: p_now

    integer(8) i, counter, step_counter ! counting steps

    allocate( p_now(n,m) )

    ! initialize based on initial data
    p_now = p_initial

    ! initialize loop variables
    step_counter = 0; counter = 2

    do i=1,nsteps

        select case (scheme)
            case(1); call forward_euler(n, m, dx, dy, dmu1, dmu2, p_now, D, dt)
            case(2); call rk2(n, m, dx, dy, dmu1, dmu2, p_now, D, dt)
        end select

        if (step_counter .EQ. check_step) then
            ! bail at the first sign of trouble
            if (abs(sum(p_now) - 1.0) .GE. float32_eps) stop "Normalization broken!"
            if (count(p_now < -float64_eps) .GE. 1) stop "Negative Probabilities!"

            p_trace(:,:,counter) = p_now
            counter = counter + 1
            step_counter = 0

        end if
        ! cycle the variables
        step_counter = step_counter + 1
    end do
end subroutine get_fd_fwd

subroutine forward_euler(n, m, dx, dy, dmu1, dmu2, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2
    real(8), dimension(n,m), intent(inout) :: p
    real(8), intent(in) :: D, dt
    real(8), dimension(:,:), allocatable :: update

    allocate( update(n,m) )

    call spatial_derivs_FD_2D(n, m, dx, dy, D, dmu1, dmu2, p, update)

    ! update the solution
    p = p + dt*update
end subroutine forward_euler

subroutine rk2(n, m, dx, dy, dmu1, dmu2, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2
    real(8), dimension(n,m), intent(inout) :: p
    real(8), intent(in) :: D, dt
    real(8), dimension(:,:), allocatable :: update, yt, dydx, dyt

    allocate( update(n,m), yt(n,m), dydx(n,m), dyt(n,m) )

    call spatial_derivs_FD_2D(n, m, dx, dy, D, dmu1, dmu2, p, dydx)
    yt = p + dt*dydx

    call spatial_derivs_FD_2D(n, m, dx, dy, D, dmu1, dmu2, yt, dyt)

    update = dt*(dydx + dyt)/2.0

    ! update the solution
    p = p + update
end subroutine rk2

subroutine spatial_derivs_FD_2D( &
    n, m, dx, dy, D, dmu1, dmu2, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dx, dy
    real(8), intent(in) :: D
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2
    real(8), dimension(n,m), intent(in) :: in_array
    real(8), dimension(n,m), intent(inout) :: update

    integer(8) i,j ! iterator variables

    !! Periodic boundary conditions:
    !! Explicity update FPE for the corners
    update(1,1) = D*(-(dmu1(2,1)*in_array(2,1)-dmu1(n,1)*in_array(n,1))/(2.0*dx) &
        + (in_array(2,1)-2.0*in_array(1,1)+in_array(n,1))/(dx*dx) &
        - (dmu2(1,2)*in_array(1,2)-dmu2(1,n)*in_array(1,n))/(2.0*dy) &
        + (in_array(1,2)-2.0*in_array(1,1)+in_array(1,n))/(dy*dy))
    update(1,m) = D*(-(dmu1(2,n)*in_array(2,n)-dmu1(n,n)*in_array(n,n))/(2.0*dx) &
        + (in_array(2,n)-2.0*in_array(1,n)+in_array(n,n))/(dx*dx) &
        - (dmu2(1,1)*in_array(1,1)-dmu2(1,n-1)*in_array(1,n-1))/(2.0*dy) &
        + (in_array(1,1)-2.0*in_array(1,n)+in_array(1,n-1))/(dy*dy))
    update(n,1) = D*(-(dmu1(1,1)*in_array(1,1)-dmu1(n-1,1)*in_array(n-1,1))/(2.0*dx) &
        + (in_array(1,1)-2.0*in_array(n,1)+in_array(n-1,1))/(dx*dx) &
        - (dmu2(n,2)*in_array(n,2)-dmu2(n,n)*in_array(n,n))/(2.0*dy) &
        + (in_array(n,2)-2.0*in_array(n,1)+in_array(n,n))/(dy*dy))
    update(n,m) = D*(-(dmu1(1,n)*in_array(1,n)-dmu1(n-1,n)*in_array(n-1,n))/(2.0*dx) &
        + (in_array(1,n)-2.0*in_array(n,n)+in_array(n-1,n))/(dx*dx) &
        - (dmu2(n,1)*in_array(n,1)-dmu2(n,n-1)*in_array(n,n-1))/(2.0*dy) &
        + (in_array(n,1)-2.0*in_array(n,n)+in_array(n,n-1))/(dy*dy))

    ! iterate through all the edge grid points, not on the corners, for both variables
    do i=2,n-1
        update(i,1) = D*(-(dmu1(i+1,1)*in_array(i+1,1)-dmu1(i-1,1)*in_array(i-1,1))/(2.0*dx) &
            + (in_array(i+1,1)-2*in_array(i,1)+in_array(i-1,1))/(dx*dx) &
            - (dmu2(i,2)*in_array(i,2)-dmu2(i,n)*in_array(i,n))/(2.0*dy) &
            + (in_array(i,2)-2*in_array(i,1)+in_array(i,n))/(dy*dy))
        update(i,m) = D*(-(dmu1(i+1,m)*in_array(i+1,m)-dmu1(i-1,m)*in_array(i-1,m))/(2.0*dx) &
            + (in_array(i+1,m)-2.0*in_array(i,m)+in_array(i-1,m))/(dx*dx) &
            - (dmu2(i,1)*in_array(i,1)-dmu2(i,m-1)*in_array(i,m-1))/(2.0*dy) &
            + (in_array(i,1)-2.0*in_array(i,m)+in_array(i,m-1))/(dy*dy))
    end do
    do j=2,m-1
        update(1,j) = D*(-(dmu1(2,j)*in_array(2,j)-dmu1(n,j)*in_array(n,j))/(2.0*dx) &
            + (in_array(2,j)-2*in_array(1,j)+in_array(n,j))/(dx*dx) &
            - (dmu2(1,j+1)*in_array(1,j+1)-dmu2(1,j-1)*in_array(1,j-1))/(2.0*dy) &
            + (in_array(1,j+1)-2*in_array(1,j)+in_array(1,j-1))/(dy*dy))
        update(n,j) = D*(-(dmu1(1,j)*in_array(1,j)-dmu1(n-1,j)*in_array(n-1,j))/(2.0*dx) &
            + (in_array(1,j)-2.0*in_array(n,j)+in_array(n-1,j))/(dx*dx) &
            - (dmu2(n,j+1)*in_array(n,j+1)-dmu2(n,j-1)*in_array(n,j-1))/(2.0*dy) &
            + (in_array(n,j+1)-2.0*in_array(n,j)+in_array(n,j-1))/(dy*dy))
    end do

    !! all points with well defined neighbours go like so:
    do i=2,n-1
        do j=2,m-1
            update(i,j) = D*(-(dmu1(i+1,j)*in_array(i+1,j)-dmu1(i-1,j)*in_array(i-1,j))/(2.0*dx) &
                + (in_array(i+1,j)-2.0*in_array(i,j)+in_array(i-1,j))/(dx*dx) &
                - (dmu2(i,j+1)*in_array(i,j+1)-dmu2(i,j-1)*in_array(i,j-1))/(2.0*dy) &
                + (in_array(i,j+1)-2.0*in_array(i,j)+in_array(i,j-1))/(dy*dy))
        end do
    end do

end subroutine spatial_derivs_FD_2D

end module FDiff