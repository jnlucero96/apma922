! FFTW demo, compile with
! gfortran -O3 -fdefault-real-8 -I/opt/local/include bvp-2d-parallel.f90 -L /opt/healpix/lib -lcfitsio -L/opt/local/lib -lfftw3_threads -lfftw3
module fft_solve
implicit none

include "fftw3.f"

real(8), parameter :: float32_eps = 1.1920928955078125e-07
real(8), parameter :: float64_eps = 8.22044604925031308084726e-12
real(8), parameter :: pi = 4.0*atan(1.0)

contains

subroutine get_steady_spectral_ft_2D( &
	n, m, dt, check_step, D, dx, dy, &
	dmu1, dmu2, ddmu1, ddmu2, p_initial, p_final &
	)
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D
    real(8), intent(in) :: dx, dy
    real(8), dimension(n,m) :: dmu1, dmu2, ddmu1, ddmu2
    real(8), dimension(n,m), intent(in) :: p_initial
    real(8), dimension(n,m), intent(inout) :: p_final

    real(8), dimension(:,:), allocatable :: p_now, p_last, p_last_ref, dv

    ! continue condition
    logical cc
    integer(8) step_counter ! counting steps
    real(8) tot_var_dist ! total variation distance

    allocate(p_now(n,m), p_last(n,m), p_last_ref(n,m), dv(n,m))

    ! initialize based on initial data
	p_last = p_initial

    ! initialize reference array
    p_last_ref = 0.0; p_now = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    do while (cc)

        call update_probability(n, dx, m, dy, dv, p_last, dmu1, ddmu1, dmu2, ddmu2, D)

		print *, sum(p_last)
		! update using Euler step
        p_now = p_last + dt*dv
		print *, sum(p_now)

        ! bail at the first sign of trouble
        ! if (abs(sum(p_now) - 1.0) .ge. float32_eps) stop "Normalization broken!"

        if (step_counter .EQ. check_step) then
			tot_var_dist = 0.5*sum(abs(p_last_ref-p_now))

            print *, abs(maxval(dt*dv)), tot_var_dist

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

end subroutine get_steady_spectral_ft_2D

subroutine update_probability(n, dx, m, dy, update, p_last, dmu1, ddmu1, dmu2, ddmu2, D)
	integer(8), intent(in) :: n, m
	real(8), intent(in) :: dx, dy
	real(8), dimension(n,m), intent(inout) :: update
	real(8), dimension(n,m), intent(in) :: p_last
	real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2
	real(8), intent(in) :: D
	real(8), dimension(n,m) :: dv, ddv

	! advection part
	call divergence(n, dx, m, dy, dmu1, ddmu1, dmu2, ddmu2, p_last, dv)

	! diffusion part
	call laplacian(n, dx, m, dy, p_last, ddv)

	update = D*(-dv + ddv)
end subroutine update_probability

subroutine divergence(n, dx, m, dy, dmu1, ddmu1, dmu2, ddmu2, in_array, out_array)
	integer(8), intent(in) :: n, m
	real(8), intent(in) :: dx, dy
	real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2
	real(8), dimension(n,m), intent(in) :: in_array
	real(8), dimension(n,m), intent(inout) :: out_array

	! declare interior variables
	complex(8), dimension(n,m) :: in, out1, out2, dst1, dst2
	integer(8) plan1, plan2, status
	integer(8) i, j

	real(8), dimension(:), allocatable :: kx, ky

	allocate( kx(n), ky(m) )

	kx = fftfreq(n, dx)
	ky = fftfreq(m, dy)

	! normalize the highest mode since taking odd derivative
	kx(n/2+1) = 0.0
	ky(n/2+1) = 0.0

	! initialize the array
	in = in_array

	! initialize the temporary arrays
	dst1 = 0.0; dst2 = 0.0
	out1 = 0.0; out2 = 0.0

	call dfftw_init_threads(status)
	call dfftw_plan_with_nthreads(2)

	call dfftw_plan_dft_2d(plan1,n,m,in,dst1,FFTW_FORWARD,FFTW_ESTIMATE)
	call dfftw_execute_dft(plan1,in,dst1)
	call dfftw_destroy_plan(plan1)

	do j = 1,m
		do i = 1,n
			dst2(i,j) = (0.0,1.0)*dst1(i,j)*ky(j)
			dst1(i,j) = (0.0,1.0)*dst1(i,j)*kx(i)
		end do
	end do

	call dfftw_plan_dft_2d(plan1,n,m,dst1,out1,FFTW_BACKWARD,FFTW_ESTIMATE)
	call dfftw_execute_dft(plan1, dst1, out1)
	call dfftw_destroy_plan(plan1)

	call dfftw_plan_dft_2d(plan2,n,m,dst2,out2,FFTW_BACKWARD,FFTW_ESTIMATE)
	call dfftw_execute_dft(plan2, dst2, out2)
	call dfftw_destroy_plan(plan2)

	call dfftw_cleanup_threads()

	out_array = 0.0
	out_array = (ddmu1+ddmu2)*in_array + (dmu1*realpart(out1)/(n*m)) + (dmu2*realpart(out2)/(n*m))
end subroutine divergence

subroutine laplacian(n, dx, m, dy, in_array, out_array)
	integer(8), intent(in) :: n, m
	real(8), intent(in) :: dx, dy
	real(8), dimension(n,m), intent(in) :: in_array
	real(8), dimension(n,m), intent(inout) :: out_array

	! declare interior variables
	complex(8), dimension(n,m) :: in, out, dst
	integer(8) plan, status
	integer(8) i, j

	real(8), dimension(:), allocatable :: kx, ky

	allocate( kx(n), ky(n) )

	kx = fftfreq(n, dx)
	ky = fftfreq(m, dy)

	! initialize the array
	in = in_array

	call dfftw_init_threads(status)
	call dfftw_plan_with_nthreads(2)

	call dfftw_plan_dft_2d(plan,n,m,in,dst,FFTW_FORWARD,FFTW_ESTIMATE)

	call dfftw_execute_dft(plan, in, dst)
	call dfftw_destroy_plan(plan)

	do j = 1,m
		do i = 1,n
			dst(i,j) = (-1.0)*(kx(i)**2 + ky(j)**2 )*dst(i,j)
		end do
	end do

	call dfftw_plan_dft_2d(plan,n,m,dst,out,FFTW_BACKWARD,FFTW_ESTIMATE)
	call dfftw_execute_dft(plan, dst, out)
	call dfftw_destroy_plan(plan)

	call dfftw_cleanup_threads()

	out_array = 0.0; out_array = realpart(out)/(n*m)
end subroutine laplacian

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

function linspace(start, end, n)
    real(8), intent(in) :: start, end
    integer(8), intent(in) :: n
    integer(8) i
    real(8) h
    real(8), dimension(:), allocatable :: linspace

    allocate( linspace(n) )

    h = (end-start)/(n-1)
    do i=1,n
        linspace(i) = start + (i-1)*h
    end do
end function linspace

end module fft_solve