! FFTW demo, compile with
! gfortran -O3 -fdefault-real-8 -I/opt/local/include bvp-2d-parallel.f90 -L /opt/healpix/lib -lcfitsio -L/opt/local/lib -lfftw3_threads -lfftw3
module fft_solve
implicit none

include "fftw3.f"

real(8), parameter :: float32_eps = 1.1920928955078125e-07
real(8), parameter :: float64_eps = 8.22044604925031308084726e-12
real(8), parameter :: pi = 4.0*atan(1.0)

contains

subroutine get_spectral_steady_ft( &
    n, m, dt, check_step, D, dx, dy, dmu1, ddmu1, dmu2, ddmu2, p_initial, p_final &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2
    real(8), dimension(n,m), intent(inout) :: p_initial
    real(8), dimension(n,m), intent(inout) :: p_final

    ! continue condition
    logical cc
    integer(8) step_counter ! counting steps
    real(8) tot_var_dist ! total variation distance
    
    complex(8), dimension(n,m) :: pr, phat_mat, p_penultimate
    complex(8), dimension(n*m+1) :: p_now
    complex(8), dimension(n*m) :: p_last_ref
    integer(8) plan0, plan1, status

    complex(8), dimension(:), allocatable :: update

    allocate( update(n*m+1) )

    print *, sum(p_initial)

    pr = p_initial

	! call dfftw_init_threads(status)
	! call dfftw_plan_with_nthreads(2)

	! planning is good
    call dfftw_plan_dft_2d(plan0,n,m,pr,phat_mat,FFTW_FORWARD,FFTW_ESTIMATE)

    ! take the distribution into k-space
    call dfftw_execute_dft(plan0,pr,phat_mat)

	call dfftw_destroy_plan(plan0)
    ! call dfftw_cleanup_threads()

    ! initialize based on fourier transform of initial data
    p_now(:n*m) = pack(phat_mat,.TRUE.)
    p_now(n*m+1) = 0.0 ! set time = 0

    phat_mat = 0.0 ! reset the phat_mat matrix

    ! initialize reference array
    p_last_ref = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    print *, maxval(abs(p_now(:n*m)))
    print *, maxval(dmu1), maxval(ddmu1), maxval(dmu2), maxval(ddmu2)

    do while (cc)

        ! update probabilities using a Gauss-Legendre scheme
        call evalf( &
            n, m, D, dx, dy, &
            dmu1, dmu2, ddmu1, ddmu2, p_now, update &
            )
        print *, "Max value check:", maxval(abs(p_now(:n*m)))

        p_now = p_now + dt*update

        ! bail at the first sign of trouble
        ! if (abs(sum(p_now(:n*m)) - 1.0) .ge. float32_eps) stop "Normalization broken!"

        if (step_counter .EQ. check_step) then
            tot_var_dist = 0.5*sum(abs(p_last_ref-p_now(:n*m)))

            print *, "Tot. var. dist = ", tot_var_dist

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0
                p_last_ref = p_now(:n*m)
            end if
        end if
        step_counter = step_counter + 1
    end do

    phat_mat = reshape(p_now(:n*m), [n,m])

    print *, "Phat_mat pass off:", maxval(abs(phat_mat))

	! planning is good
    call dfftw_plan_dft_2d(plan1,n,m,phat_mat,p_penultimate,FFTW_BACKWARD,FFTW_ESTIMATE)

    p_penultimate = 0.0
    ! take the distribution back into real-space
    call dfftw_execute_dft(plan1,phat_mat,p_penultimate)
    call dfftw_destroy_plan(plan1)

    p_final = 0.0; p_final = realpart(p_penultimate)/(n*m)

    print *, "maxvalue p final:", maxval(p_final)
    print *, "final normalization:", sum(p_final)
    
    if (abs(sum(p_final) - 1.0) .ge. float32_eps) stop "Normalization broken!"

    ! p_final = 0.0; p_final = reshape(p_now(:n*m),[n,m])
end subroutine get_spectral_steady_ft

subroutine get_spectral_steady_gl10( &
    n, m, dt, check_step, D, dx, dy, dmu1, ddmu1, dmu2, ddmu2, p_initial, p_final &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2
    real(8), dimension(n,m), intent(inout) :: p_initial
    real(8), dimension(n,m), intent(inout) :: p_final

    ! continue condition
    logical cc
    integer(8) step_counter ! counting steps
    real(8) tot_var_dist ! total variation distance
    
    complex(8), dimension(n,m) :: pr, phat_mat, p_penultimate, px_now
    complex(8), dimension(n*m+1) :: p_now
    real(8), dimension(n,m) :: p_last_ref
    integer(8) plan0, plan1, plan2, status

    print *, sum(p_initial)

    pr = p_initial

	call dfftw_init_threads(status)
	call dfftw_plan_with_nthreads(4)

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

    do while (cc)

        ! update probabilities using a Gauss-Legendre scheme
        call gl10_1D(n, dx, m, dy, dmu1, ddmu1, dmu2, ddmu2, p_now, D, dt)

        if (step_counter .EQ. check_step) then

            phat_mat = reshape(p_now(:n*m), [n,m])

            call dfftw_execute(plan1, phat_mat, px_now)

            ! bail at first sign of trouble
            if (abs(sum(realpart(px_now)/(n*m)) - 1.0) .ge. float32_eps) stop "Normalization broken!"

            ! compute total variation distance
            tot_var_dist = 0.5*sum(abs(p_last_ref-(realpart(px_now)/(n*m))))

            print *, "Tot. var. dist = ", tot_var_dist

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

    print *, "maxvalue p final:", maxval(p_final)
    print *, "final normalization:", sum(p_final)
    
    if (abs(sum(p_final) - 1.0) .ge. float32_eps) stop "Normalization broken!"

    ! p_final = 0.0; p_final = reshape(p_now(:n*m),[n,m])
end subroutine get_spectral_steady_gl10

! 10th order implicit Gauss-Legendre integrator
subroutine gl10_1D(n, dx, m, dy, dmu1, ddmu1, dmu2, ddmu2, p, D, dt)
    integer(8), parameter :: s = 5 ! number of stages
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, ddmu1, dmu2, ddmu2
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
    g = 0.0; do k = 1,16
            g = matmul(g,a)
            do i = 1,s
                call evalf(n, m, D, dx, dy, dmu1, dmu2, ddmu1, ddmu2, p+g(:,i)*dt, g(:,i))
            end do
    end do

    update = matmul(g,b)*dt

    ! update the solution
    p = p + update
end subroutine gl10_1D

subroutine evalf( &
	n, m, D, dx, dy, &
	dmu1, dmu2, ddmu1, ddmu2, in_array, update &
	)
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), intent(in) :: dx, dy
    real(8), dimension(n,m), intent(in) :: dmu1, dmu2, ddmu1, ddmu2
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update

	! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2, phat3, phat4
    complex(8), dimension(n,m) :: out0, out1, out2, out3, out4
	integer(8) plan0, plan1, plan2, plan3, plan4, plan5, plan6, plan7
	integer(8) status
    integer(8) i, j

	real(8) t

    real(8), dimension(:,:), allocatable :: kx, ky, kkx, kky
    
    allocate( kx(n,m), ky(n,m) )

    ! process the input array
    pr = 0.0; pr = reshape(in_array(:n*m), [n,m])
    t = in_array(n*m+1)

	! get the frequencies ready
	do i=1,n
        kx(:,i) = fftfreq(n, dx)
		ky(i,:) = fftfreq(m, dy)
	end do

    ! initialize real-space arrays
    out0 = 0.0; out1 = 0.0; out2 = 0.0; out3 = 0.0; out4 = 0.0
    ! initialize k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0; phat3 = 0.0; phat4 = 0.0

    ! plan all of the fft's that are to be done

    ! inverse fourier transforms back to real-space
	call dfftw_plan_dft_2d(plan0,n,m,pr,out0,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat1,out1,FFTW_BACKWARD,FFTW_ESTIMATE)
	call dfftw_plan_dft_2d(plan2,n,m,phat2,out2,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! forward fourier transforms to k-space
	call dfftw_plan_dft_2d(plan3,n,m,out0,phat0,FFTW_FORWARD,FFTW_ESTIMATE)
	call dfftw_plan_dft_2d(plan4,n,m,out1,phat1,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan5,n,m,out2,phat2,FFTW_FORWARD,FFTW_ESTIMATE)

	phat1 = (0.0,1.0)*kx*pr; phat1(n/2+1,:) = 0.0
    phat2 = (0.0,1.0)*ky*pr; phat2(:,m/2+1) = 0.0
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