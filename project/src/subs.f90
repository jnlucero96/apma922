! store a bunch of subroutines
module subs

contains

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

end module subs