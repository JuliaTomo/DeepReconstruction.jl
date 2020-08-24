using CUDA
using CUDA.CUSPARSE
using Flux
using SparseArrays

function fp_dip(x, A::CuArray{Float32,2})
    A * x
end

function fp_dip(x, A::SparseMatrixCSC{Float32,Int64})
    cu( A * cpu(x) )
end

Zygote.@adjoint function fp_dip(x, A)
    "x: [n x 1]"
    "A: [m x n]"

    return fp_dip(x, A), function(dldp)
        # dldp : [m]
        if typeof(A) != SparseMatrixCSC{Float32,Int64}
            out = A' * vec(dldp)
        else
            # if A is in cpu
            out = cu( A' * vec(cpu(dldp)) )
        end

        return (out, nothing)
    end
end

function loss(net, z, p_data, A)
    z_out = net(z)
    p_est = fp_dip(vec(z_out), A)

    _ = Flux.mae(p_est, p_data)
end

"""
    opt_dip

Reconstruct 2D image based on Deep Image Prior.

# Args
- net : UNet
- opt : Flux optimizer
- p_data : sinogram data
- A (sparse matrix Float32 in cpu) : forward projection opeartor
- H, W : image size to reconstruct
- verbose : 2 if you want to save the best result during the iteration instead of early stopping. There should be a global variable `img_gt` for the ground truth image. Moreover, if there is global variable `dresult`, we save the image.
"""
function recon2d_dip(net, opt, p_data, A::SparseMatrixCSC{Float32,Int64}, H::Int, W::Int, niter=2000, ichannel=3, verbose=2)

    # if there is no img_gt, we don't compare it.
    ~(@isdefined img_gt) && (verbose = 1)

    ps = Flux.params(net)
    p_data = cu(vec(p_data))

    if H*W <= 512*512
        A_ = CuSparseMatrixCSR( A )
    else
        A_ = A
    end

    z = CUDA.randn(Float32, H, W, ichannel, 1)
    recon = zeros(Float32, H, W)
    
    # let err_prev
    err_best = 100000.f0

    for i=1:niter
        loss_, back = Zygote.pullback( () -> loss(net, z, p_data, A_), ps )
        gs = back(1.0f0)
        Flux.update!(opt, ps, gs)
        
        if i % 50 == 0
            @show i, loss_
        end

        if verbose >= 2 && i > 1000
            z_out = net(z)
            img = cpu(dropdims(z_out, dims=(3,4)))
        
            err = sum(abs.(img_gt - est)) / sum(abs.(gt))
            if err_best > err
                copy!(recon, img)
                
                if (@isdefined dresult)
                    save("$dresult/img_dip_$i.png", recon ./ maximum(recon))
                end
                err_best = err
            end
        end
    end
    return recon
end