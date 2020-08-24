using CUDA
using CUDA.CUSPARSE
using Flux
using SparseArrays
using FileIO

function fp_dip(x, A)
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
- verbose : 1 if you want to save the best result during the iteration instead of early stopping. There should be a global variable `img_gt` for the ground truth image. Moreover, if there is global variable `dresult`, we save the image.
"""
function recon2d_dip(net, opt, p_data, A::SparseMatrixCSC{Float32,Int64}, H::Int, W::Int, niter=2000, ichannel=3; img_gt=nothing, dresult=nothing)

    # if there is no img_gt, we don't compare it.
    if ~isnothing(img_gt)
        errs = zeros(Float32, niter)
        err_best = 100000.f0
    end

    ps = Flux.params(net)
    p_data = cu(vec(p_data))

    # if the image is small enough, we can use CUDA sparse array
    if H*W <= 512*512
        A_ = CuSparseMatrixCSR( A )
    else
        A_ = A
    end

    z = CUDA.randn(Float32, H, W, ichannel, 1)
    img_best = zeros(Float32, H, W)
    

    for i=1:niter
        loss_, back = Zygote.pullback( () -> loss(net, z, p_data, A_), ps )
        gs = back(1.0f0)
        Flux.update!(opt, ps, gs)
        
        if i % 50 == 0
            @show i, loss_
        end

        if ~isnothing(img_gt)
            z_out = net(z)
            img = cpu(dropdims(z_out, dims=(3,4)))
        
            errs[i] = sum(abs.(img_gt - img)) / sum(abs.(img_gt))

            if i > 1000 && err_best > errs[i]
                copy!(img_best, img)
                err_best = errs[i]

                if ~isnothing(dresult)
                    save("$dresult/img_dip_$i.png", img_best ./ maximum(img_best))
                end
            end
        end
    end
    z_out = net(z)
    img_final = cpu(dropdims(z_out, dims=(3,4)))
    
    if isnothing(img_gt)
        return img_final
    else
        return img_final, img_best, errs
    end
end