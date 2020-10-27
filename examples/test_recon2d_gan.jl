using TomoForward
using FileIO
using Flux
using ImageIO
using SparseArrays
using CUDA

#-------------------------------------
# generate data
#-------------------------------------

# img = Float32.(load("shepp256.png"))
img = Float32.(load("2foam30.png"))
img_gt = copy(img)

H, W = size(img)
detcount = H
nangle = 30
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangle+1)[1:nangle])

if !(@isdefined A)
    A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
    p_data = Float32.(A * vec(img))
    p_DxA = Array(reshape(p_data, nangle, :)')
    p_DxAxB = reshape(p_DxA, size(p_DxA, 1), size(p_DxA, 2), 1)
    A = Float32.(A)
    
    A_cu = CUSPARSE.CuSparseMatrixCSC(A)
end

# yang 2020 paperf
#-------------------------------------
# make networks
#-------------------------------------
include("../src/recon2d_gan.jl")
gen = make_generator(size(p_DxA, 1), size(p_DxA, 2)) |> gpu
dscr = make_discriminator() |> gpu

# G_out = G(p_DxAxB)

opt_gen = ADAM(0.0001)
opt_dscr = ADAM(0.0001)

p_DxAxB = cu(p_DxAxB)
train(p_DxAxB, gen, dscr, opt_gen, opt_dscr)
