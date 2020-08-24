using TomoForward
using DeepImagePrior
using FileIO
using Flux

img = Float32.(load("shepp256.png"))
img_gt = copy(img)

H, W = size(img)
detcount = H
nangles = 30
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
p_data = Float32.(A * vec(img))

net = UNet(3; use_two_skip=true) # 128, 5
opt = ADAM(0.01f0) # Gadelha
u_dip = recon2d_dip(net, opt, p_data, Float32.(A), H, W)

# save("recon.png", u_dip") # save the output

## impose noise and check it