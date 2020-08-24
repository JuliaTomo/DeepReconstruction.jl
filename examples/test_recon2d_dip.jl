using TomoForward
using DeepImagePrior
using FileIO
using Flux
using ImageIO

img = Float32.(load("shepp256.png"))
img_gt = copy(img)

H, W = size(img)
detcount = H
nangles = 30
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
p_data = Float32.(A * vec(img))
# p_data = vec(reshape(p_data, nangles, :)') # detector count should be the first axis

net = UNet(3; use_two_skip=true) # 128, 5
net = net |> gpu
opt = ADAM(0.01f0) # Gadelha
dresult = "../result/"
mkpath(dresult)
u_out, u_best, errs = recon2d_dip(net, opt, p_data, Float32.(A), H, W; img_gt=img_gt, dresult=dresult)

using Plots
plot(errs)
savefig("$dresult/errs.png")

save("$dresult/recon.png", u_out ./ maximum(u_out))
save("$dresult/recon_best.png", u_best ./ maximum(u_best)) # save the output


## impose noise and check it