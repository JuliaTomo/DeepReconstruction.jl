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
lr = 0.01f0

net = UNet(3; use_two_skip=true) # 128, 5
opt = ADAM(lr) # Gadelha
dresult = "../result/"
mkpath(dresult)
u_out, losses, u_best, errs = recon2d_dip(net, opt, p_data, Float32.(A), H, W; img_gt=img_gt)

using Plots
# plot(errs)
# savefig("$dresult/errs.png")

# plot(losses, xaxis=:log)
# savefig("$dresult/losses.png")

save("$dresult/recon.png", u_out ./ maximum(u_out))
save("$dresult/recon_best.png", u_best ./ maximum(u_best)) # save the output

#----------------------
# noisy case
#----------------------

using Random
using LinearAlgebra
function impose_gaussian_noise(arr, eta)
    Random.seed!(1)
    e_hat = randn(Float32, size(arr))
    e_hat = eta .* e_hat * (norm(vec(arr)) / norm(vec(e_hat)))
    noised = max.(arr + e_hat, 0.f0)
    return noised
end

p_data = impose_gaussian_noise(p_data, 0.02)
net = UNet(3; use_two_skip=true) # 128, 5
opt = ADAM(lr) # Gadelha
u_out_noisy, losses_noisy, u_best_noisy, errs_noisy = recon2d_dip(net, opt, p_data, Float32.(A), H, W; img_gt=img_gt)

plot(errs[100:end], label="clean", title="reconstruction error")
plot!(errs_noisy[100:end], label="noisy")
savefig("$dresult/errs.png")

plot(losses, xaxis=:log, label="clean", title="loss")
plot!(losses_noisy, label="noisy")
savefig("$dresult/losses.png")

save("$dresult/recon_noisy.png", u_out_noisy ./ maximum(u_out_noisy))
save("$dresult/recon_best_noisy.png", u_best_noisy ./ maximum(u_best_noisy)) # save the output


## impose noise and check it