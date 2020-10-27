using CUDA
using Flux
using NNlib
using Statistics
using Flux: logitbinarycrossentropy

# @with_kw struct HyperParams
#     batch_size::Int = 128
#     latent_dim::Int = 100
#     epochs::Int = 1
#     lr_dscr::Float64 = 0.0002
#     lr_gen::Float64 = 0.0002
# end

Flux.@nograd function normalize_sinogram(x)
    I1 = (x .- mean(x)) ./ std(x)
    return (I1 .- minimum(I1)) ./ (maximum(I1) - minimum(I1))
end

# sinogram -> recon
function make_generator(detpixels, nangle)
    return Chain(
        normalize_sinogram,
        x -> reshape(x, detpixels*nangle, :),
        Dense(detpixels*nangle, 256, softplus),
        Dense(256, 256, softplus),
        Dense(256, 256, softplus),
        Dense(256, detpixels*detpixels, softplus),
        x -> reshape(x, detpixels, detpixels, 1, :),
        # Dense(wsize, wsize), # reshape to 16 x 16
        # LayerNorm(1),
        Conv((3,3), 1 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        Conv((3,3), 64 => 1; pad = 1),
    )
end

function make_discriminator()
    return Chain(
        normalize_sinogram,
        x -> reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)),
        Conv((3,3), 1 => 64, relu; stride=2, pad = 1),
        Conv((3,3), 64 => 128, relu; pad = 1),
        Conv((3,3), 128 => 256, relu; pad = 1),
        Conv((3,3), 256 => 512, relu; pad = 1),
        x -> reshape(x, :, size(x, 4)),
        # Dense(1966080, 1)
    )
end

"See https://github.com/FluxML/model-zoo/blob/master/vision/dcgan_mnist/dcgan_mnist.jl"
function discriminator_loss(real, fake)
    real_loss = mean(logitbinarycrossentropy(real, 1f0))
    fake_loss = mean(logitbinarycrossentropy(fake, 0f0))
    return real_loss + fake_loss
end

generator_loss(fake) = mean(logitbinarycrossentropy(fake, 1f0))

function train_discriminator!(gen, dscr, x, opt_dscr)
    global fake_img
    ps = Flux.params(dscr)
    # Taking gradient
    fake_img = gen(x)
    # fake_p = A_cu * vec(fake_img)
    fake_p = reshape(reshape(A * vec(cpu(fake_img)), nangle, :)', :, nangle, 1)
    fake_p = cu(Array(fake_p))
    loss, back = Flux.pullback(ps) do
        discriminator_loss(dscr(x), dscr(fake_p))
    #    mean(dscr(fake_p))
    end
    grad = back(1f0)
    Flux.update!(opt_dscr, ps, grad)
    return loss
end

function train_generator!(gen, dscr, x, opt_gen)
    ps = Flux.params(gen)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        fake_img = gen(x)
        fake_p = reshape(reshape(A * vec(cpu(fake_img)), nangle, :)', :, nangle, 1)
        fake_p = cu(Array(fake_p))
        generator_loss(dscr(fake_p))
    end
    grad = back(1f0)

    # gs = gradient(ps) do
    #     # generator_loss(dscr(gen(x)))
    #     fake_img = gen(x)
    #     fake_p = reshape(reshape(A * vec(cpu(fake_img)), nangle, :)', :, nangle, 1)
    #     fake_p = cu(Array(fake_p))
    #     return mean(dscr(fake_p))
    # end
    Flux.update!(opt_gen, ps, grad)
    return loss
end

function train(p_DxAxB, gen, dscr, opt_gen, opt_dscr; nepoch=1)
    # Training
    train_steps = 0
    for ep in 1:nepoch
        @info "Epoch $ep"

        loss_dscr = train_discriminator!(gen, dscr, p_DxAxB, opt_dscr)
        # loss_gen = train_generator!(gen, dscr, p_DxAxB, opt_gen)

        # println(loss_dscr, loss_gen)

        # for x in data
        #     # Update discriminator and generator

        #     if train_steps % hparams.verbose_freq == 0
        #         @info("Train step $(train_steps), Discriminator loss = $(loss_dscr), Generator loss = $(loss_gen)")
        #         # Save generated fake image
        #         output_image = create_output_image(gen, fixed_noise, hparams)
        #         save(@sprintf("output/dcgan_steps_%06d.png", train_steps), output_image)
        #     end
        #     train_steps += 1
        # end
    end

end