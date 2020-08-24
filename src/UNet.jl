# This code is modified from
# https://github.com/chinglamchoi/julia_unet/

using CUDA
using CUDA.CUSPARSE
using Flux

block1(in_channels, features) = Chain(Conv((3,3), in_channels=>features, pad=1),
    BatchNorm(features, relu), #calls n-1'th dim
    Conv((3,3), features=>features, pad=1),
    BatchNorm(features, relu)) 

block2(in_channels, features) = Chain(MaxPool((2,2), stride=2),
    Conv((3,3), in_channels=>features, pad=1),
    BatchNorm(features, relu), #calls n-1'th dim
    Conv((3,3), features=>features, pad=1),
    BatchNorm(features, relu)) 

include("upsample.jl")

upsample(x) = bilinear_upsample2d(x, (2,2))
# upsample(x) = upsample_nearest(x)
upconv(in_channels, features) = Conv((3,3), in_channels=>features, pad=1)
# upconv(in_channels, features) = ConvTranspose((2,2), in_channels=>features, stride=2)
conv(in_channels, out_channels) = Conv((1,1), in_channels=>out_channels)

struct UNet
    conv_block
    conv_block2
    bottle
    upconv_block
    conv_
    use_two_skip
end

Flux.@functor UNet

function UNet(ichannel=1; cmul=8, use_two_skip=true) # original Unet : cmul=32
    conv_block = (block1(ichannel, cmul), block2(cmul, cmul*2), block2(cmul*2, cmul*4), block2(cmul*4, cmul*8))
    
    if use_two_skip == false
        conv_block2 = (block1(cmul*16, cmul*8), block1(cmul*8, cmul*4), block1(cmul*4, cmul*2), block1(cmul*2, cmul))
    else
        conv_block2 = (block1(cmul*16, cmul*8), block1(cmul*8, cmul*4), block1(cmul*2, cmul*2), block1(cmul, cmul))
    end

    bottle = block2(cmul*8, cmul*16)
    upconv_block = (upconv(cmul*16, cmul*8), upconv(cmul*8, cmul*4), upconv(cmul*4, cmul*2), upconv(cmul*2, cmul))
    conv_ = conv(cmul, 1)
    UNet(conv_block, conv_block2, bottle, upconv_block, conv_, use_two_skip) |> gpu
end

function (u::UNet)(x)
    enc1 = u.conv_block[1](x) # |> gpu
    enc2 = u.conv_block[2](enc1) # |> gpu
    enc3 = u.conv_block[3](enc2) # |> gpu
    enc4 = u.conv_block[4](enc3) # |> gpu
    
    bn = u.bottle(enc4) # |> gpu
	bn = upsample(bn)
    dec4 = u.upconv_block[1]( bn ) # |> gpu
    dec4 = cat(dims=3, dec4, enc4) # |> gpu
    dec4 = u.conv_block2[1](dec4) # |> gpu
    
    dec4 = upsample(dec4)
    dec3 = u.upconv_block[2]( dec4 ) # |> gpu
    dec3 = cat(dims=3, dec3, enc3) # |> gpu
    dec3 = u.conv_block2[2](dec3) # |> gpu
    
    dec3 = upsample(dec3)
    dec2 = u.upconv_block[3]( dec3 ) # |> gpu
    
    if u.use_two_skip == false
        dec2 = cat(dims=3, dec2, enc2) # |> gpu
    end
    
    dec2 = u.conv_block2[3](dec2) # |> gpu
    dec2 = upsample(dec2)
    dec1 = u.upconv_block[4]( dec2 ) # |> gpu
    
    if u.use_two_skip == false
        dec1 = cat(dims=3, dec1, enc1) # |> gpu
    end
    
    dec1 = u.conv_block2[4](dec1) # |> gpu
    dec1 = u.conv_(dec1) # |> gpu
    dec1 = relu.(dec1)
    # dec1 = sigmoid.(dec1)
end
