import copy
import math
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import BatchNorm2d
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear, LoRALinearLayer, LoRAConv2dLayer

from models.layers.batchnorm2d import RandBatchNorm2d
from models.layers.conv2d import RandConv2d
from models.layers.linear import RandLinear
from models.layers.layernorm import RandLayerNorm
from models.layers.groupnorm import RandGroupNorm

from models.layers.lora_layer import LoRACompatibleRandConv, LoRACompatibleRandLinear, LoRARandLinearLayer
from oft_utils.attention_processor import OFTRandAttnProcessor

convert_params = 0.0
overall_params = 0.0
kl_count = 0
module_count = 0


def set_sigma_module_for_unet(module, sigma_blocks):
    """
    Args:
        module: nn.Module, current module
        sigma_blocks: a list as shape [a, b, c] where a, b, c are sigma for upblock, midblock and downblock
    Returns:
        new module with same structure as module and parameter as corresponding sigma
    """
    new_module = copy.deepcopy(module)
    for i, key in enumerate(new_module._modules):
        print(key, i)
        for param in new_module._modules[key].parameters():
            with torch.no_grad():
                param.fill_(sigma_blocks[i])
    return new_module


def add_bayesian_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--convert_conv",
        action="store_false",
        dest="skip_conv",
        default=True
    )
    parser.add_argument(
        "--convert_linear",
        action="store_false",
        dest="skip_linear",
        default=True
    )
    parser.add_argument(
        "--convert_attn2",
        action="store_false",
        dest="skip_attn2",
        default=True
    )
    parser.add_argument(
        "--convert_ff",
        action="store_false",
        dest="skip_ff",
        default=True
    )
    parser.add_argument(
        "--convert_time",
        action="store_false",
        dest="skip_time",
        default=True
    )
    parser.add_argument(
        "--convert_layernorm",
        action="store_false",
        dest="skip_ln",
        default=True
    )
    parser.add_argument(
        "--convert_groupnorm",
        action="store_false",
        dest="skip_gn",
        default=True
    )
    parser.add_argument(
        "--convert_up_block",
        action="store_false",
        dest="skip_up_block",
        default=True
    )
    parser.add_argument(
        "--convert_down_block",
        action="store_false",
        dest="skip_down_block",
        default=True
    )
    parser.add_argument(
        "--convert_mid_block",
        action="store_false",
        dest="skip_mid_block",
        default=True
    )
    parser.add_argument(
        "--convert_subblock_0",
        action="store_false",
        dest="skip_subblock_0",
        default=True
    )
    parser.add_argument(
        "--convert_subblock_1",
        action="store_false",
        dest="skip_subblock_1",
        default=True
    )
    parser.add_argument(
        "--convert_subblock_2",
        action="store_false",
        dest="skip_subblock_2",
        default=True
    )
    parser.add_argument(
        "--convert_subblock_3",
        action="store_false",
        dest="skip_subblock_3",
        default=True
    )
    parser.add_argument(
        "--bayes_only",
        action="store_true",
    )
    parser.add_argument(
        "--init_sigma",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--prior_sigma",
        type=float,
        default=0.02,
    )
    # parser.add_argument(
    #     "--multiple_sigma_lr",
    #     type=float,
    #     default=1.0,
    # )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--init_mu_from_module",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--test_sigma",
        type=float,
        default=-1.0
    )


def convert_with_config(module, args):
    return convert(module, init_sigma=args.init_sigma, init_mu_from_module=args.init_mu_from_module,
                   skip_Conv=args.skip_conv, skip_Linear=args.skip_linear, skip_attn2=args.skip_attn2,
                   skip_time=args.skip_time, skip_ln=args.skip_ln, skip_gn=args.skip_gn,
                   skip_mid_block=args.skip_mid_block, skip_up_block=args.skip_up_block,
                   skip_down_block=args.skip_down_block, skip_ff=args.skip_ff,
                   skip_subblock_0=args.skip_subblock_0, skip_subblock_1=args.skip_subblock_1,
                   skip_subblock_2=args.skip_subblock_2, skip_subblock_3=args.skip_subblock_3, args=args)


def convert(module, init_sigma=-10, init_mu_from_module=True, skip_Conv=True, skip_Linear=False, skip_attn2=True,
            skip_time=True, skip_ln=True, skip_gn=True, skip_up_block=False, skip_mid_block=False,
            skip_down_block=False, skip_ff=False, current_key=None, skip_subblock_0=None, skip_subblock_1=None,
            skip_subblock_2=None, skip_subblock_3=None, args=None):
    global convert_params, overall_params, module_count
    is_base = not any(module.children())
    init_sigma_from_module = isinstance(init_sigma, torch.nn.Module)
    if is_base or isinstance(module, LoRALinearLayer):
        type_list = ['RandConv2d', 'RandLinear', 'RandConv2d', 'RandLinear', 'RandBatchNorm2d', 'RandLayerNorm',
                     'RandGroupNorm']
        overall_params += sum(p.numel() for p in module.parameters())
        if isinstance(module, LoRACompatibleConv):
            if skip_Conv:
                print(f'Skipping Conv, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            if module.lora_layer is not None:
                raise ValueError("LORA NOT SUPPORTED YET")
            bayesian_module = RandConv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                         kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                         dilation=module.dilation, groups=module.groups, bias=module.bias is not None,
                                         init_s=0.0 if init_sigma_from_module else init_sigma)
            module_count += 1
            convert_params += sum(p.numel() for p in module.parameters())
            type_id = 0
        elif isinstance(module, LoRALinearLayer):
            bayesian_module = LoRARandLinearLayer(in_features=module.in_features,
                                                  out_features=module.out_features,
                                                  rank=module.rank,
                                                  init_s=init_sigma,
                                                  network_alpha=module.network_alpha)
            if init_mu_from_module:
                bayesian_module.up.mu_weight.data.copy_(module.up.weight.data)
                bayesian_module.down.weight.data.copy_(module.down.weight.data)
            if init_sigma_from_module:
                bayesian_module.up.sigma_weight.data.copy_(init_sigma.up.weight.data)
            return bayesian_module
        elif isinstance(module, LoRACompatibleLinear):
            if skip_Linear:
                print(f'Skipping Linear, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            if module.lora_layer is not None:
                raise ValueError("LORA NOT SUPPORTED YET")
            else:
                bayesian_module = RandLinear(in_features=module.in_features, out_features=module.out_features,
                                             bias=module.bias is not None,
                                             init_s=0.0 if init_sigma_from_module else init_sigma)
                module_count += 1
                convert_params += sum(p.numel() for p in module.parameters())
                type_id = 1
        elif isinstance(module, torch.nn.Conv2d):
            if skip_Conv:
                print(f'Skipping Conv, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            bayesian_module = RandConv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                         kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                         dilation=module.dilation, groups=module.groups, bias=module.bias is not None,
                                         init_s=0.0 if init_sigma_from_module else init_sigma)
            module_count += 1
            convert_params += sum(p.numel() for p in module.parameters())
            type_id = 2
        elif isinstance(module, torch.nn.Linear):
            if skip_Linear:
                print(f'Skipping Linear, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            bayesian_module = RandLinear(in_features=module.in_features, out_features=module.out_features,
                                         bias=module.bias is not None,
                                         init_s=0.0 if init_sigma_from_module else init_sigma)
            module_count += 1
            convert_params += sum(p.numel() for p in module.parameters())
            type_id = 3
        elif isinstance(module, torch.nn.BatchNorm2d):
            raise NotImplementedError
            bayesian_module = RandBatchNorm2d(num_features=module.num_features, eps=module.eps,
                                              momentum=module.momentum,
                                              affine=module.affine, track_running_stats=module.track_running_stats,
                                              init_s=0.0 if init_sigma_from_module else init_sigma)
            module_count += 1
            bayesian_module.running_mean.data.copy_(module.running_mean.data)
            bayesian_module.running_var.data.copy_(module.running_var.data)
            bayesian_module.num_batches_tracked.data.copy_(module.num_batches_tracked.data)
            convert_params += sum(p.numel() for p in module.parameters())
            type_id = 4
        elif isinstance(module, torch.nn.LayerNorm):
            if skip_ln:
                print(f'Skipping LayerNorm, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            bayesian_module = RandLayerNorm(normalized_shape=module.normalized_shape, eps=module.eps,
                                            elementwise_affine=module.elementwise_affine, bias=module.bias is not None,
                                            init_s=0.0 if init_sigma_from_module else init_sigma)
            module_count += 1
            convert_params += sum(p.numel() for p in module.parameters())
            type_id = 5
        elif isinstance(module, torch.nn.GroupNorm):
            if skip_gn:
                print(f'Skipping GroupNorm, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            bayesian_module = RandGroupNorm(num_groups=module.num_groups, num_channels=module.num_channels,
                                            eps=module.eps, affine=module.affine,
                                            init_s=0.0 if init_sigma_from_module else init_sigma)
            module_count += 1
            convert_params += sum(p.numel() for p in module.parameters())
            type_id = 6
        else:
            return copy.deepcopy(module)  # not a layer to be converted into Bayesian
        if init_mu_from_module:
            bayesian_module.mu_weight.data.copy_(module.weight.data)
            if module.bias is not None:
                bayesian_module.mu_bias.data.copy_(module.bias.data)
        if init_sigma_from_module:
            bayesian_module.sigma_weight.data.copy_(init_sigma.weight.data)
            if module.bias is not None:
                bayesian_module.sigma_bias.data.copy_(init_sigma.bias.data)
        return bayesian_module

    else:
        new_module = copy.deepcopy(module)
        for key in module._modules:
            if (skip_mid_block and key == 'mid_block') or (skip_up_block and key == 'up_blocks') or (
                    skip_down_block and key == 'down_blocks'):
                print(f'Skipping {key}, params: {sum(p.numel() for p in module._modules[key].parameters())}')
                overall_params += sum(p.numel() for p in module._modules[key].parameters())
                continue
            if skip_attn2:
                if key == 'attn2':
                    print(f'Skipping {key}, params: {sum(p.numel() for p in module._modules[key].parameters())}')
                    overall_params += sum(p.numel() for p in module._modules[key].parameters())
                    continue
            if skip_ff:
                if key == 'ff':
                    print(f'Skipping {key}, params: {sum(p.numel() for p in module._modules[key].parameters())}')
                    overall_params += sum(p.numel() for p in module._modules[key].parameters())
                    continue
            if skip_time:
                if key == 'time_emb_proj':
                    print(f'Skipping {key}, params: {sum(p.numel() for p in module._modules[key].parameters())}')
                    overall_params += sum(p.numel() for p in module._modules[key].parameters())
                    continue
                if key == 'time_embedding':
                    print(f'Skipping {key}, params: {sum(p.numel() for p in module._modules[key].parameters())}')
                    overall_params += sum(p.numel() for p in module._modules[key].parameters())
                    continue
            if (current_key == 'up_blocks') or (current_key == 'down_blocks'):
                if (skip_subblock_0 and key == '0') or (skip_subblock_1 and key == '1') or (
                        skip_subblock_2 and key == '2') or (skip_subblock_3 and key == '3'):
                    print(
                        f'Skipping {current_key}:{key}, params: {sum(p.numel() for p in module._modules[key].parameters())}')
                    overall_params += sum(p.numel() for p in module._modules[key].parameters())
                    continue

            new_module._modules[key] = convert(module._modules[key],
                                               init_sigma=init_sigma._modules[
                                                   key] if init_sigma_from_module else init_sigma,
                                               init_mu_from_module=init_mu_from_module,
                                               skip_Conv=skip_Conv,
                                               skip_Linear=skip_Linear,
                                               skip_attn2=skip_attn2,
                                               skip_ff=skip_ff,
                                               skip_time=skip_time,
                                               skip_ln=skip_ln,
                                               skip_gn=skip_gn,
                                               skip_up_block=skip_up_block,
                                               skip_down_block=skip_down_block,
                                               skip_mid_block=skip_mid_block,
                                               skip_subblock_0=skip_subblock_0,
                                               skip_subblock_1=skip_subblock_1,
                                               skip_subblock_2=skip_subblock_2,
                                               skip_subblock_3=skip_subblock_3,
                                               current_key=key,
                                               args=args)
            # if module_count >= 1:
            #     return new_module
        return new_module


def convert_reverse(module):
    is_base = not any(module.children())
    if is_base or isinstance(module, LoRARandLinearLayer):
        if isinstance(module, LoRACompatibleRandConv):
            raise NotImplementedError
        elif isinstance(module, LoRARandLinearLayer):
            non_bayesian_lora_layer = LoRALinearLayer(module.in_features, module.out_features,
                                                      rank=module.rank,
                                                      network_alpha=module.network_alpha)
            non_bayesian_lora_layer.up.weight.copy_(module.up.mu_weight)
            non_bayesian_lora_layer.down.weight.copy_(module.down.weight)
            non_bayesian_module = non_bayesian_lora_layer
            return non_bayesian_module
        elif isinstance(module, RandLinear):
            raise NotImplementedError
            # non_bayesian_module = torch.nn.Linear(module.in_features, module.out_features,
            #                                       bias=module.mu_bias is not None)
        elif isinstance(module, RandBatchNorm2d):
            raise NotImplementedError
        elif isinstance(module, RandLayerNorm):
            raise NotImplementedError
        elif isinstance(module, RandGroupNorm):
            raise NotImplementedError
        else:
            return copy.deepcopy(module)  # not a layer to be converted into Bayesian

        non_bayesian_module.weight.data.copy_(module.mu_weight.data)
        if module.mu_bias is not None:
            non_bayesian_module.bias.data.copy_(module.mu_bias.data)
        return non_bayesian_module

    else:
        new_module = copy.deepcopy(module)
        for key in module._modules:
            new_module._modules[key] = convert_reverse(module._modules[key])
        return new_module


def cal_KL(mu1, sigma1, mu2, sigma2):
    """
    KL(P1 || P2), P1 ~ N(mu1, exp(logsigma1)^2)
    :param mu1:
    :param logsigma1: (prior)
    :param mu2:
    :param logsigma2:
    :return:
    """
    # print(mu1, logsigma1, mu2, logsigma2)
    kl1 = torch.log(sigma2 / sigma1)

    kl2 = - 0.5 + 0.5 * (sigma1 / sigma2) ** 2
    kl3 = 0.5 * (mu1 - mu2) ** 2 / (sigma2 ** 2)
    kl = kl1 + kl2 + kl3
    # kl = logsigma2 + (torch.exp(2 * logsigma1) + (mu1 - mu2) ** 2) / (2 * torch.exp(2 * logsigma2))
    if torch.isnan(kl).any():
        print(sigma2, sigma1)
        print(kl1, kl2, mu1, mu2)
        exit()
    # print(kl1.mean(), kl2.mean(), kl3.mean())
    if (kl < -1e-7).any():
        print("kl < 0")
        pos = torch.where(kl < 0)
        kl[pos] *= 0.0
    res = kl.sum()
    # if res < 0:
    #     print(logsigma1, logsigma2, mu1, mu2)
    return res


def cal_KL_modules(curr_module, prior_module, prior_sigma):
    """
    Args:
        curr_module: nn.Module, current module
        prior_module: nn.Module, prior module
        prior_sigma: torch.tensor or nn.Module
    Returns:
    """
    global kl_count
    is_base = not any(curr_module.children())
    prior_sigma_from_module = isinstance(prior_sigma, torch.nn.Module)
    prior_mu_from_module = isinstance(prior_module, torch.nn.Module)
    if is_base or isinstance(curr_module, LoRARandLinearLayer) or isinstance(curr_module, OFTRandAttnProcessor):
        # type_list = ['LoRACompatibleRandConv', 'LoRACompatibleRandLinear', 'RandConv2d', 'RandLinear', 'RandBatchNorm2d']
        if isinstance(curr_module, LoRARandLinearLayer):
            kl_weight1 = cal_KL(0.0,
                                prior_sigma.weight.detach() if prior_sigma_from_module else prior_sigma.detach(),
                                curr_module.up.mu_weight, curr_module.up.sigma_weight)
            return kl_weight1
        elif isinstance(curr_module, OFTRandAttnProcessor):
            kl_weight = 0.0
            for layer in [curr_module.to_k_oft, curr_module.to_q_oft, curr_module.to_v_oft, curr_module.to_out_oft]:
                kl_weight += cal_KL(0.0,
                                   prior_sigma.weight.detach() if prior_sigma_from_module else prior_sigma.detach(),
                                   layer.mu_R, layer.sigma_R)
            return kl_weight
        elif (isinstance(curr_module, RandConv2d) or isinstance(curr_module, RandLinear) or
              isinstance(curr_module, RandBatchNorm2d) or isinstance(curr_module, RandLayerNorm)
              or isinstance(curr_module, RandGroupNorm)):
            kl_count += 1
            kl_weight = cal_KL(prior_module.weight.detach(),
                               prior_sigma.weight.detach() if prior_sigma_from_module else prior_sigma.detach(),
                               curr_module.mu_weight, curr_module.sigma_weight)
            if curr_module.mu_bias is not None:
                kl_bias = cal_KL(prior_module.bias.detach(),
                                 prior_sigma.bias.detach() if prior_sigma_from_module else prior_sigma.detach(),
                                 curr_module.mu_bias,
                                 curr_module.sigma_bias)
            else:
                kl_bias = 0
            return kl_weight + kl_bias
        else:
            return 0.0  # not a layer to be converted into Bayesian
    else:
        kl = torch.tensor(0.0, device="cuda")
        for key in curr_module._modules:
            kl += cal_KL_modules(curr_module=curr_module._modules[key],
                                 prior_module=prior_module._modules[key] if prior_mu_from_module else prior_module,
                                 prior_sigma=prior_sigma._modules[key] if prior_sigma_from_module else prior_sigma)
        return kl


def reset_sigma(curr_module: torch.nn.Module, sigma: float) -> None:
    if sigma < 0.0:
        raise ValueError('sigma must be positive')
    is_base = not any(curr_module.children())
    if is_base:
        if (isinstance(curr_module, RandConv2d) or isinstance(curr_module, RandLinear) or
                isinstance(curr_module, RandBatchNorm2d) or isinstance(curr_module, RandLayerNorm)
                or isinstance(curr_module, RandGroupNorm)):
            curr_module.sigma_weight.fill_(sigma)
            if curr_module.sigma_bias is not None:
                curr_module.sigma_bias.fill_(sigma)
    else:
        for key in curr_module._modules:
            reset_sigma(curr_module._modules[key], sigma=sigma)


if __name__ == '__main__':
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).cuda()
    # bmodel = convert(model, init_sigma=-1).cuda()
    # test_inp = torch.randn(100, 3, 32, 32).cuda()
    # bmodel.eval()
    # model.eval()
    # m1_res = model(test_inp)
    # bm1_res = bmodel(test_inp)
    # assert torch.allclose(m1_res, bm1_res)
    # print("Finished")
    #
    # print(cal_KL_modules(curr_module=bmodel, prior_module=model, prior_sigma=model))
    from diffusers import DiffusionPipeline, UNet2DConditionModel
    import torch

    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", dtype=torch.float16, safety_checker=None
    )

    print(pipe.unet)
    for key in pipe.unet._modules:
        print(key, sum(p.numel() for p in pipe.unet._modules[key].parameters()))
    # new_unet = convert(pipe.unet, skip_up_block=True, skip_down_block=True)
    new_unet = convert(pipe.unet, skip_mid_block=True, skip_down_block=True, skip_subblock_0=True,
                       skip_subblock_1=False, skip_subblock_2=False, skip_subblock_3=False)
    print(convert_params / overall_params)
