import torch


def get_loss(opt, Ll1, ssim, image, gt_image, gaussians, radii):
    if opt.reg == 0:
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    elif opt.reg == 1:  # add optical flow loss
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            + opt.lambda_reg * torch.sum(gaussians._motion) / gaussians._motion.shape[0]
        )

    elif opt.reg == 9:  # regularizer on the rotation
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            + opt.lambda_reg * torch.sum(gaussians._omega[radii > 0] ** 2)
        )

    elif opt.reg == 10:  # regularizer on the rotation
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            + opt.lambda_reg * torch.sum(gaussians._motion[radii > 0] ** 2)
        )

    elif opt.reg == 4:
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            + opt.lambda_reg * torch.sum(gaussians.get_scaling) / gaussians._motion.shape[0]
        )

    elif opt.reg == 5:
        loss = Ll1

    elif opt.reg == 6:
        ratio = torch.mean(gt_image) - 0.5 + opt.lambda_dssim
        ratio = torch.clamp(ratio, 0.0, 1.0)
        loss = (1.0 - ratio) * Ll1 + ratio * (1.0 - ssim(image, gt_image))

    elif opt.reg == 7:
        Ll1 = Ll1 / (torch.mean(gt_image) * 2.0)  # normalize L1 loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    elif opt.reg == 8:
        N = gaussians._xyz.shape[0]
        mean = torch.mean(gaussians._xyz, dim=0, keepdim=True)
        variance = (mean - gaussians._xyz) ** 2  # / N
        loss = (1.0 - opt.lambda_dssim) * Ll1 + 0.0002 * opt.lambda_dssim * torch.sum(variance) / N

    return loss
