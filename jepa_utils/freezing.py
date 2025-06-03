def freeze_non_diffusion(model, *, train_bn=False):
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(("diffusion.", "diffusion_model."))
    if not train_bn:
        for m in model.modules():
            if m.__class__.__name__.startswith("BatchNorm"):
                m.eval()


def freeze_jepa(model, *, train_bn=False):
    for n, p in model.named_parameters():
        p.requires_grad = not n.startswith(("diffusion.", "diffusion_model."))
    if not train_bn:
        for m in model.modules():
            if m.__class__.__name__.startswith("BatchNorm"):
                m.eval()

