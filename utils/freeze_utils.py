def freeze_hrnet_stages(model, train_stages=["STAGE4"]):
    """
    Freezes all HRNet backbone layers except those in `train_stages`.

    """

    for name, param in model.backbone.named_parameters():
        if any(stage in name for stage in train_stages):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    for param in model.mlp.parameters():
        param.requires_grad = True

    print("\n Trainable parameters: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}")