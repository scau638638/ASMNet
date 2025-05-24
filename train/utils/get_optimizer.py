def get_optimizer(optimizer_name, params, learning_rate, l2_weight_decay, momentum):
    if optimizer_name == 'SGD':
        from torch.optim import SGD
        optimizer = SGD(params=params, lr=learning_rate, weight_decay=l2_weight_decay, momentum=momentum)

    elif optimizer_name == 'Adam':
        from torch.optim import Adam
        optimizer = Adam(params=params, lr=learning_rate, weight_decay=l2_weight_decay, betas=(0.9, 0.999))

    elif optimizer_name == 'RAdam':
        from torch.optim import RAdam
        optimizer = RAdam(params=params, lr=learning_rate, weight_decay=l2_weight_decay, betas=(0.9, 0.999))

    elif optimizer_name == 'RMS':
        from torch.optim.rmsprop import RMSprop
        optimizer = RMSprop(params=params, lr=learning_rate, weight_decay=l2_weight_decay)

    elif optimizer_name == 'Lookahead(Adam)':
        from .optimizer_plus.optimizer import Lookahead
        from torch.optim import Adam
        base_optimizer = Adam(params=params, lr=learning_rate, weight_decay=l2_weight_decay)
        optimizer = Lookahead(base_optimizer=base_optimizer)

    else:
        raise NotImplementedError

    return optimizer


def get_loss(loss_name, dataset_name, data_path):
    if loss_name == 'BCE':
        from loss.bce import BCELoss
        loss = BCELoss()
    elif loss_name == 'WBCE':
        from loss.wbce import WBCELoss
        loss = WBCELoss(dataset_name, data_path)
    elif loss_name == 'HED-BCE':
        from loss.hed_bce import HBCELoss
        loss = HBCELoss()
    elif loss_name == 'CustomSigmoidCrossEntropyLoss':
        from loss.CustomSigmoidCrossEntropyLoss import CustomSigmoidCrossEntropyLoss
        loss = CustomSigmoidCrossEntropyLoss()
    elif loss_name == 'wbce_dice_loss':
        from loss.wbce_dice import BCEDiceLoss
        loss = BCEDiceLoss(dataset_name, data_path)
    elif loss_name == 'cross_entropy_loss_RCF':
        from loss.cross_entropy_loss_RCF import cross_entropy_loss_RCF
        loss = cross_entropy_loss_RCF()
    elif loss_name == 'miou_loss':
        from loss.miou_loss import mIoULoss
        loss = mIoULoss()
    elif loss_name == 'focal_tversky_loss':
        from loss.focal_tversky_loss import FocalTverskyLoss
        loss = FocalTverskyLoss()
    else:
        raise NotImplementedError

    return loss
