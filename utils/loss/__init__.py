import functools

from utils.loss.loss import (
    cross_entropy2d
)


key2loss = {
    "cross_entropy": cross_entropy2d
}


def get_loss_function(loss_fc, **loss_params):
    if loss_fc is None:
        print("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_name = loss_fc
        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        print("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
