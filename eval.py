import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, valloader, loss_fc, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.long
    n_val = len(valloader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for images_val, labels_val, f_name in valloader:
            imgs, true_masks = images_val, labels_val
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            tot += loss_fc(input=mask_pred, target=true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val
