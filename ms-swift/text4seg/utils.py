from .segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch
from enum import Enum
import torch.distributed as dist
from torchvision.ops.boxes import box_area
import re
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def translate_sequence(sequence_str):
    """
    Translates a comma-separated sequence of categorical data to numerical labels,
    identifying categories from the sequence.

    Parameters:
    sequence_str (str): The comma-separated sequence of categorical data.

    Returns:
    list: The sequence of numerical labels.
    """
    # Split the string into a list of categories
    sequence = sequence_str.split('|')

    # strip the whitespace from each category
    sequence = [seq.strip() for seq in sequence]

    # Identify unique categories from the sequence
    unique_categories = list(dict.fromkeys(sequence))

    # place "others" at the beginning of the list
    if "others" in unique_categories:
        unique_categories.remove("others")
        unique_categories.insert(0, "others")

    # Create a dictionary to map each category to a unique integer
    category_to_label = {category: idx for idx, category in enumerate(unique_categories)}

    # Translate the sequence using the dictionary
    translated_sequence = [category_to_label[item] for item in sequence]

    return translated_sequence


def decode_mask(encoded_str):
    rows = encoded_str.strip("\n").split("\n ")
    decoded_list = []
    for row in rows:
        tokens = row.split("| ")
        for token in tokens:
            label, count = token.split(" *")
            decoded_list.extend([label] * int(count))
    return "|".join(decoded_list)


def decode_mask_s(encoded_str):
    rows = encoded_str.strip("\n")
    decoded_list = []
    tokens = rows.split("| ")
    for token in tokens:
        label, count = token.split(" *")
        decoded_list.extend([label] * int(count))
    return "|".join(decoded_list)


def decode_mask_l(encoded_str):
    rows = encoded_str.strip("\n")
    tokens = rows.split("| ")
    return "|".join(tokens)


def compute_logits_from_mask(mask, eps=1e-3):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    # resize to the expected mask shape of SAM (256x256)
    assert logits.ndim == 2
    expected_shape = (256, 256)

    if logits.shape == expected_shape:  # shape matches, do nothing
        pass

    elif logits.shape[0] == logits.shape[1]:  # shape is square
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])

    else:  # shape is not square
        # resize the longest side to expected shape
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])

        # pad the other side
        h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
        pad_width = ((0, padh), (0, padw))
        logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

    logits = logits[None]
    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits


def masks_sample_points(masks):
    """Sample points on mask
    """
    masks = masks.unsqueeze(0)
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    k = np.random.randint(10, 11)
    samples_pos = []
    for b_i in range(len(masks)):
        select_mask = (masks[b_i] > 0.5)
        x_idx = torch.masked_select(x, select_mask)
        y_idx = torch.masked_select(y, select_mask)

        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:, None], samples_y[:, None]), dim=1)
        samples_pos.append(samples_xy)

    samples_pos = torch.cat(samples_pos)

    k = np.random.randint(10, 11)
    samples_neg = []
    for b_i in range(len(masks)):
        select_mask = (masks[b_i] < 0.5)
        x_idx = torch.masked_select(x, select_mask)
        y_idx = torch.masked_select(y, select_mask)

        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:, None], samples_y[:, None]), dim=1)
        samples_neg.append(samples_xy)

    samples_neg = torch.cat(samples_neg)

    # get the point labels
    point_labels = np.concatenate(
        [
            np.ones(len(samples_pos), dtype="uint8"),
            np.zeros(len(samples_neg), dtype="uint8"),
        ], axis=0
    )
    point_coords = np.concatenate([samples_pos, samples_neg], axis=0).astype("float64")

    return point_coords, point_labels


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def crf_refine(image, labels, n_labels=2, gt_prob=0.9):
    """
    Refine a coarse mask using DenseCRF.

    :param image: Input image (HxWx3)
    :param labels: Coarse mask (HxW) where 0=background, 1=first class, 2=second class, ...
    :return: Refined mask
    """
    # Create the DenseCRF model
    # image = image.transpose((1,2,0))
    h, w = image.shape[:2]
    image = image.astype(np.uint8)
    # print(h, w)
    # print(image.shape)
    # print(labels.shape)
    labels = labels.flatten()
    d = dcrf.DenseCRF2D(w, h, n_labels)

    # Get unary potentials (negative log probabilities)
    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(unary)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Inference
    refined_mask = d.inference(5)  # Number of iterations
    refined_mask = np.argmax(refined_mask, axis=0)
    refined_mask = refined_mask.reshape(h, w)

    return refined_mask