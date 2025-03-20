import hashlib
from collections import OrderedDict
import re
import os
import torch.distributed as dist
import torch
import torch.nn.functional as F


def remove_module_in_state_dict(state_dict):
    # Either the whole is in DDP or None, no in-between
    model_dict = OrderedDict()
    pattern = re.compile("module.")
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict = state_dict
    return model_dict


def get_ema_checkpoint(state_dict):
    # Note that EMA checkpoints contain both the model and the EMA model
    model_dict = OrderedDict()
    pattern = re.compile("ema_model.")
    for k, v in state_dict.items():
        if re.search("ema_model", k):
            model_dict[re.sub(pattern, "", k)] = v
    if len(model_dict):
        print("Using EMA checkpoint.")
        return model_dict
    return state_dict


def print0(*args, **kwargs):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        print(*args, **kwargs)


def running_in_ddp():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size() > 1
    else:
        return False


def is_rank_0():
    return not running_in_ddp() or os.environ.get("LOCAL_RANK", 0) == "0"


def seed_from_subject_id(subject_id: str):
    hash = int(hashlib.sha256(subject_id.encode()).hexdigest(), 16) % (2**64)
    return hash


def image_gradient_2d(x, return_components=False):
    """
    Compute the gradient of an image using a Sobel operator.
    """
    # Define filters
    dx = (
        torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(x.device)
    )
    dy = dx.permute(0, 1, 3, 2)

    # Filter
    x_dx = F.conv2d(x, dx, padding=1)
    x_dy = F.conv2d(x, dy, padding=1)

    magn = torch.sqrt(x_dx**2 + x_dy**2)
    if return_components:
        return magn, x_dx, x_dy
    else:
        return magn


def image_gradient_3d(x, return_components=False):
    """
    Compute the gradient of an image using a Sobel operator in 3D
    """
    # Define filters
    dx = (
        torch.tensor(
            [
                [
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                ]
            ],
            dtype=torch.float32,
        )
        .view(1, 1, 3, 3, 3)
        .to(x.device)
    )
    dy = dx.permute(0, 1, 3, 2, 4)
    dz = dx.permute(0, 1, 4, 3, 2)

    # Filter
    x_dx = F.conv3d(x, dx, padding=1)
    x_dy = F.conv3d(x, dy, padding=1)
    x_dz = F.conv3d(x, dz, padding=1)
    magn = torch.sqrt(x_dx**2 + x_dy**2 + x_dz**2)
    if return_components:
        return magn, x_dx, x_dy, x_dz
    else:
        return magn


@torch.no_grad()
def close_mask(mask, dilate_size=7, erode_size=5):
    org_shape, org_dtype = mask.shape, mask.dtype
    mask = (mask > 0).float()

    # dilate
    filter = (
        torch.ones((dilate_size, dilate_size, dilate_size)).unsqueeze(0).unsqueeze(0)
    )
    conv_res = torch.nn.functional.conv3d(
        mask.unsqueeze(0), filter, padding=dilate_size // 2
    )
    dil = conv_res > 0

    # erode
    filter = torch.ones((erode_size, erode_size, erode_size)).unsqueeze(0).unsqueeze(0)
    conv_res = torch.nn.functional.conv3d(
        dil.float(), filter, padding=erode_size // 2
    ).squeeze(dim=0)
    erode = conv_res == filter.sum()

    assert erode.shape == org_shape

    return erode.to(org_dtype)


@torch.no_grad()
def erode(mask, erode_size=3):
    org_shape, org_dtype = mask.shape, mask.dtype
    mask = (mask > 0).float()

    # erode
    filter = torch.ones((erode_size, erode_size, erode_size)).unsqueeze(0).unsqueeze(0)
    conv_res = torch.nn.functional.conv3d(
        mask.unsqueeze(0), filter, padding=erode_size // 2
    ).squeeze(dim=0)
    erode = conv_res == filter.sum()

    assert erode.shape == org_shape

    return erode.to(org_dtype)


def cross_filter(erode_size=3):
    filter = torch.zeros((erode_size, erode_size, erode_size))
    filter[:, 1, 1] = 1
    filter[1, :, 1] = 1
    filter[1, 1, :] = 1
    return filter


def merge_fs_labels(mask):
    # Define pairs of labels to merge: (left label, right label)
    label_pairs = [
        (1, 40),
        (2, 41),
        (3, 42),
        (4, 43),
        (5, 44),
        (6, 45),
        (7, 46),
        (8, 47),
        (9, 48),
        (10, 49),
        (11, 50),
        (12, 51),
        (13, 52),
        (17, 53),
        (18, 54),
        (19, 55),
        (20, 56),
        (25, 57),
        (26, 58),
        (27, 59),
        (28, 60),
        (29, 61),
        (30, 62),
        (31, 63),
        (32, 64),
        (33, 65),
        (34, 66),
        (35, 67),
        (36, 68),
        (37, 69),
        (38, 70),
        (39, 71),
        (73, 74),
        (78, 79),
        (81, 82),
        (83, 84),
    ]

    # Replace left labels with right labels
    for left_label, right_label in label_pairs:
        mask[mask == left_label] = right_label

    return mask


mini_yoda = """
⠀⢀⣠⣄⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⡾⠿⠿⠿⠿⢷⣶⣦⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢰⣿⡟⠛⠛⠛⠻⠿⠿⢿⣶⣶⣦⣤⣤⣀⣀⡀⣀⣴⣾⡿⠟⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠻⢿⣷⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⣀⣀⣀⡀
⠀⠻⣿⣦⡀⠀⠉⠓⠶⢦⣄⣀⠉⠉⠛⠛⠻⠿⠟⠋⠁⠀⠀⠀⣤⡀⠀⠀⢠⠀⠀⠀⣠⠀⠀⠀⠀⠈⠙⠻⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠟⠛⠛⢻⣿
⠀⠀⠈⠻⣿⣦⠀⠀⠀⠀⠈⠙⠻⢷⣶⣤⡀⠀⠀⠀⠀⢀⣀⡀⠀⠙⢷⡀⠸⡇⠀⣰⠇⠀⢀⣀⣀⠀⠀⠀⠀⠀⠀⣀⣠⣤⣤⣶⡶⠶⠶⠒⠂⠀⠀⣠⣾⠟
⠀⠀⠀⠀⠈⢿⣷⡀⠀⠀⠀⠀⠀⠀⠈⢻⣿⡄⣠⣴⣿⣯⣭⣽⣷⣆⠀⠁⠀⠀⠀⠀⢠⣾⣿⣿⣿⣿⣦⡀⠀⣠⣾⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⣠⣾⡟⠁⠀
⠀⠀⠀⠀⠀⠈⢻⣷⣄⠀⠀⠀⠀⠀⠀⠀⣿⡗⢻⣿⣧⣽⣿⣿⣿⣧⠀⠀⣀⣀⠀⢠⣿⣧⣼⣿⣿⣿⣿⠗⠰⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠋⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠙⢿⣶⣄⡀⠀⠀⠀⠀⠸⠃⠈⠻⣿⣿⣿⣿⣿⡿⠃⠾⣥⡬⠗⠸⣿⣿⣿⣿⣿⡿⠛⠀⢀⡟⠀⠀⠀⠀⠀⠀⣀⣠⣾⡿⠋⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⣷⣶⣤⣤⣄⣰⣄⠀⠀⠉⠉⠉⠁⠀⢀⣀⣠⣄⣀⡀⠀⠉⠉⠉⠀⠀⢀⣠⣾⣥⣤⣤⣤⣶⣶⡿⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⢻⣿⠛⢿⣷⣦⣤⣴⣶⣶⣦⣤⣤⣤⣤⣬⣥⡴⠶⠾⠿⠿⠿⠿⠛⢛⣿⣿⣿⣯⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣧⡀⠈⠉⠀⠈⠁⣾⠛⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⣿⠟⠉⣹⣿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣸⣿⣿⣦⣀⠀⠀⠀⢻⡀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣶⣿⠋⣿⠛⠃⠀⣈⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡿⢿⡀⠈⢹⡿⠶⣶⣼⡇⠀⢀⣀⣀⣤⣴⣾⠟⠋⣡⣿⡟⠀⢻⣶⠶⣿⣿⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣷⡈⢿⣦⣸⠇⢀⡿⠿⠿⡿⠿⠿⣿⠛⠋⠁⠀⣴⠟⣿⣧⡀⠈⢁⣰⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⢻⣦⣈⣽⣀⣾⠃⠀⢸⡇⠀⢸⡇⠀⢀⣠⡾⠋⢰⣿⣿⣿⣿⡿⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠿⢿⣿⣿⡟⠛⠃⠀⠀⣾⠀⠀⢸⡇⠐⠿⠋⠀⠀⣿⢻⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠁⢀⡴⠋⠀⣿⠀⠀⢸⠇⠀⠀⠀⠀⠀⠁⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⡿⠟⠋⠀⠀⠀⣿⠀⠀⣸⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣁⣀⠀⠀⠀⠀⣿⡀⠀⣿⠀⠀⠀⠀⠀⠀⢀⣈⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠟⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""

the_force = """


.................                                                        .................
...............                                                             ..............
............                                                                   ...........
..........                                                                      ..........
........                                                                          ........
......                                                                              ......
.....            @@@@  @@@@  @@@@@  @@@. @@@@   @@@@@@@@@@@  @@@ @@@@@@@             .....
...              @@@@@@@@@@ @@@ @@%  @@@@@@       +@@   @@@@@@@@ @@@                   ...
.                @@@@@@@@@@:@@  @@@   :@@@        +@@   @@@@@@@@ @@@@@%                  .
.                @@@ @@ @@@@@@@@@@@@   @@@        +@@   @@@  @@@ @@@@@@@@                .
                 +++    +++++.   +++   ++*        -++   +++  +++ ++++++++                 

    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@             @@@@@        .@@@@@            @@@@@@@@@          @             @@@@   
   @@@             @@=             %@@              @@@@             @             @@@@   
   @@@             @       @@@       @     %@@@@     @               @             @@@@   
   @@@     @@@@@@@@@     @@@@@@#     @     %@@@@     @     %@@@@@@@@@@     @@@@@@@@@@@@   
   @@@          *@@@     @@@@@@@     @             @@@     @@@@@@@@@@@          @@@@@@@   
   @@@     @@@@@@@@@     .@@@@@:     @           @@@@@      @@@@@@@@@@     @@@@@@@@@@@@   
   @@@     @@@@@@@@@@               @@     %@         @              @              @@@   
   @@@     @@@@@@@@@@@             @@@     %@@@       @@%            @              @@@   
   @@@@@@@@@@@@@@@@@@@@@@@#. :%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
    *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*    

    @@@@@@@@- @@@@@@@   @@@  @@@ @@@@@%@@@@@@@@@@@  @@@   @@@@  @@@@ @@@@@@* #@@   @@@    
    @@@   @@@ @@@++++    @@*@@@@ @@@@@%++*@@+++@@@  @@@    *@@@@@@  @@@  +@@@#@@   @@@    
    @@@@@@@@  @@@@@@     @@@@@@@@@@ @@%  -@@   @@@@@@@@      @@@@  @@@:   @@@#@@   @@@    
    @@@   @@@:@@@@@@@@    @@@@ @@@@ @@%  -@@   @@@  @@@      +@@    @@@. #@@@.@@# .@@@    
    @@@@@@@@* @@@@@@@@    @@@- @@@  @@%  -@@   @@@  @@@      +@@     @@@@@@:   @@@@@+    
.                                                                                        
..                                                                                      ..
.....                                                                                .....
.......                                                                            .......
.........                                                                         ........
..........                                                                     ...........
"""
