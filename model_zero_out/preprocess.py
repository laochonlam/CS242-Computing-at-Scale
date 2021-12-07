import torchvision.transforms.functional as TF
import random
import math
import numpy as np

# My Preprocess Lib for cropping

class MyEraseTransform:
    """Erase."""

    i = 0
    j = 0
    h = 0
    w = 0
    v = 0

    def __init__(self, i, j, h, w, v):
        self.i = i,
        self.j = j,
        self.h = h,
        self.w = w,
        self.v = v,

    def __call__(self, x):
        x[:, self.i[0]:self.i[0] + self.h[0], self.j[0]:self.j[0] + self.w[0]] = self.v[0]
        return x

class MyEraseCircleTransform:
    """Erase from outer circle."""

    total_size = 0
    hidden_ratio = 0
    v = 0

    def __init__(self, total_size, hidden_ratio, v):
        self.total_size = total_size,
        self.hidden_ratio = hidden_ratio,
        self.v = v,

    def __call__(self, x):
        total_size = self.total_size[0]
        hidden_ratio = self.hidden_ratio[0]
        v = self.v[0]

        bound = 224 - int(math.sqrt(total_size * total_size * hidden_ratio))
        # print(str(bound) + " * " + str(bound))
        x[:, 0: int(bound / 2 + bound % 2), 0: total_size] = self.v[0]
        # print("[%d: %d, %d: %d]" % (0, int(bound / 2 + bound % 2), 0, total_size))    
        x[:, total_size - int(bound / 2): total_size, 0: total_size] = self.v[0]
        # print("[%d: %d, %d: %d]" % (total_size - int(bound / 2), total_size, 0, total_size))
        x[:, 0: total_size, total_size - int(bound / 2 + bound % 2) : total_size] = self.v[0]
        # print("[%d: %d, %d: %d]" % (0, total_size, total_size - int(bound / 2 + bound % 2), total_size))
        x[:, 0: total_size, 0: int(bound / 2 )] = self.v[0]
        # print("[%d: %d, %d: %d]" % (0, total_size, 0, int(bound / 2 )))
        
        return x

class MyEraseEvenTransform:
    """Erase."""

    total_size = 0
    erase_per = 0
    erase_piece = 0
    v = 0

    def __init__(self, total_size, hidden_ratio, v):
        self.total_size = total_size,
        self.erase_per = int(1 / hidden_ratio),
        print("earse per: " + str(self.erase_per))
        self.v = v,

    def __call__(self, x):
        total_size = self.total_size[0]
        erase_per = self.erase_per[0]
        v = self.v[0]

        i = 0
        j = 0
        k = 0
        while j < total_size:
            x[:, i, j] = v
            i = i + erase_per
            k = k + 1
            if (i) == 224:
                i = i - total_size + 4
                j = j + 1
            elif (i) > 224:
                i = i - total_size - 4
                j = j + 1

        # while j < total_size:
        #     x[:, i, j] = v
        #     i = i + erase_per
        #     k = k + 1
        #     if (i) >= 224:
        #         i = i - total_size
        #         j = j + 1


        # print("total erase: " + str(k))
        return x


class MyRandomErasePixelTransform:
    """Erase pixel randomly"""

    total_size = 0
    erase_ratio = 0.0
    v = 0

    def __init__(self, total_size, erase_ratio, v):
        self.total_size = total_size,
        self.erase_ratio = erase_ratio,
        self.v = v,

    def __call__(self, x):
        total_size = self.total_size[0]
        erase_ratio = self.erase_ratio[0]
        v = self.v[0]

        total_pixel_size = total_size * total_size 
        num_erase_pixel = int(total_pixel_size * erase_ratio)
        
        idx = np.random.choice(total_pixel_size, size=num_erase_pixel, replace=0)
        out = np.column_stack((np.unravel_index(idx,(total_size,total_size))))
        
        for i, j in out:
            x[:, i, j] = 0
        
        return x

class MyEraseJPEGTransform:
    """Erase 8x8 pixel like for JPEG validation"""

    i = 0
    j = 0
    h = 0
    w = 0
    v = 0
    
    total_size = 0
    erase_ratio = 0.0
    total_blocks = 0
    delete_blocks = 0
    stride = 0
    v = 0
    x_list = []
    y_list = []

    def __init__(self, total_size, erase_ratio, v, input_delete_block=0):
        self.total_size = total_size,
        self.erase_ratio = erase_ratio,
        self.total_blocks = total_size / 8 * total_size / 8
        self.stride = int(total_size / 8)
        self.delete_blocks = int(erase_ratio * self.total_blocks)
        x_list = list(range(1, int(total_size / 8)))
        y_list = list(range(1, int(total_size / 8)))
        if (input_delete_block):
            self.delete_blocks = input_delete_block
        self.v = v,

    def __call__(self, x):
        delete_blocks = self.delete_blocks
        total_blocks = int(self.total_blocks)
        v = self.v[0]
        stride = int(self.stride)
        # print(total_blocks)
        # print(delete_blocks)
        idx = np.random.choice(total_blocks, size=delete_blocks, replace=0)
        # print(idx)
        
        out = np.column_stack(np.unravel_index(idx, (stride, stride)))
        # print(out)
        # out=[[2, 0], [0, 2], [20, 20]]
        for i, j in out:
            x[:, i*8:i*8 + 8, j*8:j*8 + 8] = v
        # print(out)
        # exit(1)
        # for i in delete_blocks:
        #     random.choice
        #     x[:, i[0]:i[0] + h[0], j[0]:j[0] + w[0]] = v[0]
        return x


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

# rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])