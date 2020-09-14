import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset


class Vis_dataset(Dataset):
    def __init__(self, dirs, verbose_parsing=True):
        self.dirs = dirs
        self.size = 0

        self.file_paths = []
        for directory in dirs:
            for fname in os.listdir(directory):
                filename, extension = os.path.splitext(fname)
                if extension in ['.png', '.jpg']:
                    self.file_paths.append(os.path.join(directory, fname))
                elif verbose_parsing:
                    print('Can\'t parse "{}". Unknown extension "{}"'.format(fname, extension))
        if self.transform:
            # Нужно для аугментации. Сэмплы, где может находиться ббокс
            bbox_positions = []
            for f in self.file_paths:
                filename, _ = os.path.splitext(f)
                bbox_positions.append(self.get_points_from_filename(filename))
            self.set_bbox_augmenation(bbox_positions)

        self.size = len(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def get_points_from_filename(filename):
        return order_points(np.array(filename.split('/')[-1].split("_")[:8], dtype='float32')).flatten()

    def __getitem__(self, idx):
        # filename example:
        # -0.001470_-0.230860_0.821296_0.110275_0.717603_0.753787_-0.072814_0.496837_-8749095987662151642_GQ-870
        # [-1] is plate's license number, [-2] is annotationID
        filename, _ = os.path.splitext(self.file_paths[idx])
        points = self.get_points_from_filename(filename)

        image = cv2.imread(self.file_paths[idx], 1)
        image = cv2.resize(image, (n_cols, n_rows))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        points_augmentated = points

        if self.transform:

            if random.random() < 0.7:
                scale = random.random() + 0.4

                image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                image = cv2.resize(image, (192, 64), interpolation=cv2.INTER_AREA)

            # Brightness modification
            if random.random() < 0.8:
                tmp = cv2.convertScaleAbs(image, alpha=random.random() * 1.2 + 0.2, beta=random.random() * 32 - 16)
            else:
                tmp = 255.0 - image
            #             # Revert if image too dark
            #             if np.max(tmp) > 10:
            #                 image = tmp
            image = np.clip(tmp, 0, 255)

            if random.random() < self.augm_warp_prob:
                # Edit with boolean flag and 2 if-else

                def __scale_bbox(points):
                    return np.array([
                        [points[0] * 192, points[1] * 64],
                        [points[2] * 192, points[3] * 64],
                        [points[4] * 192, points[5] * 64],
                        [points[6] * 192, points[7] * 64]], dtype="float32")

                def __distance(x1, y1, x2, y2):
                    return np.linalg.norm([x2 - x1, y2 - y1])

                def __check_augmented_bbox(bbox, min_dist=10):
                    [x1, y1, x2, y2, x3, y3, x4, y4] = list(
                        np.array(bbox).flatten())  # from top left point in clockwise order
                    d1_4 = __distance(x1, y1, x4, y4)
                    d2_3 = __distance(x2, y2, x3, y3)
                    return d1_4 > min_dist and d2_3 > min_dist

                # Take dst and points_augmentated from augmentation
                if random.random() < 0.75:
                    points_augmentated = self.get_bbox_augmentation()
                    src = __scale_bbox(points)
                    dst = __scale_bbox(points_augmentated)

                # Modify dst and points_augmentated 
                else:
                    points_augmentated = self.get_bbox_augmentation()
                    src = __scale_bbox(points)
                    dst = __scale_bbox(points_augmentated)
                    func = random.choice([
                        self.vertical_perspective_pts,
                        self.horizontal_perspective_pts,
                        self.random_perspective_pts,
                        self.trapezoid_perspective_pts,
                    ])
                    __dst = func(dst)
                    if __check_augmented_bbox(__dst):
                        dst = __dst
                        points_augmentated = __dst / np.array([192, 64])

                M = cv2.getPerspectiveTransform(src, dst)
                image = cv2.warpPerspective(image, M, (192, 64), borderMode=cv2.BORDER_REPLICATE)

        points_augmentated = points_augmentated.flatten().astype(np.float32)

        # AREA LOSS
        full_area = 192 * 64
        norm_points = np.array([
            [points_augmentated[0] * 192, points_augmentated[1] * 64],
            [points_augmentated[2] * 192, points_augmentated[3] * 64],
            [points_augmentated[4] * 192, points_augmentated[5] * 64],
            [points_augmentated[6] * 192, points_augmentated[7] * 64]], dtype="float32")
        points_area = abs((norm_points[0][0] * norm_points[1][1] - norm_points[0][1] * norm_points[1][0]) +
                          (norm_points[1][0] * norm_points[2][1] - norm_points[1][1] * norm_points[2][0]) +
                          (norm_points[2][0] * norm_points[3][1] - norm_points[2][1] * norm_points[3][0]) +
                          (norm_points[3][0] * norm_points[0][1] - norm_points[3][1] * norm_points[0][0])) / 2
        w = 1.5 - points_area / full_area

        image = image / 255.0

        # Аугментации ниже работают с числами в картинках [0,1], поэтому они должны располагаться после нормализации!
        if self.transform:
            # Пошумим
            if random.random() < 0.3:
                noise_prob = random.random()
                if noise_prob < 0.33:
                    image = self.gauss_noise(image, var=random.uniform(0, 0.06))
                elif noise_prob < 0.66:
                    image = self.s_and_p(image, s_vs_p=random.random(), amount=random.uniform(0, 0.06))
                else:
                    image = self.poisson_noise(image)

            # А теперь цензура
            if random.random() < 0.3:
                noise_prob = random.random()
                if noise_prob < 0.33:
                    image = self.gaus_blur(image, kernel_size=random.randrange(1, 8, 2))
                elif noise_prob <= 0.66:
                    image = self.motion_blur_horizontal(image, kernel_size=random.randrange(1, 8, 2))
                else:
                    image = self.motion_blur_vertical(image, kernel_size=random.randrange(1, 8, 2))

        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        return image, points_augmentated, w