# -*- coding: utf-8 -*-
"""
A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
no "up" or "down" in the images.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_geometric_ops as geometric_ops
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_image_boxes_validation_utils \
    as validation_utils
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_patch_sampling_ops \
    as patch_sampling_ops
from twomartens.masterthesis.ssd_keras.data_generator import object_detection_2d_photometric_ops as photometric_ops


class DataAugmentationSatellite:
    """
    A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
    no "up" or "down" in the images.

    Applies a chain of photometric and geometric image transformations. For documentation, please refer
    to the documentation of the individual transformations involved.
    """
    
    def __init__(self,
                 resize_height: int,
                 resize_width: int,
                 random_brightness: Tuple[int, int, float] = (-48, 48, 0.5),
                 random_contrast: Tuple[float, float, float] = (0.5, 1.8, 0.5),
                 random_saturation: Tuple[float, float, float] = (0.5, 1.8, 0.5),
                 random_hue: Tuple[int, float] = (18, 0.5),
                 random_flip: float = 0.5,
                 random_rotate: Tuple[Tuple[int, int, int], float] = ([90, 180, 270], 0.5),
                 min_scale: float = 0.3,
                 max_scale: float = 2.0,
                 min_aspect_ratio: float = 0.8,
                 max_aspect_ratio: float = 1.25,
                 n_trials_max: int = 3,
                 clip_boxes: bool = True,
                 overlap_criterion: str = 'area',
                 bounds_box_filter: Tuple[float, float] = (0.3, 1.0),
                 bounds_validator: Tuple[float, float] = (0.5, 1.0),
                 n_boxes_min: int = 1,
                 background: Tuple[int, int, int] = (0, 0, 0),
                 labels_format: Optional[Dict[str, int]] = None) -> None:
        
        if labels_format is None:
            labels_format = {'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.overlap_criterion = overlap_criterion
        self.bounds_box_filter = bounds_box_filter
        self.bounds_validator = bounds_validator
        self.n_boxes_min = n_boxes_min
        self.background = background
        self.labels_format = labels_format
        
        # Determines which boxes are kept in an image after the transformations have been applied.
        self.box_filter_patch = validation_utils.BoxFilter(check_overlap=True,
                                                           check_min_area=False,
                                                           check_degenerate=False,
                                                           overlap_criterion=self.overlap_criterion,
                                                           overlap_bounds=self.bounds_box_filter,
                                                           labels_format=self.labels_format)
        
        self.box_filter_resize = validation_utils.BoxFilter(check_overlap=False,
                                                            check_min_area=True,
                                                            check_degenerate=True,
                                                            min_area=16,
                                                            labels_format=self.labels_format)
        
        # Determines whether the result of the transformations is a valid training image.
        self.image_validator = validation_utils.ImageValidator(
            overlap_criterion=self.overlap_criterion,
            bounds=self.bounds_validator,
            n_boxes_min=self.n_boxes_min,
            labels_format=self.labels_format)
        
        # Utility transformations
        self.convert_to_3_channels = photometric_ops.ConvertTo3Channels()  # Make sure all images
        # end up having 3 channels.
        self.convert_RGB_to_HSV = photometric_ops.ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = photometric_ops.ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = photometric_ops.ConvertDataType(to='float32')
        self.convert_to_uint8 = photometric_ops.ConvertDataType(to='uint8')
        self.resize = geometric_ops.Resize(height=resize_height,
                                           width=resize_width,
                                           box_filter=self.box_filter_resize,
                                           labels_format=self.labels_format)
        
        # Photometric transformations
        self.random_brightness = photometric_ops.RandomBrightness(lower=random_brightness[0],
                                                                  upper=random_brightness[1],
                                                                  prob=random_brightness[2])
        self.random_contrast = photometric_ops.RandomContrast(lower=random_contrast[0],
                                                              upper=random_contrast[1],
                                                              prob=random_contrast[2])
        self.random_saturation = photometric_ops.RandomSaturation(lower=random_saturation[0],
                                                                  upper=random_saturation[1],
                                                                  prob=random_saturation[2])
        self.random_hue = photometric_ops.RandomHue(max_delta=random_hue[0], prob=random_hue[1])
        
        # Geometric transformations
        self.random_horizontal_flip = geometric_ops.RandomFlip(dim='horizontal', prob=random_flip,
                                                               labels_format=self.labels_format)
        self.random_vertical_flip = geometric_ops.RandomFlip(dim='vertical', prob=random_flip,
                                                             labels_format=self.labels_format)
        self.random_rotate = geometric_ops.RandomRotate(angles=random_rotate[0],
                                                        prob=random_rotate[1],
                                                        labels_format=self.labels_format)
        self.patch_coord_generator = patch_sampling_ops.PatchCoordinateGenerator(must_match='w_ar',
                                                                                 min_scale=min_scale,
                                                                                 max_scale=max_scale,
                                                                                 scale_uniformly=False,
                                                                                 min_aspect_ratio=min_aspect_ratio,
                                                                                 max_aspect_ratio=max_aspect_ratio)
        self.random_patch = patch_sampling_ops.RandomPatch(
            patch_coord_generator=self.patch_coord_generator,
            box_filter=self.box_filter_patch,
            image_validator=self.image_validator,
            n_trials_max=self.n_trials_max,
            clip_boxes=self.clip_boxes,
            prob=1.0,
            can_fail=False,
            labels_format=self.labels_format)
        
        # Define the processing chain.
        self.transformations = [self.convert_to_3_channels,
                                self.convert_to_float32,
                                self.random_brightness,
                                self.random_contrast,
                                self.convert_to_uint8,
                                self.convert_RGB_to_HSV,
                                self.convert_to_float32,
                                self.random_saturation,
                                self.random_hue,
                                self.convert_to_uint8,
                                self.convert_HSV_to_RGB,
                                self.random_horizontal_flip,
                                self.random_vertical_flip,
                                self.random_rotate,
                                self.random_patch,
                                self.resize]
    
    def __call__(self, image: np.ndarray, labels: Optional[np.ndarray] = None) -> Union[
            Tuple[np.ndarray, np.ndarray], np.ndarray]:
        
        self.random_patch.labels_format = self.labels_format
        self.random_horizontal_flip.labels_format = self.labels_format
        self.random_vertical_flip.labels_format = self.labels_format
        self.random_rotate.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format
        
        if not (labels is None):
            for transform in self.transformations:
                image, labels = transform(image, labels)
            return image, labels
        else:
            for transform in self.transformations:
                image = transform(image)
            return image
