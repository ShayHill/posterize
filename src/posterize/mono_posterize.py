"""Monochrome posterization functionality.

This module provides functionality for posterizing monochrome (grayscale) images
by clustering pixel values and creating a TargetImage.

:author: Shay Hill
:created: 2025-01-23
"""

from __future__ import annotations

from typing import Annotated, TypeAlias

import numpy as np
from basic_colormath import floats_to_uint8, get_delta_e_matrix
from cluster_colors import stack_pool_cut_colors
from numpy import typing as npt

from posterize.quantization import TargetImage

_Pixels: TypeAlias = Annotated[npt.NDArray[np.uint8], "(m, n)"]


def posterize_mono(
    pixels: _Pixels,
    num_cols: int,
) -> TargetImage:
    """Posterize a monochrome image by clustering pixel values.

    :param pixels: (m, n) shaped array of uint8 grayscale pixel values
    :param num_cols: number of colors to reduce to
    :return: TargetImage instance with clustered palette

    Performs simple quantization by clustering the pixel values into num_cols
    values and assigning the closest exemplar to each pixel value.
    """
    # Reshape pixels to 1D array
    pixels_flat = pixels.flatten()
    m, n = pixels.shape

    # Convert monochrome pixels to RGBA format for stack_pool_cut_colors
    # Each grayscale value becomes (gray, gray, gray, 255)
    rgba_colors = np.zeros((len(pixels_flat), 4), dtype=np.uint8)
    rgba_colors[:, 0] = pixels_flat  # R
    rgba_colors[:, 1] = pixels_flat  # G
    rgba_colors[:, 2] = pixels_flat  # B
    rgba_colors[:, 3] = 255  # A

    # Cluster the colors using stack_pool_cut_colors
    # This returns float values, we'll convert to uint8
    clustered = stack_pool_cut_colors(rgba_colors)
    
    # Get the exemplars (cluster centers) - these are the num_cols representative colors
    # Take only RGB channels and convert to uint8
    exemplars_rgb = floats_to_uint8(clustered[:, :3])
    
    # Get unique exemplars (in case clustering returned duplicates)
    exemplars_rgb, unique_indices = np.unique(exemplars_rgb, axis=0, return_index=True)
    
    # If we got fewer than num_cols unique exemplars, pad with the last exemplar
    if len(exemplars_rgb) < num_cols:
        last_exemplar = exemplars_rgb[-1:]
        padding = np.repeat(last_exemplar, num_cols - len(exemplars_rgb), axis=0)
        exemplars_rgb = np.vstack([exemplars_rgb, padding])
    elif len(exemplars_rgb) > num_cols:
        # If we got more, take the first num_cols
        exemplars_rgb = exemplars_rgb[:num_cols]
    
    # Expand exemplars to (num_cols, 3) - they're already RGB from the clustering
    # But since they came from grayscale, they should be (gray, gray, gray)
    palette = exemplars_rgb.astype(np.uint8)

    # Assign each pixel to the nearest exemplar
    # Convert pixels back to RGB format for comparison
    pixels_rgb = np.zeros((len(pixels_flat), 3), dtype=np.uint8)
    pixels_rgb[:, 0] = pixels_flat
    pixels_rgb[:, 1] = pixels_flat
    pixels_rgb[:, 2] = pixels_flat

    # Use delta E matrix to find nearest color
    unique_pixels, reverse_index = np.unique(pixels_rgb, axis=0, return_inverse=True)
    pmatrix = get_delta_e_matrix(unique_pixels, palette)
    indices_flat = np.argmin(pmatrix[reverse_index], axis=1)

    # Reshape indices back to original image shape
    indices = indices_flat.reshape((m, n))

    # Create proximity matrix for the palette
    palette_pmatrix = get_delta_e_matrix(palette)

    # Create TargetImage
    target_image = TargetImage(palette, indices, palette_pmatrix)

    return target_image
