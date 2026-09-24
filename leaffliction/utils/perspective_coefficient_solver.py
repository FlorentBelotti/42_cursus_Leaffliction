"""Solve the eight coefficients of a Pillow perspective transformation."""

import numpy

PERSPECTIVE_COEFFICIENT_COUNT = 8


def find_perspective_transform_coefficients(
    source_corner_points,
    destination_corner_points,
):
    """Return the eight coefficients of a perspective transformation.

    Pillow walks the output image and asks, for every output pixel,
    where to read in the input image. The coefficients therefore encode
    the mapping destination -> source, which is why the destination
    points build the matrix and the source points the right hand side.
    Both point sequences must list the corners in the same order.
    """
    equation_matrix = _build_equation_matrix(
        source_corner_points,
        destination_corner_points,
    )
    target_vector = _build_target_vector(source_corner_points)
    solved_coefficients = numpy.linalg.lstsq(
        equation_matrix,
        target_vector,
        rcond=None,
    )[0]
    return tuple(float(coefficient) for coefficient in solved_coefficients)


def _build_equation_matrix(source_corner_points, destination_corner_points):
    """Build the eight by eight matrix of the perspective system."""
    matrix_rows = []
    for corner_index in range(len(source_corner_points)):
        source_x, source_y = source_corner_points[corner_index]
        destination_x, destination_y = destination_corner_points[corner_index]
        matrix_rows.append(
            _build_horizontal_equation_row(
                source_x,
                destination_x,
                destination_y,
            )
        )
        matrix_rows.append(
            _build_vertical_equation_row(
                source_y,
                destination_x,
                destination_y,
            )
        )
    return numpy.array(matrix_rows, dtype=numpy.float64)


def _build_horizontal_equation_row(source_x, destination_x, destination_y):
    """Build the row constraining the horizontal output coordinate."""
    return [
        destination_x,
        destination_y,
        1.0,
        0.0,
        0.0,
        0.0,
        -source_x * destination_x,
        -source_x * destination_y,
    ]


def _build_vertical_equation_row(source_y, destination_x, destination_y):
    """Build the row constraining the vertical output coordinate."""
    return [
        0.0,
        0.0,
        0.0,
        destination_x,
        destination_y,
        1.0,
        -source_y * destination_x,
        -source_y * destination_y,
    ]


def _build_target_vector(source_corner_points):
    """Flatten the source corners into the system right hand side."""
    flattened_coordinates = []
    for source_x, source_y in source_corner_points:
        flattened_coordinates.append(source_x)
        flattened_coordinates.append(source_y)
    return numpy.array(flattened_coordinates, dtype=numpy.float64)
