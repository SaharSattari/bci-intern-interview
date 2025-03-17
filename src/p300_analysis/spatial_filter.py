import numpy as np


class SpatialFilter:
    """
    spatial filter design based on Contrastive Principal Component Analysis (CPCA)

    Attributes:
        foreground_data (numpy.ndarray): The foreground data used for calculating the spatial filter.
        background_data (numpy.ndarray): The background data used for calculating the spatial filter.

    Methods:
        calculate_spatial_filter(): Calculates the spatial filter based on the foreground and background data.
        apply_spatial_filter(spatial_filter): Applies the spatial filter to the foreground and background data.

    """

    def __init__(self, foreground_data, background_data):
        self.foreground_data = foreground_data
        self.background_data = background_data

    def _dimension_check(self, data):
        """
        Checks the dimensions of the data and transposes it if necessary.
        the size should be (num_channels, num_samples) at the end

        Args:
            data (numpy.ndarray): The data to be checked.

        Returns:
            numpy.ndarray: The checked data.

        """
        if data.shape[0] > data.shape[1]:
            data = data.T
        return data

    def calculate_spatial_filter(self):
        """
        Calculates the spatial filter based on the foreground and background data and contrastive PCA algorithm.

        Returns:
            numpy.ndarray: The calculated spatial filter. (set of weights with the size of num_channels)

        """
        cov_foreground = np.cov(self._dimension_check(self.foreground_data))
        cov_background = np.cov(self._dimension_check(self.background_data))

        eigen_values, eigen_vectors = np.linalg.eig(cov_foreground - cov_background)

        sorted_indices = np.argsort(eigen_values)[::-1]
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
        spatial_filter = sorted_eigen_vectors[:, 0]

        return spatial_filter

    def apply_spatial_filter(self, spatial_filter):
        """
        (this function is not used in the main code)
        Applies the spatial filter to the foreground and background data.

        Args:
            spatial_filter (numpy.ndarray): The spatial filter to be applied.

        Returns:
            tuple: A tuple containing the projected foreground and background data.

        """
        foreground_projected = np.dot(
            spatial_filter.T, self._dimension_check(self.foreground_data)
        )
        background_projected = np.dot(
            spatial_filter.T, self._dimension_check(self.background_data)
        )

        return foreground_projected, background_projected
