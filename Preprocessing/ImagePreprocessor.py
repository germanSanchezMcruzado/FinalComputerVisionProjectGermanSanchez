import numpy as np
import cv2
from sklearn.cluster import KMeans

class ImagePreprocessor:

    @staticmethod
    def normalize_image(image):
        """
        Normalize the pixel values of the input image to the range [0, 1].

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Normalized image.
        """
        image_array = np.array(image)
        normalized_image = image_array.astype(np.float32) / 255.0
        return normalized_image

    @staticmethod
    def augment_data(image, rotate=False, scale=False, translate=False):
        """
        Apply data augmentation techniques to the input image.

        Args:
            image (numpy.ndarray): Input image.
            rotate (bool): Whether to perform random rotation.
            scale (bool): Whether to perform random scaling.
            translate (bool): Whether to perform random translation.

        Returns:
            numpy.ndarray: Augmented image.
        """
        # Initialize transformation matrix
        transformation_matrix = np.eye(3, dtype=np.float32)

        # Random rotation
        if rotate:
            angle = np.random.uniform(-30, 30)
            rows, cols = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # Random scaling
        if scale:
            scale_factor = np.random.uniform(0.8, 1.2)
            transformation_matrix[0, 0] *= scale_factor
            transformation_matrix[1, 1] *= scale_factor

        # Random translation
        if translate:
            tx = np.random.randint(-20, 20)
            ty = np.random.randint(-20, 20)
            transformation_matrix[0, 2] = tx
            transformation_matrix[1, 2] = ty

        # Apply affine transformation
        augmented_image = cv2.warpAffine(image, transformation_matrix, (image.shape[1], image.shape[0]))

        return augmented_image

    @staticmethod
    def invert_image(image):
        """
        Inverts the intensity values of the image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Inverted image.
        """
        return np.ones_like(image) * 255 - image

    @staticmethod
    def raise_contrast(image, linear=True):
        """
        Increases the contrast of the image.

        Args:
            image (numpy.ndarray): Input image.
            linear (bool): If True, performs linear contrast adjustment. Otherwise, performs non-linear contrast adjustment.

        Returns:
            numpy.ndarray: Image with increased contrast.
        """
        if linear:
            return np.clip(image * 2, 0, 255).astype(np.uint8)
        else:
            return np.clip((image / 255) ** 2 * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def histogram_equalization(image):
        """
        Applies histogram equalization to enhance the contrast of the image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image after histogram equalization.
        """
        # Check if the image is already single-channel
        if len(image.shape) == 2:
            # Convert to uint8 if necessary
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply histogram equalization
            equalized_image = cv2.equalizeHist(image)
            return equalized_image
        else:
            raise ValueError("Input image must be single-channel")

    @staticmethod
    def gamma_correction(image, gamma=1.0):
        """
        Performs gamma correction on the image to adjust its brightness.

        Args:
            image (numpy.ndarray): Input image.
            gamma (float): Gamma value for correction.

        Returns:
            numpy.ndarray: Image after gamma correction.
        """
        # Gamma correction formula
        gamma_corrected = ((image / 255) ** (1 / gamma)) * 255
        return np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    @staticmethod
    def noise_reduction(image, ksize=(5, 5), sigma=0):
        """
        Reduces noise in the image using Gaussian blur.

        Args:
            image (numpy.ndarray): Input image.
            :param image:
            :param sigma:
            :param ksize:

        Returns:
            numpy.ndarray: Image after noise reduction.
        """
        # Use Gaussian blur for noise reduction
        blurred_image = cv2.GaussianBlur(image, ksize, sigma)
        return blurred_image

    @staticmethod
    def gamma_and_bilinear_filtering(image, gamma=1.0):
        """
        Applies both gamma correction and bilinear filtering to the image for enhanced quality.

        Args:
            image (numpy.ndarray): Input image.
            gamma (float): Gamma value for correction.

        Returns:
            numpy.ndarray: Image after gamma correction and bilinear filtering.
        """
        # Apply gamma correction
        gamma_corrected = ImagePreprocessor.gamma_correction(image, gamma=gamma)

        # Apply bilinear filtering
        filtered_image = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)
        return filtered_image

    @staticmethod
    def mean_filtering(image, kernel_size=3):
        """
        Applies mean filtering to the image.

        Args:
            image (numpy.ndarray): Input image.
            kernel_size (int): Size of the kernel for filtering.

        Returns:
            numpy.ndarray: Image after mean filtering.
        """
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def kmeans_image(image, num_clusters=16):
        """
        Perform K-means clustering on an image.

        Args:
        - image (numpy.ndarray): Input image.
        - num_clusters (int): Number of clusters for K-means.

        Returns:
        - numpy.ndarray: Output image where each pixel is assigned the color of its cluster centroid.
        """

        # Reshape the image to a 2D array of pixels
        height, width, channels = image.shape
        reshaped_image = image.reshape((height * width, channels))

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(reshaped_image)

        # Assign each pixel to its nearest cluster centroid
        labels = kmeans.predict(reshaped_image)

        # Assign cluster centroids to the output image
        output_image = kmeans.cluster_centers_[labels].reshape(image.shape)

        return output_image.astype(np.uint8)

    @staticmethod
    def cartoonization(image, num_down=2, num_bilateral=7):
        """
        Applies cartoonization effect to the image.

        Args:
            image (numpy.ndarray): Input image.
            num_down (int): Number of times to downscale the image.
            num_bilateral (int): Number of iterations for bilateral filtering.

        Returns:
            numpy.ndarray: Cartoonized image.
        """
        # Downsample the image multiple times
        img_color = image
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)

        # Apply bilateral filtering multiple times
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

        # Upsample the image back to its original size
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)

        # Convert to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply median blur to the grayscale image
        img_blur = cv2.medianBlur(img_gray, 7)

        # Create edge mask
        img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

        # Combine the color and edge-detected images
        img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edge)

        return img_cartoon

    @staticmethod
    def subsample_with_antialiasing(image, scale_percent=50):
        """
        Subsamples the image with antialiasing.

        Args:
            image (numpy.ndarray): Input image.
            scale_percent (int): Percentage of the original size for subsampling.

        Returns:
            numpy.ndarray: Subsampled image with antialiasing.
        """
        # Calculate the new dimensions
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        # Resize the image using cubic interpolation
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return resized_image

    @staticmethod
    def gradient_magnitude(image):
        """
        Computes the gradient magnitude of the image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Gradient magnitude image.
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Compute Sobel gradients in x and y directions
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag = np.uint8(grad_mag)

        return grad_mag

    @staticmethod
    def partial_derivatives(image):
        """
        Computes the partial derivatives of the image with respect to x and y.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray, numpy.ndarray: Partial derivatives with respect to x and y.
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Compute partial derivatives with respect to x and y
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        return grad_x, grad_y

    @staticmethod
    def gradient_orientation(grad_x, grad_y):
        """
        Computes the gradient orientation of the image.

        Args:
            grad_x (numpy.ndarray): Partial derivative of the image with respect to x.
            grad_y (numpy.ndarray): Partial derivative of the image with respect to y.

        Returns:
            numpy.ndarray: Gradient orientation image.
        """
        # Compute gradient orientation
        grad_orient = np.arctan2(grad_y, grad_x) * 180 / np.pi
        grad_orient = np.uint8(grad_orient)

        return grad_orient

    @staticmethod
    def edge_detection(image, threshold1=100, threshold2=200):
        """
        Detects edges in the image using the Canny edge detection algorithm.

        Args:
            image (numpy.ndarray): Input image.
            threshold1 (int): First threshold for hysteresis procedure.
            threshold2 (int): Second threshold for hysteresis procedure.

        Returns:
            numpy.ndarray: Image with detected edges.
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, threshold1, threshold2)

        return edges

    @staticmethod
    def corner_detection(image, max_corners=100, quality_level=0.01, min_distance=10):
        """
        Detects corners in the image using the Shi-Tomasi corner detection algorithm.

        Args:
            image (numpy.ndarray): Input image.
            max_corners (int): Maximum number of corners to return.
            quality_level (float): Parameter characterizing the minimal accepted quality of image corners.
            min_distance (int): Minimum possible Euclidean distance between the returned corners.

        Returns:
            numpy.ndarray: Image with detected corners.
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Detect corners using Shi-Tomasi algorithm
        corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
        corners = np.int0(corners)

        # Draw corners on the image
        corner_image = np.copy(image)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corner_image, (x, y), 3, (0, 255, 0), -1)

        return corner_image
