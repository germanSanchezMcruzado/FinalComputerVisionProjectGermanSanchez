from matplotlib import pyplot as plt


class DataVisualizer:
    @staticmethod
    def show_images(images, rows, cols):
        """
        Display multiple images in a grid layout.

        Parameters:
        - images: List of images to display.
        - rows: Number of rows in the grid.
        - cols: Number of columns in the grid.
        """
        if images is not None and len(images) > 0:
            plt.figure(figsize=(cols * 16, rows * 9))
            i = 0
            for image in images:
                plt.subplot(rows, cols, i + 1)
                plt.imshow(image)
                plt.title("Image {i}".format(i=i))
                i += 1
            plt.tight_layout()
            plt.show()
        else:
            print("Error: No images provided.")

    @staticmethod
    def show_image(image):
        """
        Display a single image.

        Parameters:
        - image: Image to display.
        """
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        plt.imshow(image)
        plt.title("Image")
        plt.axis('off')  # Hide axes
        plt.show()
