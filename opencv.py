import cv2
import numpy as np

def get_homography_from_points(src_image, dst_image):
    """
    Get user-defined points from both images and calculate the homography matrix.
    """
    def click_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param.append((x, y))
            print(f"Point captured: ({x}, {y})")

    print("Select at least 4 corresponding points in the source and target images. Close the window when done.")

    # Get points from the source image
    src_points = []
    cv2.imshow("Source Image - Click Points", src_image)
    cv2.setMouseCallback("Source Image - Click Points", click_points, src_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Get points from the target image
    dst_points = []
    cv2.imshow("Target Image - Click Points", dst_image)
    cv2.setMouseCallback("Target Image - Click Points", click_points, dst_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("At least 4 points are required in both images to compute homography.")

    # Convert points to numpy arrays
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Compute the homography matrix
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    return H

def main():
    # Load the source and target images
    pdf_image = cv2.imread('image_pdf.png')
    ifc_image = cv2.imread('image_ifc.png')

    if pdf_image is None or ifc_image is None:
        raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")

    # Get homography matrix from user-defined points
    H = get_homography_from_points(pdf_image, ifc_image)

    def draw_transformed_dot(event, x, y, flags, param):
        """
        Callback to draw the transformed dot on the target image when clicking on the source image.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Transform the clicked point using homography
            dot_position = np.array([[x, y, 1]], dtype=np.float32).T  # Homogeneous coordinates
            transformed_position = np.dot(H, dot_position)
            transformed_cx = int(transformed_position[0] / transformed_position[2])
            transformed_cy = int(transformed_position[1] / transformed_position[2])

            print(f"Blue dot clicked at ({x}, {y}) in source image.")
            print(f"Corresponding position in target image: ({transformed_cx}, {transformed_cy})")

            # Draw the blue dot on the target image
            cv2.circle(ifc_image, (transformed_cx, transformed_cy), radius=10, color=(255, 0, 0), thickness=-1)

            # Show updated target image
            cv2.imshow("Target Image with Blue Dot", ifc_image)

    # Display the source image and set up the callback
    cv2.imshow("Source Image - Click to Add Dot", pdf_image)
    cv2.setMouseCallback("Source Image - Click to Add Dot", draw_transformed_dot)

    # Display the target image
    cv2.imshow("Target Image with Blue Dot", ifc_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the updated target image
    cv2.imwrite('target_with_blue_dots.jpg', ifc_image)
    print("Target image with blue dots saved as 'target_with_blue_dots.jpg'.")

if __name__ == "__main__":
    main()
