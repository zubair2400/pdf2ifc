from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

# Global variables to store the images and homography matrix
pdf_image = cv2.imread('image_pdf.png')
ifc_image = cv2.imread('image_ifc.png')
homography_matrix = None

if pdf_image is None or ifc_image is None:
    raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")


@app.route('/set-points', methods=['POST'])
def set_points():
    """
    API to accept points from the user interface for calculating the homography matrix.
    Expected input: JSON with two keys: 'sourcePoints' and 'targetPoints'.
    Each key contains a list of (x, y) points.
    """
    global homography_matrix

    data = request.json
    src_points = np.array(data.get("sourcePoints"), dtype=np.float32)
    dst_points = np.array(data.get("targetPoints"), dtype=np.float32)

    if len(src_points) < 4 or len(dst_points) < 4:
        return jsonify({"error": "At least 4 points are required in both images to compute homography."}), 400

    # Compute the homography matrix
    homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    return jsonify({"message": "Homography matrix computed successfully."})


@app.route('/get-transformed-point', methods=['POST'])
def get_transformed_point():
    """
    API to accept a point from the user interface and transform it using the homography matrix.
    Expected input: JSON with a key 'point' containing (x, y).
    """
    global homography_matrix

    if homography_matrix is None:
        return jsonify({"error": "Homography matrix has not been set. Please set the points first."}), 400

    data = request.json
    point = np.array([[data.get("point")]], dtype=np.float32)  # Convert to a 3D array for transformation
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)

    transformed_x = float(transformed_point[0][0][0])
    transformed_y = float(transformed_point[0][0][1])

    return jsonify({"transformedPoint": (transformed_x, transformed_y)})


if __name__ == '__main__':
    app.run(debug=True)
