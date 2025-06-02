import cv2
import numpy as np
import math
import os
from scipy import stats

def load_image(image_path):
    """Load an image using OpenCV with error handling."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load image")
    return img

def preprocess_image(img):
    """Advanced preprocessing for robust feature detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Multi-scale edge enhancement
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Adaptive contrast enhancement
    mean_intensity = np.mean(grad)
    clahe = cv2.createCLAHE(clipLimit=7.0 if mean_intensity < 50 else 5.5, tileGridSize=(22, 22))
    enhanced = clahe.apply(grad)
    # Noise reduction
    enhanced = cv2.GaussianBlur(enhanced, (7, 7), 0.4)
    enhanced = cv2.bilateralFilter(enhanced, 27, 260, 260)
    # Dynamic thresholding
    block_size = max(29, int(min(img.shape[:2]) * 0.18) | 1)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, block_size, -0.5)
    # Morphological operations
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=9)
    # Multi-scale edge detection
    edges = cv2.Canny(thresh, 15, 420, apertureSize=3)
    if np.sum(edges) < 1500:  # Fallback for sparse edges
        edges = cv2.Canny(enhanced, 40, 320, apertureSize=3)
    if np.sum(edges) < 500:  # Second fallback
        edges = cv2.Canny(gray, 60, 280, apertureSize=3)
    cv2.imwrite("debug_preprocess_enhanced.jpg", enhanced)
    cv2.imwrite("debug_preprocess_grad.jpg", grad)
    return enhanced, thresh, edges

def get_contour_orientation(img, thresh):
    """Estimate orientation using robust contour selection."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, thresh, None, None
    
    # Filter contours by area, perimeter, and shape
    min_area = max(6000, img.shape[0] * img.shape[1] * 0.05)
    min_perimeter = min(img.shape[:2]) * 0.4
    valid_contours = [
        c for c in contours 
        if cv2.contourArea(c) > min_area and 
           cv2.arcLength(c, True) > min_perimeter and 
           cv2.isContourConvex(cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True))
    ]
    
    if not valid_contours:
        return 0, thresh, None, None
    
    # Evaluate top contours
    angles = []
    best_contour = None
    best_box = None
    max_score = 0
    for c in sorted(valid_contours, key=cv2.contourArea, reverse=True)[:12]:
        rect = cv2.minAreaRect(c)
        area = cv2.contourArea(c)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.85:
            continue
        perimeter = cv2.arcLength(c, True)
        width, height = rect[1]
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        if aspect_ratio < 0.35 or aspect_ratio > 0.85:
            continue
        score = rectangularity * (area / (img.shape[0] * img.shape[1])) * (perimeter / min_perimeter) * aspect_ratio
        if score > 0.1:
            angle = rect[2]
            # Robust orientation handling
            if width < height:
                if abs(angle) > 45:
                    angle = angle + 88 if angle < 0 else angle - 88
            else:
                angle = angle + 88 if angle < 0 else angle - 88
            angle = normalize_angle(angle)
            angles.append((angle, score))
            if score > max_score:
                max_score = score
                best_contour = c
                best_box = np.int32(cv2.boxPoints(rect))
    
    if not angles:
        return 0, thresh, None, None
    
    # Weighted average of contour angles
    weights = [s for _, s in angles]
    angles = [a for a, _ in angles]
    contour_angle = np.average(angles, weights=weights) if sum(weights) > 0 else angles[0]
    
    return contour_angle, thresh, best_box, best_contour

def normalize_angle(angle):
    """Normalize angle to nearest vertical or horizontal axis."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    # Map to nearest 0°, 90°, 180°, 270°
    candidates = [0, 90, -90, 180, -180, 270, -270]
    closest = min(candidates, key=lambda x: abs((angle - x) % 360))
    angle_diff = (angle - closest) % 360
    if angle_diff > 180:
        angle_diff -= 360
    return angle_diff

def get_hough_angle(edges, img_shape):
    """Detect dominant angle using optimized Hough transform."""
    min_line_length = min(img_shape[:2]) * 0.6
    lines = cv2.HoughLinesP(edges, 1, np.pi / 5760, threshold=80, minLineLength=min_line_length, maxLineGap=0.3)
    
    hough_angle = 0
    if lines is not None:
        angles = []
        lengths = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 0.2 < abs(angle) < 89.8:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angles.append(angle)
                lengths.append(length)
        if angles:
            angles = np.array(angles)
            weights = np.array(lengths) / np.sum(lengths) if lengths else np.ones_like(angles) / len(angles)
            kde = stats.gaussian_kde(angles, weights=weights, bw_method=0.008)
            angles_range = np.linspace(min(angles), max(angles), 1500)
            hough_angle = angles_range[np.argmax(kde(angles_range))]
            hough_angle = normalize_angle(hough_angle)
    
    return hough_angle, edges

def get_moment_angle(contour):
    """Calculate orientation using contour moments."""
    if contour is None:
        return 0
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return 0
    mu11 = moments['mu11']
    mu20 = moments['mu20']
    mu02 = moments['mu02']
    angle = 0.5 * math.atan2(2 * mu11, mu20 - mu02) * 180 / np.pi
    return normalize_angle(angle)

def rotate_image(img, angle):
    """Rotate image with high-precision interpolation."""
    if abs(angle) < 0.005:
        return img
    (h, w) = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def symmetry_check(img):
    """Check image symmetry to validate alignment."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    half = gray[:, :w//2]
    flipped_half = cv2.flip(gray[:, w//2:], 1)
    diff = cv2.absdiff(half, flipped_half)
    symmetry_score = np.mean(diff)
    return symmetry_score

def edge_alignment_check(img):
    """Check edge alignment to vertical/horizontal axes."""
    _, thresh, edges = preprocess_image(img)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 5760, threshold=80, minLineLength=min(img.shape[:2]) * 0.6, maxLineGap=0.3)
    if lines is None:
        # Fallback with relaxed parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi / 5760, threshold=40, minLineLength=min(img.shape[:2]) * 0.25, maxLineGap=3)
    if lines is None:
        # Final fallback with original grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 250, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 5760, threshold=30, minLineLength=min(img.shape[:2]) * 0.2, maxLineGap=5)
    if lines is None:
        return float('inf')
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
        angle = normalize_angle(angle)
        if abs(angle) < 2:
            angles.append(abs(angle))
    return np.mean(angles) if angles else float('inf')

def multi_orientation_check(img):
    """Check symmetry and edge alignment at multiple orientations."""
    angles = [0, 90, -90, 180, -180, 270, -270]
    best_angle = 0
    best_score = float('inf')
    for angle in angles:
        test_img = rotate_image(img, angle)
        symmetry_score = symmetry_check(test_img)
        edge_score = edge_alignment_check(test_img)
        # Shape-based validation
        _, test_thresh, _ = preprocess_image(test_img)
        _, _, _, test_contour = get_contour_orientation(test_img, test_thresh)
        shape_score = 0
        if test_contour is not None:
            rect = cv2.minAreaRect(test_contour)
            width, height = rect[1]
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
            shape_score = 1 if 0.35 < aspect_ratio < 0.85 else 15
        combined_score = symmetry_score + edge_score * 25 + shape_score * 8
        if combined_score < best_score:
            best_score = combined_score
            best_angle = angle
        debug_img = test_img.copy()
        cv2.putText(debug_img, f"Orientation Check: {angle:.4f}, Symmetry: {symmetry_score:.2f}, Edge: {edge_score:.4f}, Shape: {shape_score:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(f"debug_orientation_check_{angle:.0f}.jpg", debug_img)
    return best_angle, best_score

def detect_tilt_angle(img):
    """Detect tilt angle with robust multi-method approach."""
    enhanced, thresh, edges = preprocess_image(img)
    contour_angle, thresh, box, contour = get_contour_orientation(img, thresh)
    hough_angle, edges = get_hough_angle(edges, img.shape)
    moment_angle = get_moment_angle(contour)
    
    # Voting mechanism with stricter criteria
    valid_angles = [a for a in [contour_angle, hough_angle, moment_angle] if abs(a) > 0.04]
    weights = [0.6, 0.25, 0.15] if len(valid_angles) == 3 else [0.7, 0.3, 0.0][:len(valid_angles)]
    initial_angle = np.average(valid_angles, weights=weights[:len(valid_angles)]) if valid_angles else 0
    
    # Multi-orientation check
    orientation_angle, orientation_score = multi_orientation_check(img)
    if orientation_score < 12:
        initial_angle = normalize_angle(initial_angle + orientation_angle)
    
    # Iterative refinement with symmetry and edge optimization
    max_iterations = 80
    current_angle = initial_angle
    debug_angles = [current_angle]
    tolerance = 0.005
    best_score = float('inf')
    best_edge_score = float('inf')
    best_angle = current_angle
    best_img = img
    for i in range(max_iterations):
        if abs(current_angle) < tolerance:
            break
        test_img = rotate_image(img, current_angle)
        symmetry_score = symmetry_check(test_img)
        edge_score = edge_alignment_check(test_img)
        combined_score = symmetry_score + edge_score * 25
        if combined_score < best_score:
            best_score = combined_score
            best_edge_score = edge_score
            best_angle = current_angle
            best_img = test_img
        _, test_thresh, _ = preprocess_image(test_img)
        test_angle, _, _, test_contour = get_contour_orientation(test_img, test_thresh)
        moment_angle = get_moment_angle(test_contour)
        combined_angle = 0.7 * test_angle + 0.3 * moment_angle if abs(moment_angle) < 90 else test_angle
        if abs(combined_angle) > tolerance or symmetry_score > 12 or edge_score > 0.25:
            step = combined_angle * 0.5
            current_angle += step
            current_angle = normalize_angle(current_angle)
            debug_angles.append(current_angle)
            debug_img = test_img.copy()
            cv2.putText(debug_img, f"Angle: {current_angle:.4f}, Symmetry: {symmetry_score:.2f}, Edge: {edge_score:.4f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(f"debug_iteration_{i+1}.jpg", debug_img)
        else:
            break
    final_angle = best_angle
    
    # Final validation
    final_img = best_img
    _, final_thresh, _ = preprocess_image(final_img)
    final_angle_check, _, _, final_contour = get_contour_orientation(final_img, final_thresh)
    moment_angle = get_moment_angle(final_contour)
    final_symmetry = symmetry_check(final_img)
    final_edge_score = edge_alignment_check(final_img)
    if abs(moment_angle) > tolerance or abs(final_angle_check) > tolerance or final_symmetry > 12 or final_edge_score > 0.25:
        correction = (0.6 * final_angle_check + 0.4 * moment_angle) if abs(moment_angle) < 90 else final_angle_check
        final_angle += correction * 0.5
        final_angle = normalize_angle(final_angle)
        final_img = rotate_image(img, final_angle)
        debug_img = final_img.copy()
        cv2.putText(debug_img, f"Final Correction: {final_angle:.4f}, Symmetry: {final_symmetry:.2f}, Edge: {final_edge_score:.4f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite("debug_final_correction.jpg", debug_img)
    
    # Save debug outputs
    debug_img = img.copy()
    if box is not None:
        cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 3)
    if contour is not None:
        cv2.drawContours(debug_img, [contour], 0, (255, 0, 0), 2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 5760, threshold=80, minLineLength=min(img.shape[:2]) * 0.6, maxLineGap=0.3)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(debug_img, f"Initial Angle: {initial_angle:.4f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("debug_edges.jpg", edges)
    cv2.imwrite("debug_thresh.jpg", thresh)
    cv2.imwrite("debug_orientation.jpg", debug_img)
    
    with open("debug_angles.txt", "w") as f:
        f.write(f"Initial contour angle: {contour_angle:.4f}\n")
        f.write(f"Initial Hough angle: {hough_angle:.4f}\n")
        f.write(f"Initial moment angle: {moment_angle:.4f}\n")
        f.write(f"Orientation check angle: {orientation_angle:.4f}\n")
        f.write(f"Iteration angles: {', '.join([f'{a:.4f}' for a in debug_angles])}\n")
        f.write(f"Final angle: {final_angle:.4f}\n")
        f.write(f"Final symmetry score: {final_symmetry:.2f}\n")
        f.write(f"Final edge alignment score: {final_edge_score:.4f}\n")
    
    return final_angle, final_img

def process_image(input_path, output_path):
    """Process image to achieve perfect vertical/horizontal alignment."""
    try:
        # Load image
        img = load_image(input_path)
        
        # Detect tilt angle and get corrected image
        angle, img_rotated = detect_tilt_angle(img)
        print(f"Detected tilt angle: {angle:.4f} degrees")
        
        # Final validation
        _, thresh, _ = preprocess_image(img_rotated)
        final_angle, _, _, _ = get_contour_orientation(img_rotated, thresh)
        symmetry_score = symmetry_check(img_rotated)
        edge_score = edge_alignment_check(img_rotated)
        if abs(final_angle) > 0.01 or symmetry_score > 12 or edge_score > 0.25:
            print(f"Warning: Final image may still be tilted by {final_angle:.4f} degrees (symmetry score: {symmetry_score:.2f}, edge score: {edge_score:.4f})")
        else:
            print("Final image is aligned within ±0.01 degrees")
        
        # Save corrected image
        cv2.imwrite(output_path, img_rotated)
        print(f"Corrected image saved to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

if __name__ == "__main__":
    input_image = "input/biscuit.jpg"
    output_image = "output/biscuit_product.jpg"
    process_image(input_image, output_image)