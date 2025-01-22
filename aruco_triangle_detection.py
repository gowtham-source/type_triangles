import cv2
import numpy as np
import math
from triangle_detection import detect_color_points, calculate_angle, draw_angle_arc, classify_triangle
import os
import google.generativeai as genai

# Configure Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyBMZshYv40LxHWivZdoRQfR1Z6aGZddzC8"

def detect_aruco_markers(image):
    """
    Detect ArUco markers in the image and return their centers and IDs.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create ArUco dictionary and detector
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    # parameters = cv2.aruco.DetectorParameters()
    # detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) >= 2:
        # Get centers of markers
        centers = {}
        for i, corner in enumerate(corners):
            center = np.mean(corner[0], axis=0)
            centers[int(ids[i][0])] = tuple(map(int, center))
            
            # Draw marker centers for debugging
            cv2.circle(image, centers[int(ids[i][0])], 5, (0, 255, 0), -1)
        
        print(f"Detected {len(centers)} ArUco markers with IDs: {list(centers.keys())}")
        return centers, corners
    
    raise ValueError("No ArUco markers detected")

def get_arena_roi(image, corners):
    """
    Get the region of interest (ROI) defined by the ArUco markers.
    Returns the ROI and the offset coordinates.
    """
    # Get all corner points
    all_corners = np.concatenate(corners)
    
    # Get bounding box
    min_x = int(np.min(all_corners[:, :, 0]))
    min_y = int(np.min(all_corners[:, :, 1]))
    max_x = int(np.max(all_corners[:, :, 0]))
    max_y = int(np.max(all_corners[:, :, 1]))
    
    # Add small padding
    padding = 10
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image.shape[1], max_x + padding)
    max_y = min(image.shape[0], max_y + padding)
    
    # Extract ROI
    roi = image[min_y:max_y, min_x:max_x]
    
    return roi, (min_x, min_y)

def detect_color_points(image, color_ranges, corners):
    """
    Detect red, blue, and green points within the arena defined by ArUco markers.
    Returns a dictionary mapping color to point coordinates.
    """
    # Get ROI within ArUco markers
    roi, (offset_x, offset_y) = get_arena_roi(image, corners)
    
    # Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    points = {}
    debug_image = roi.copy()
    
    for color, ranges in color_ranges.items():
        # Combine masks if multiple ranges
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Save mask for debugging
        cv2.imwrite(f'{color}_mask.jpg', mask)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 50:  # Minimum area threshold
                # Get centroid
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Add offset to get coordinates in original image
                    global_cx = cx + offset_x
                    global_cy = cy + offset_y
                    points[color] = (global_cx, global_cy)
                    
                    # Draw detected point on debug image
                    color_bgr = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0)}
                    cv2.circle(debug_image, (cx, cy), 5, color_bgr[color], -1)
                    cv2.putText(debug_image, color, (cx+10, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr[color], 2)
    
    # Save debug image
    cv2.imwrite('detected_points.jpg', debug_image)
    cv2.imwrite('arena_roi.jpg', roi)  # Save ROI for debugging
    
    print(f"Detected {len(points)} colored points within arena: {list(points.keys())}")
    
    if len(points) < 3:
        raise ValueError(f"Could not detect all three colored points within arena. Only found {len(points)} points")
    
    return points

def calculate_pixel_to_cm_ratio(marker_centers):
    """
    Calculate the pixel to cm ratio using the known distance between markers (30 cm)
    """
    if len(marker_centers) < 2:
        raise ValueError("Need at least 2 markers for calibration")
    
    # Calculate distances between markers in pixels
    distances = []
    marker_ids = list(marker_centers.keys())
    
    # Calculate distances between each pair of markers
    for i in range(len(marker_ids)):
        for j in range(i + 1, len(marker_ids)):
            p1 = marker_centers[marker_ids[i]]
            p2 = marker_centers[marker_ids[j]]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            distances.append(distance)
    
    # Sort distances and take the median of the closest pairs
    # This helps filter out diagonal distances
    distances.sort()
    if len(distances) > 2:
        # Take average of the shortest distances (likely the adjacent markers)
        avg_distance_pixels = np.mean(distances[:len(marker_ids)-1])
    else:
        avg_distance_pixels = distances[0]
    
    # Known distance between markers in cm
    MARKER_DISTANCE_CM = 30.0
    
    # Calculate ratio (pixels/cm)
    ratio = avg_distance_pixels / MARKER_DISTANCE_CM
    print(f"Calibration: {ratio:.2f} pixels = 1 cm")
    return ratio

def calculate_real_distances(points, pixel_to_cm_ratio):
    """
    Calculate real-world distances between points in centimeters.
    """
    distances = {}
    vertices = ['A', 'B', 'C']
    colors = ['red', 'blue', 'green']
    
    for i in range(3):
        p1 = points[colors[i]]
        p2 = points[colors[(i + 1) % 3]]
        
        # Calculate Euclidean distance in pixels
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Convert to centimeters using the calibration ratio
        cm_distance = pixel_distance / pixel_to_cm_ratio
        
        # Store the distance with vertex labels
        distances[f"{vertices[i]}-{vertices[(i + 1) % 3]}"] = cm_distance
    
    return distances

def calculate_triangle_perimeter(distances):
    """
    Calculate the perimeter of the triangle.
    """
    return sum(distances.values())

def calculate_triangle_area(distances):
    """
    Calculate the area of the triangle using Heron's formula.
    """
    # Get the three sides
    sides = list(distances.values())
    
    # Calculate semi-perimeter
    s = sum(sides) / 2
    
    # Calculate area using Heron's formula
    # Area = √(s(s-a)(s-b)(s-c)) where s is semi-perimeter and a,b,c are sides
    area = math.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))
    
    return area

def process_image_with_calibration(image_source):
    """
    Process an image to detect and analyze a triangle.
    
    Args:
        image_source: Either a string (file path) or numpy array (image data)
    
    Returns:
        tuple: (processed_image, angles, distances, classification, area, perimeter)
    """
    try:
        # Handle both file path and direct image data
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError("Could not read the image")
        else:
            image = image_source
            
        # Detect ArUco markers
        marker_centers, corners = detect_aruco_markers(image)
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, corners)
        
        # Calculate calibration ratio
        pixel_to_cm_ratio = calculate_pixel_to_cm_ratio(marker_centers)
        
        # Define color ranges in HSV
        color_ranges = {
            'red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),    # Lower red
                (np.array([160, 50, 50]), np.array([180, 255, 255]))  # Upper red
            ],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
        }
        
        # Detect colored pins within arena
        points = detect_color_points(image, color_ranges, corners)
        
        if len(points) == 3:
            vertices = list(points.values())
            
            # Draw triangle
            cv2.line(image, vertices[0], vertices[1], (255, 255, 255), 2)
            cv2.line(image, vertices[1], vertices[2], (255, 255, 255), 2)
            cv2.line(image, vertices[2], vertices[0], (255, 255, 255), 2)
            
            # Calculate angles
            angles = {
                'A': calculate_angle(vertices[1], vertices[0], vertices[2]),
                'B': calculate_angle(vertices[0], vertices[1], vertices[2]),
                'C': calculate_angle(vertices[0], vertices[2], vertices[1])
            }
            
            # Calculate real-world distances
            distances = calculate_real_distances(points, pixel_to_cm_ratio)
            
            # Draw arcs and add text for angles and distances
            for i, (point, angle, vertex) in enumerate(zip(['A', 'B', 'C'], angles.values(), vertices)):
                # Draw arc
                prev_vertex = vertices[(i-1) % 3]
                next_vertex = vertices[(i+1) % 3]
                draw_angle_arc(image, vertex, prev_vertex, next_vertex)
                
                # Add text for angle
                text = f'{point}: {angle:.1f} deg'
                cv2.putText(image, text, (vertex[0] - 20, vertex[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add distance labels
            for i, (label, distance) in enumerate(distances.items()):
                mid_point = (
                    (vertices[i][0] + vertices[(i+1)%3][0]) // 2,
                    (vertices[i][1] + vertices[(i+1)%3][1]) // 2
                )
                text = f'{label}: {distance:.1f} cm'
                cv2.putText(image, text, mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Classify triangle
            angle_type, side_type = classify_triangle(list(angles.values()), vertices)
            classification = f"{angle_type} {side_type} Triangle"
            cv2.putText(image, classification, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Calculate and display area and perimeter
            perimeter = calculate_triangle_perimeter(distances)
            area = calculate_triangle_area(distances)
            
            # Add area and perimeter to the image
            cv2.putText(image, f"Area: {area:.2f} cm²", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Perimeter: {perimeter:.2f} cm", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return image, angles, distances, classification, area, perimeter
        else:
            raise ValueError(f"Could not detect all three colored points within arena. Only found {len(points)} points")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def explain_triangle_with_gemini(angles, distances, classification):
    """
    Use Gemini to explain the triangle classification in a student-friendly way
    """
    # Configure Gemini
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create the model with student-friendly settings
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }

    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
    )

    # Craft a student-friendly prompt
    prompt = f"""
    You are a friendly math teacher explaining a triangle to middle school students (grades 6-8).
    Here's a triangle we measured in class:
    Angles: {angles['A']:.1f}°, {angles['B']:.1f}°, {angles['C']:.1f}°
    Sides: {distances['A-B']:.1f} cm, {distances['B-C']:.1f} cm, {distances['C-A']:.1f} cm
    Type: {classification}

    Write a short, explanation about this triangle focusing on its angles and sides. 
    Include a simple real-world example where we might find this type of triangle.
    Use friendly, conversational language that a middle school student would understand easily.
    Avoid using complex mathematical terms or making it sound like a textbook.

    start like "Here this triangle is a "
    """

    try:
        # Get explanation from Gemini
        response = model.generate_content(prompt)
        
        print("\nTriangle Explanation for Students:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
    except Exception as e:
        print(f"\nError getting explanation from Gemini: {str(e)}")

def main():
    try:
        # Process the image
        image, angles, distances, classification, area, perimeter = process_image_with_calibration('up.png')
        
        # Save the result
        cv2.imwrite('aruco_triangle_result.jpg', image)
        
        # Print measurements
        print("\nTriangle Analysis Results:")
        print("-" * 50)
        print("Angles:")
        for vertex, angle in angles.items():
            print(f"  Vertex {vertex}: {angle:.2f} degrees")
        print("\nDistances:")
        for edge, distance in distances.items():
            print(f"  Edge {edge}: {distance:.2f} cm")
        print(f"\nTriangle Classification: {classification}")
        print(f"Area: {area:.2f} cm²")
        print(f"Perimeter: {perimeter:.2f} cm")
        print("-" * 50)
        
        # Add Gemini explanation
        explain_triangle_with_gemini(angles, distances, classification)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
