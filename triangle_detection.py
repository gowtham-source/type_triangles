import cv2
import numpy as np
import math

def detect_color_points(image, color_ranges):
    points = {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    for color_name, (lower, upper) in color_ranges.items():
        # Create mask for current color
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply some morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points[color_name] = (cx, cy)
                    # Draw circle at detected point for debugging
                    cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)
    
    return points

def calculate_angle(p1, p2, p3):
    # Calculate vectors
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate angle in radians
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Convert to degrees
    return math.degrees(angle)

def draw_angle_arc(image, vertex, p1, p2, radius=30):
    # Calculate vectors
    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]], dtype=np.float32)
    v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]], dtype=np.float32)
    
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Calculate start and end angles
    start_angle = np.arctan2(v1[1], v1[0])
    end_angle = np.arctan2(v2[1], v2[0])
    
    # Ensure positive angle
    if end_angle < start_angle:
        end_angle += 2 * np.pi
    
    # Draw arc
    num_points = 30
    for i in range(num_points):
        t = i / (num_points - 1)
        curr_angle = start_angle + t * (end_angle - start_angle)
        x = int(vertex[0] + radius * np.cos(curr_angle))
        y = int(vertex[1] + radius * np.sin(curr_angle))
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

def classify_triangle(angles, vertices):
    # Classify by angles
    angle_type = ""
    if any(abs(angle - 90) < 1 for angle in angles):  # 1 degree tolerance
        angle_type = "Right"
    elif all(angle < 90 for angle in angles):
        angle_type = "Acute"
    else:
        angle_type = "Obtuse"
    
    # Calculate side lengths
    sides = []
    for i in range(3):
        next_i = (i + 1) % 3
        side_length = np.sqrt(
            (vertices[next_i][0] - vertices[i][0])**2 + 
            (vertices[next_i][1] - vertices[i][1])**2
        )
        sides.append(side_length)
    
    # Classify by sides
    sides = sorted(sides)
    tolerance = sides[0] * 0.05  # 5% tolerance for floating point comparison
    
    if all(abs(sides[0] - side) < tolerance for side in sides):
        side_type = "Equilateral"
    elif (abs(sides[0] - sides[1]) < tolerance or 
          abs(sides[1] - sides[2]) < tolerance):
        side_type = "Isosceles"
    else:
        side_type = "Scalene"
    
    return angle_type, side_type

def process_triangle_image(image_path):
    """
    Process an image to detect colored pins forming a triangle, calculate angles,
    and classify the triangle type.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        tuple: (processed_image, angles_dict, classification_str)
            - processed_image: Image with triangle drawn and measurements
            - angles_dict: Dictionary of angles for each vertex
            - classification_str: String describing triangle type
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")

    # Define color ranges in HSV
    color_ranges = {
        'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
        'blue': (np.array([100, 100, 100]), np.array([130, 255, 255])),
        'green': (np.array([40, 100, 100]), np.array([80, 255, 255]))
    }

    # Detect color points
    points = detect_color_points(image, color_ranges)

    if len(points) == 3:
        # Draw triangle and calculate angles
        vertices = list(points.values())
        
        # Draw lines
        cv2.line(image, vertices[0], vertices[1], (255, 255, 255), 2)
        cv2.line(image, vertices[1], vertices[2], (255, 255, 255), 2)
        cv2.line(image, vertices[2], vertices[0], (255, 255, 255), 2)
        
        # Calculate angles
        angles = {
            'A': calculate_angle(vertices[1], vertices[0], vertices[2]),  # Red point angle
            'B': calculate_angle(vertices[0], vertices[1], vertices[2]),  # Blue point angle
            'C': calculate_angle(vertices[0], vertices[2], vertices[1])   # Green point angle
        }
        
        # Draw arcs and add text for angles
        for i, (point, angle, vertex) in enumerate(zip(['A', 'B', 'C'], angles.values(), vertices)):
            # Draw arc
            prev_vertex = vertices[(i-1) % 3]
            next_vertex = vertices[(i+1) % 3]
            draw_angle_arc(image, vertex, prev_vertex, next_vertex)
            
            # Add text for angle
            text = f'{point}: {angle:.1f} deg'
            cv2.putText(image, text, (vertex[0] - 20, vertex[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Classify triangle
        angle_type, side_type = classify_triangle(list(angles.values()), vertices)
        classification = f"{angle_type} {side_type} Triangle"
        cv2.putText(image, classification, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return image, angles, classification
    else:
        raise ValueError(f"Could not detect all three colored points. Only found {len(points)} points")

def main():
    # Process the image
    try:
        image, angles, classification = process_triangle_image('trio.jpg')
        
        # Save the result
        cv2.imwrite('triangle_result.jpg', image)
        
        # Print results
        print(f"Angles: {angles}")
        print(f"Triangle Classification: {classification}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
