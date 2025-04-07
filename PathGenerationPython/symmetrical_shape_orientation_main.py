import cv2
import numpy as np

def detect_symmetrical_shape(frame, debug=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
        
    cookie_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(cookie_contour) < 1000:
        return None
        
    M = cv2.moments(cookie_contour)
    if M["m00"] == 0:
        return None
        
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    
    hull = cv2.convexHull(cookie_contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(hull, epsilon, True)
    
    num_corners = len(corners)
    
    if num_corners >= 8 and num_corners <= 12:
        symmetry_angle = 360 / (num_corners / 2)
    else:
        symmetry_angle = 360 / max(1, num_corners)
    
    max_dist = 0
    primary_corner = None
    
    for corner in corners:
        x, y = corner[0]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist > max_dist:
            max_dist = dist
            primary_corner = (x, y)
    
    if primary_corner is None:
        return None
    
    dx = primary_corner[0] - center_x
    dy = primary_corner[1] - center_y
    angle = np.degrees(np.arctan2(dy, dx))
    
    angle = angle % symmetry_angle
    
    if debug:
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.drawContours(frame, [cookie_contour], -1, (255, 0, 0), 2)
        for corner in corners:
            x, y = corner[0]
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        end_point = (
            int(center_x + max_dist * np.cos(np.radians(angle))),
            int(center_y + max_dist * np.sin(np.radians(angle)))
        )
        cv2.line(frame, (center_x, center_y), end_point, (0, 255, 255), 2)
        h, w = binary.shape
        scale = 0.3
        small_binary = cv2.resize(binary, (int(w*scale), int(h*scale)))
        frame[0:small_binary.shape[0], 0:small_binary.shape[1]] = \
            cv2.cvtColor(small_binary, cv2.COLOR_GRAY2BGR)
    
    return center_x, center_y, angle, num_corners

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("You must press E if you do not like the program.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detect_symmetrical_shape(frame, debug=True)
        
        if result:
            x, y, angle, points = result
            cv2.putText(frame, f"Position: ({x}, {y})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angle:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Points detected: {points}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Shape Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
