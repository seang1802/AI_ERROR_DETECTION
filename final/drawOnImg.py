import cv2 as cv
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: {(x, y)}")

def main(img_path):
    global points

    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    H, W = img.shape[:2]
    print(f"Loaded image {img_path} at {W}x{H}")

    if (W, H) != (1280, 720):
        print("WARNING: Image is NOT 1280x720. ROI will match this image size only.")

    clone = img.copy()

    cv.namedWindow("Draw ROI")
    cv.setMouseCallback("Draw ROI", mouse_callback)

    print("Click to add points. Press ENTER to finish, 'r' = reset, 'u' = undo.")

    while True:
        disp = clone.copy()

        # Draw points
        for p in points:
            cv.circle(disp, p, 4, (0, 0, 255), -1)

        # Draw outline
        if len(points) > 1:
            cv.polylines(disp, [np.array(points, np.int32)], False, (0, 255, 0), 2)

        cv.imshow("Draw ROI", disp)
        key = cv.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            break
        elif key == ord('r'):
            points = []
            print("Points reset.")
        elif key == ord('u'):
            if points:
                removed = points.pop()
                print(f"Undo: removed {removed}")

    cv.destroyAllWindows()

    if not points:
        print("No points selected.")
        return

    # Print final ROI
    print("\nFINAL ROI POINTS (paste into your code):")
    print("[")
    for x, y in points:
        print(f"    ({x}, {y}),")
    print("]")

    # Save preview
    out = img.copy()
    if len(points) > 1:
        cv.polylines(out, [np.array(points, np.int32)], True, (0, 255, 0), 2)
    cv.imwrite("roi_preview.png", out)

    print("\nSaved roi_preview.png")

if __name__ == "__main__":
    main("RPI4(OLD)/print_imgs/test05.jpg")

