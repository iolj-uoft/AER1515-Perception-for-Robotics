from matplotlib import pyplot as plt
import cv2 as cv
import os
from R2D2 import R2D2

image_folder_path = 'test/left'
images = []
drawed_images = []
reliability_scores = []

for filename in os.listdir(image_folder_path):
    if filename.endswith('.png'):
        img_path = os.path.join(image_folder_path, filename)
        R2D2_obj = R2D2(img_path)
        img = cv.imread(img_path)
        # cv.imshow(f"{filename}", img)
        # cv.waitKey(0)
        if img is not None:
            images.append(img)
        else:
            print(f"Could not load image: {img_path}")
        
        r2d2_path = os.path.splitext(img_path)[0] + '.r2d2'
        
        if not os.path.exists(r2d2_path):
            print(f"No corresponding .r2d2 file found for {filename}")

        keypoints, descriptors, scores = R2D2_obj.load_data()
        reliability_scores.append(scores)
        print(descriptors.shape)
        
        # Convert keypoints to cv2.KeyPoint objects
        keypoints_cv = [cv.KeyPoint(x=float(x), y=float(y), size=float(s)) for x, y, s in keypoints]

        # Draw keypoints with confidence-based coloring
        img_draw = img.copy()
        for kp, score in zip(keypoints_cv[:30], scores):
            # Map score (0 to 1) to color (red to green)
            color = (int(255 * (1 - score)), int(255 * score), 0)  # BGR format
            cv.drawKeypoints(img_draw, [kp], img_draw, color=color, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        drawed_images.append(img_draw)
        
midpoint = len(drawed_images) // 2
group1 = drawed_images[:midpoint]
group2 = drawed_images[midpoint:]

fig1, axes1 = plt.subplots(len(group1), 1, figsize=(10, 10))
for i, ax in enumerate(axes1):
    ax.imshow(cv.cvtColor(group1[i], cv.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    ax.axis('off')  # Turn off axes
    ax.set_title(f"Image {i + 1}")  # Add a title for each subplot
plt.tight_layout()
plt.savefig("Q1.1.1.png", dpi=300)
plt.show()

# Plot the second group
fig2, axes2 = plt.subplots(len(group2), 1, figsize=(10, 10))
for i, ax in enumerate(axes2):
    ax.imshow(cv.cvtColor(group2[i], cv.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    ax.axis('off')  # Turn off axes
    ax.set_title(f"Image {midpoint + i + 1}")  # Add a title for each subplot
plt.tight_layout()
plt.savefig("Q1.1.2.png", dpi=300)
plt.show()