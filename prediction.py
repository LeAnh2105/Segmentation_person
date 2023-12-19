import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
# Download the mask_rcnn_coco.h5 file from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
model.load_weights(filepath="mask_rcnn_coco.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("i.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Filter instances for 'person' class (class_id = 1)
person_indices = [i for i, class_id in enumerate(r['class_ids']) if class_id == 1]

# Only keep the information related to 'person'
person_boxes = r['rois'][person_indices]
person_masks = r['masks'][:, :, person_indices]
person_class_ids = r['class_ids'][person_indices]
person_scores = r['scores'][person_indices]

# Visualize the detected 'person' instances
mrcnn.visualize.display_instances(image=image,
                                  boxes=person_boxes,
                                  masks=person_masks,
                                  class_ids=person_class_ids,
                                  class_names=CLASS_NAMES,
                                  scores=person_scores)

# Tạo một danh sách để chứa tất cả thông tin của person_masks
all_person_info = []

# Lặp qua từng instance 'person' và xử lý thông tin tọa độ
for i in range(len(person_indices)):
    mask = person_masks[:, :, i]

    # Tạo một mặt nạ nhị phân bằng cách áp dụng ngưỡng
    binary_mask = (mask > 0.5).astype(np.uint8)  # Bạn có thể điều chỉnh ngưỡng nếu cần

    # Lấy tọa độ (indices) của các pixel có giá trị 1 trong mặt nạ
    indices = np.column_stack(np.where(binary_mask == 1))

    # Thêm thông tin của person_mask vào danh sách
    person_info = {
        'Person_ID': i + 1,
        'Coordinates': indices.tolist()
    }

    all_person_info.append(person_info)

# Xuất thông tin của tất cả person_masks vào file person.txt
output_filename = "person.txt"
with open(output_filename, 'w') as file:
    file.write("Person_ID\tCoordinates\n")
    for person_info in all_person_info:
        file.write(f"{person_info['Person_ID']}\t{person_info['Coordinates']}\n")

print(f"Thông tin của tất cả person_masks đã được xuất ra file: {output_filename}")


# Tạo một mặt nạ tổng hợp bằng cách kết hợp tất cả các mặt nạ cá nhân
composite_mask = np.zeros_like(person_masks[:, :, 0], dtype=np.uint8)
for i in range(len(person_indices)):
    mask = person_masks[:, :, i]
    binary_mask = (mask > 0.5).astype(np.uint8)
    composite_mask += binary_mask

# Hiển thị mặt nạ tổng hợp
plt.imshow(composite_mask, cmap='gray')
plt.axis('off')  # Tắt hiển thị trục
plt.savefig('result_mask.jpg', bbox_inches='tight', pad_inches=0)  # Thiết lập bbox_inches và pad_inches
print(f"Hình ảnh đã được lưu vào file: result_mask.jpg")
plt.show()


# Tạo một ảnh tổng hợp để chứa tất cả các mặt nạ
overlay_image = image.copy()

# Iterate through each 'person' instance and blend its mask with the overlay image
for i in range(len(person_indices)):
    mask = person_masks[:, :, i]

    # Create a binary mask by thresholding
    binary_mask = (mask > 0.5).astype(np.uint8)  # You can adjust the threshold if needed

    # Random color for each mask
    color = np.random.rand(3)

    # Blend the mask with the overlay image using alpha-blending
    overlay_image[binary_mask == 1] = (overlay_image[binary_mask == 1] * 0.6 + color * 255 * 0.4).astype(np.uint8)


# Display the result image with all masks
plt.imshow(overlay_image)
plt.axis('off')
plt.show()
# Lưu hình ảnh ra file
output_filename = 'result.jpg'
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

print(f"Hình ảnh kết quả đã được lưu vào file: {output_filename}")
