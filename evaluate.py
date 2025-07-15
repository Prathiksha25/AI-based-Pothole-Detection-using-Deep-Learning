from ultralytics import YOLO

# Load the trained model
model = YOLO(r"D:\VINYASA\FINAL Project\Pothole_Final\runs\train\pothole_model4\weights\best.pt") # Adjust if needed

# Run evaluation on validation set (by default, if set up during training)
metrics = model.val()

# Print available metrics
print("📊 Evaluation Metrics:")
print(f"Precision (mean): {metrics.box.mp:.4f}")
print(f"Recall (mean): {metrics.box.mr:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


import matplotlib.pyplot as plt

# Metrics
metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
before = [0.72, 0.60, 0.70, 0.45]
after = [0.83, 0.68, 0.79, 0.50]

x = range(len(metrics))

# Plot
plt.figure(figsize=(10, 6))
plt.bar(x, before, width=0.35, label='Before Augmentation', align='center')
plt.bar([i + 0.35 for i in x], after, width=0.35, label='After Augmentation', align='center')

plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.title('Model Performance Before vs After Data Augmentation')
plt.xticks([i + 0.175 for i in x], metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

