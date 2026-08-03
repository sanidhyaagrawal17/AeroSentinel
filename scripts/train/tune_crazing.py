from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

# 1. Monkey-patch the fitness function to solely maximize Crazing (Class 0) mAP50
# By default, YOLO maximizes the average mAP50-95 across all classes. We want an expert!
original_fitness = DetMetrics.fitness

def custom_crazing_fitness(self):
    try:
        # Crazing is class 0 in split_full.yaml
        if 0 in self.ap_class_index:
            idx = list(self.ap_class_index).index(0)
            return float(self.box.ap50[idx])
        return 0.0
    except Exception as e:
        print(f"Custom fitness error: {e}")
        return 0.0

DetMetrics.fitness = property(custom_crazing_fitness)

if __name__ == '__main__':
    print("=== Starting GA Hyperparameter Tuning (Crazing Maximization) ===")
    # Initialize the winning architecture
    model = YOLO('yolov9s.pt')
    
    # Run Ray Tune GA optimization
    # iterations=10: search 10 different hyperparameter combinations
    # epochs=30: each run trains for 30 epochs
    model.tune(
        data='split_full.yaml',
        epochs=30,
        iterations=10,
        optimizer='AdamW',
        plots=False,
        save=False,
        val=True,
        device=0
    )
    
    print("=== GA Tuning Finished! ===")
    print("Best hyperparameters saved to runs/detect/tune/best_hyperparameters.yaml")
