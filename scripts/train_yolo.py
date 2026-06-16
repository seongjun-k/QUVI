#!/usr/bin/env python3
"""
QUVI YOLOv8 Custom Training Tool
────────────────────────────────
A user-friendly command-line script to train a custom YOLOv8 model
for 3D printed objects and automatically deploy it to the QUVI pipeline.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def check_dependencies():
    print("🔄 Checking system requirements...")
    try:
        import torch
        print(f"  - PyTorch version: {torch.__version__}")
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"  - NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("  - ⚠️ No NVIDIA GPU detected. Training will run on CPU (this will be very slow!).")
        return gpu_available
    except ImportError:
        print("❌ Error: PyTorch is not installed. Please run: pip install torch torchvision")
        sys.exit(1)

def check_yolo():
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  - Ultralytics (YOLO) version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Error: Ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

def validate_dataset(dataset_dir):
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory '{dataset_path}' does not exist.")
        return None

    yaml_file = dataset_path / "data.yaml"
    if not yaml_file.exists():
        # Search for any yaml file in the directory
        yaml_files = list(dataset_path.glob("*.yaml"))
        if yaml_files:
            yaml_file = yaml_files[0]
            print(f"ℹ️ Found configuration file: {yaml_file.name}")
        else:
            print(f"❌ Error: Could not find 'data.yaml' or any config file in '{dataset_path}'.")
            print("   Please make sure your exported YOLOv8 dataset contains a config yaml file.")
            return None

    # Read and inspect yaml to check target class
    try:
        import yaml
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            classes = data.get('names', [])
            if isinstance(classes, dict):
                classes = list(classes.values())
            
            print(f"📊 Dataset Classes: {classes}")
            if 'print_object' not in classes:
                print("⚠️ Warning: 'print_object' class not found in the dataset configuration.")
                print(f"   The QUVI pipeline expects detecting 'print_object'.")
                confirm = input("   Do you want to proceed anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    sys.exit(0)
    except Exception as e:
        print(f"⚠️ Warning: Could not inspect classes in configuration file: {e}")

    return yaml_file

def main():
    parser = argparse.ArgumentParser(description="QUVI YOLOv8 Easy Trainer")
    parser.add_argument("--dataset", type=str, help="Path to the dataset directory (containing data.yaml)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="", help="Device to train on (e.g. 0, cpu)")
    args = parser.parse_args()

    print("=========================================")
    print("       QUVI YOLOv8 Training Tool         ")
    print("=========================================")

    # 1. Dependency checks
    gpu_available = check_dependencies()
    check_yolo()
    print("=========================================")

    # 2. Get dataset path
    dataset_dir = args.dataset
    if not dataset_dir:
        dataset_dir = input("📂 Enter the path to your dataset directory: ").strip()
        if not dataset_dir:
            print("❌ Error: Dataset path cannot be empty.")
            sys.exit(1)

    yaml_file = validate_dataset(dataset_dir)
    if not yaml_file:
        sys.exit(1)

    # 3. Setup training parameters interactively if not provided via args
    epochs = args.epochs
    if not args.dataset: # interactive mode
        try:
            epochs_input = input(f"🔁 Enter number of epochs (default {epochs}): ").strip()
            if epochs_input:
                epochs = int(epochs_input)
        except ValueError:
            print("⚠️ Invalid input, using default epochs.")

    batch = args.batch
    if not args.dataset: # interactive mode
        try:
            batch_input = input(f"📦 Enter batch size (default {batch}): ").strip()
            if batch_input:
                batch = int(batch_input)
        except ValueError:
            print("⚠️ Invalid input, using default batch size.")

    device = args.device
    if not device:
        if gpu_available:
            device = "0"
            print("🚀 Defaulting to GPU training (device=0)")
        else:
            device = "cpu"
            print("⚠️ Defaulting to CPU training (this will be slow)")

    print("=========================================")
    print("🏁 Starting Training with configuration:")
    print(f"  - Dataset config: {yaml_file}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch}")
    print(f"  - Image size: {args.imgsz}")
    print(f"  - Training device: {device}")
    print("=========================================")

    confirm = input("▶️ Ready to start training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Training cancelled.")
        sys.exit(0)

    # 4. Run training using ultralytics YOLO API
    from ultralytics import YOLO
    
    print("\n🚀 Loading pre-trained yolov8n.pt...")
    model = YOLO("yolov8n.pt")

    print("\n🔥 Training model... (Please do not close the terminal)\n")
    try:
        results = model.train(
            data=str(yaml_file),
            epochs=epochs,
            batch=batch,
            imgsz=args.imgsz,
            device=device,
            project="quvi_yolo_train",
            name="print_object_model",
            exist_ok=True
        )
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)

    print("\n✅ Training finished successfully!")

    # 5. Automatically copy best model to target path
    project_dir = Path("quvi_yolo_train") / "print_object_model"
    best_weights = project_dir / "weights" / "best.pt"

    if not best_weights.exists():
        # Fallback search if path is different
        found_weights = list(Path("runs").glob("**/best.pt"))
        if found_weights:
            best_weights = found_weights[0]
        else:
            print("⚠️ Warning: Could not find the trained 'best.pt' weights automatically.")
            print("   Please check your output directory and move the file manually.")
            sys.exit(0)

    target_path = Path("/workspace/data/models/best.pt")
    # Fallback to local path if running outside workspace mapping
    if not Path("/workspace").exists():
        target_path = Path("./data/models/best.pt")

    print(f"\n📂 Found trained model at: {best_weights}")
    print(f"📦 Deploying model to: {target_path}")

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, target_path)
        print("🎉 Success! The new YOLO model has been successfully deployed to the QUVI pipeline.")
        print("   Restart the launch files (run.sh) for the model to take effect.")
    except Exception as e:
        print(f"❌ Error copying model to target path: {e}")
        print(f"   Please manually copy '{best_weights}' to '{target_path}'")

if __name__ == "__main__":
    main()
