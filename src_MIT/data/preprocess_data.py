import os
import wfdb
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_and_process_record(record_path):
    """Load and process single ECG record"""
    try:
        # Read record and annotations
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        # Get signal data and annotations
        signals = record.p_signal
        labels = annotation.symbol
        sample_points = annotation.sample

        return signals, labels, sample_points
    except Exception as e:
        print(f"Error processing record {record_path}: {str(e)}")
        return None, None, None


def extract_heartbeats(signals, sample_points, window_size=250):
    """Extract heartbeat segments from signals"""
    heartbeats = []
    for point in sample_points:
        # Ensure window is within signal range
        start = max(0, point - window_size // 2)
        end = min(len(signals), point + window_size // 2)

        # Extract heartbeat segment
        beat = signals[start:end]

        # Pad if segment length is insufficient
        if len(beat) < window_size:
            pad_width = window_size - len(beat)
            beat = np.pad(beat, ((0, pad_width), (0, 0)), mode='constant')

        heartbeats.append(beat)

    return np.array(heartbeats)


def preprocess_data():
    """Preprocess MIT-BIH dataset"""
    print("Starting data preprocessing...")

    # Create processed directory
    os.makedirs('data/processed', exist_ok=True)

    # Get all records
    record_paths = []
    raw_dir = '/tmp/pycharm_project_669/data/raw'
    for file in os.listdir(raw_dir):
        if file.endswith('.dat'):
            record_name = file[:-4]
            record_paths.append(os.path.join(raw_dir, record_name))

    # Collect all data
    all_heartbeats = []
    all_labels = []

    for record_path in tqdm(record_paths, desc="Processing records"):
        signals, labels, sample_points = load_and_process_record(record_path)
        if signals is None:
            continue

        # Extract heartbeats
        heartbeats = extract_heartbeats(signals, sample_points)

        # Keep only main classes: N(Normal), V(VEB), A(SVEB)
        valid_beats = []
        valid_labels = []
        for beat, label in zip(heartbeats, labels):
            if label in ['N', 'V', 'A']:
                valid_beats.append(beat)
                valid_labels.append(label)

        all_heartbeats.extend(valid_beats)
        all_labels.extend(valid_labels)

    # Convert to numpy arrays
    X = np.array(all_heartbeats)
    y = np.array(all_labels)

    # Label encoding
    label_map = {'N': 0, 'V': 1, 'A': 2}
    y = np.array([label_map[label] for label in y])

    # Dataset split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Prepare data dictionary
    processed_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'classes': ['Normal', 'VEB', 'SVEB'],
        'scaler': scaler
    }

    # Save processed data
    with open('data/processed/processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

    print("\nData preprocessing completed!")
    print(f"Total samples: {len(y)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    print("\nClass distribution:")
    for i, class_name in enumerate(['Normal', 'VEB', 'SVEB']):
        print(f"{class_name}: {sum(y == i)} samples")


if __name__ == '__main__':
    preprocess_data()
    # print(f"✅ processed_data.pkl 保存路径：{processed_data.pkl}")  # 打印完整路径