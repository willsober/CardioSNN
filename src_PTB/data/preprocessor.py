import numpy as np
import wfdb
from scipy import signal as sig
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
import matplotlib.pyplot as plt
import os

class ECGPreprocessor:
    def __init__(self, sampling_rate=360, data_dir='data/raw'):
        self.sampling_rate = sampling_rate
        self.data_dir = data_dir
        self.valid_types = ['N', 'A', 'V', 'L', 'R']
    
    def filter_ecg(self, ecg_signal):
      
        b, a = sig.butter(3, [0.5, 45], btype='band', fs=self.sampling_rate)
        return sig.filtfilt(b, a, ecg_signal)
    
    def detect_r_peaks(self, ecg_signal):
       
        diff_signal = np.diff(ecg_signal)
        squared_signal = diff_signal ** 2
        window_size = int(0.1 * self.sampling_rate)
        moving_avg = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
        peaks, _ = sig.find_peaks(moving_avg, distance=50)
        return peaks
    
    def segment_beats(self, signal, peaks, window_size=100):
      
        segments = []
        for peak in peaks:
            if peak - window_size >= 0 and peak + window_size < len(signal):
                segment = signal[peak-window_size:peak+window_size]
                segments.append(segment)
        return np.array(segments)
    
    def match_beats_annotations(self, r_peaks, annotation_samples, annotation_symbols):
       
        beat_labels = []
        for peak in r_peaks:
            closest_idx = np.argmin(np.abs(annotation_samples - peak))
            beat_labels.append(annotation_symbols[closest_idx])
        return beat_labels
    
    def get_available_records(self):
     
        records = set()
        for file in os.listdir(self.data_dir):
            if file.endswith('.dat'):
                record_num = int(file.split('.')[0])
               
                if (os.path.exists(os.path.join(self.data_dir, f"{record_num}.atr")) and
                    os.path.exists(os.path.join(self.data_dir, f"{record_num}.hea"))):
                    records.add(record_num)
        return sorted(list(records))
    
    def load_multiple_records(self, signal_length=360000):
     
        record_numbers = self.get_available_records()
        print(f"Found {len(record_numbers)} available records: {record_numbers}")
        
        all_segments = []
        all_labels = []

        total_records = len(record_numbers)
        processed_records = 0
        
        for record_num in record_numbers:
            record_path = os.path.join(self.data_dir, str(record_num))
            try:
                
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, 'atr')
                
                segments_for_record = []
                labels_for_record = []
                
                for start in range(0, len(record.p_signal), signal_length//4):  
                    end = start + signal_length
                    if end > len(record.p_signal):
                        break
                        
                   
                    signal = record.p_signal[start:end, 0]
                    filtered_signal = self.filter_ecg(signal)
                    r_peaks = self.detect_r_peaks(filtered_signal)
              
                    valid_annotations = [(s-start, sym) for s, sym in zip(annotation.sample, annotation.symbol)
                                    if start <= s < end]
                    if not valid_annotations:
                        continue
                    valid_samples, valid_symbols = zip(*valid_annotations)
                    
                    segments = self.segment_beats(filtered_signal, r_peaks)
                    labels = self.match_beats_annotations(r_peaks, valid_samples, valid_symbols)
                    
                   
                    min_len = min(len(segments), len(labels))
                    segments = segments[:min_len]
                    labels = labels[:min_len]
                  
                    valid_types = ['N', 'A', 'V', 'L', 'R'] 
                    valid_indices = [i for i, label in enumerate(labels) if label in valid_types]
                    segments = [segments[i] for i in valid_indices]
                    labels = [labels[i] for i in valid_indices]
                 
                    segments_for_record.extend(segments)
                    labels_for_record.extend(labels)
                
            
                label_counts = Counter(labels_for_record)
                if len(label_counts) >= 2: 
                    min_count = min(label_counts.values())
                    max_samples_per_class = min(min_count * 5, 1000) 
                    
                    balanced_segments = []
                    balanced_labels = []
                    
                    for label in valid_types:
                        if label in label_counts:
                            indices = [i for i, l in enumerate(labels_for_record) if l == label]
                            if len(indices) > max_samples_per_class:
                                indices = np.random.choice(indices, max_samples_per_class, replace=False)
                            balanced_segments.extend([segments_for_record[i] for i in indices])
                            balanced_labels.extend([label] * len(indices))
                    
                    all_segments.extend(balanced_segments)
                    all_labels.extend(balanced_labels)
                
                processed_records += 1
                print(f"\nRecord {record_num} ({processed_records}/{total_records}):")
                print(f"Original distribution: {Counter(labels_for_record)}")
                if len(label_counts) >= 2:
                    print(f"Balanced distribution: {Counter(balanced_labels)}")
                
            except Exception as e:
                print(f"Error processing record {record_num}: {str(e)}")
                continue
        
        print("\nBefore final balancing:")
        print(Counter(all_labels))
        
      
        label_counts = Counter(all_labels)
        target_count = max(min(label_counts.values()) * 5, 500) 
        
        balanced_segments = []
        balanced_labels = []
        
        for label in valid_types:
            if label in label_counts and label_counts[label] >= 50: 
                label_indices = [i for i, l in enumerate(all_labels) if l == label]
                if len(label_indices) >= target_count:
                    selected_indices = np.random.choice(label_indices, target_count, replace=False)
                else:
                    selected_indices = np.random.choice(label_indices, target_count, replace=True)
                
                balanced_segments.extend([all_segments[i] for i in selected_indices])
                balanced_labels.extend([label] * target_count)
        
        print("\nAfter final balancing:")
        print(Counter(balanced_labels))
        
        return np.array(balanced_segments), np.array(balanced_labels)
    
    def preprocess_beats(self, segments, labels, window_size=200):
   
        try:
         
            segments = np.array([seg[:window_size] if len(seg) > window_size 
                            else np.pad(seg, (0, window_size - len(seg))) 
                            for seg in segments])
            
            scaler = StandardScaler()
            scaled_segments = np.array([scaler.fit_transform(seg.reshape(-1, 1)).ravel() 
                                    for seg in segments])
            
            aligned_segments = []
            for seg in scaled_segments:
                try:
                    r_peak_idx = np.argmax(seg)
                    center = len(seg) // 2
                    shift = center - r_peak_idx
                    aligned_seg = np.roll(seg, shift)
                    aligned_segments.append(aligned_seg)
                except Exception as e:
                    print(f"Warning: R-peak alignment failed for a segment: {str(e)}")
                    aligned_segments.append(seg)  
            
            filtered_segments = np.array([self.filter_ecg(seg) for seg in aligned_segments])
            
            print(f"Preprocessed {len(segments)} segments")
            return filtered_segments
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return segments  
        
    def extract_features(self, segments):
   
        features = []
        for seg in segments:
            try:
              
                stats = [
                    np.mean(seg),
                    np.std(seg),
                    np.max(seg),
                    np.min(seg),
                    np.median(seg),
                    np.percentile(seg, 25), 
                    np.percentile(seg, 75),
                    np.sum(np.abs(np.diff(seg))) 
                ]
                
             
                peaks, properties = sig.find_peaks(seg, height=0)
                if len(peaks) > 1:
                    rr_features = [
                        np.mean(np.diff(peaks)),
                        np.std(np.diff(peaks)),
                        len(peaks), 
                        np.mean(properties['peak_heights']) 
                    ]
                else:
                    rr_features = [0, 0, 0, 0]
            
                fft_features = np.abs(np.fft.fft(seg))[:10]  
             
                combined = np.concatenate([seg, stats, rr_features, fft_features])
                features.append(combined)
                
            except Exception as e:
                print(f"Warning: Feature extraction failed for a segment: {str(e)}")
    
                features.append(np.zeros_like(features[0]) if features else np.zeros(len(seg) + 22))
        
        return np.array(features)
    
    def prepare_snn_data(self, X, y, test_size=0.2, val_size=0.1):
     
        try:
           
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
       
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), 
                stratify=y_temp, random_state=42
            )
            
       
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"Warning: SMOTE failed ({str(e)}). Using original data.")
                X_train_balanced, y_train_balanced = X_train, y_train
            
           
            tensors = {
                'train_x': torch.FloatTensor(X_train_balanced),
                'train_y': torch.LongTensor(y_train_balanced),
                'val_x': torch.FloatTensor(X_val),
                'val_y': torch.LongTensor(y_val),
                'test_x': torch.FloatTensor(X_test),
                'test_y': torch.LongTensor(y_test)
            }
            
      
            print("\nDataset split information:")
            print(f"Training set: {len(X_train_balanced)} samples")
            print(f"Validation set: {len(X_val)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            return tensors, le.classes_
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            return None

    def process_data(self):
  
        X_all, y_all = self.load_multiple_records()
        
        X_processed = self.preprocess_beats(X_all, y_all)
        X_features = self.extract_features(X_processed)
        
        train_data = self.prepare_snn_data(X_features, y_all)
        
        if train_data is not None:
            processed_data = {
                'X_train': train_data[0]['train_x'],
                'y_train': train_data[0]['train_y'],
                'X_val': train_data[0]['val_x'],
                'y_val': train_data[0]['val_y'],
                'X_test': train_data[0]['test_x'],
                'y_test': train_data[0]['test_y'],
                'classes': train_data[1],
                'sampling_rate': self.sampling_rate,
                'feature_shape': X_features.shape[1:]
            }
            return processed_data
        return None 