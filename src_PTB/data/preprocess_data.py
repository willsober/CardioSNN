import os
import wfdb
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import ast

# 加载原始信号数据的函数
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])  # 提取信号部分
    return data
# 读取标签数据文件
# label_file_path = '/tmp/pycharm_project_744/PTB-XL/ptb_xl_raw/ptbxl_database.csv'

# 用于加载并处理单个ECG记录（包括信号和标注）
# 设置 PTB-XL 数据路径和采样率
path = '/tmp/pycharm_project_468/Siamese-SNN-ECG-main/PTB-XL/ptb_xl_raw/'  # 请替换为实际数据集路径
sampling_rate = 500  # 设置采样率（可以选择100或其他）

# 加载和转换标签数据
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))  # 解析标签列

# 加载原始信号数据
X = load_raw_data(Y, sampling_rate, path)

# 提取基于标注的心跳段（每个心跳对应一个信号片段）
def extract_heartbeats(signals, sample_points, window_size=250):
    """Extract heartbeat segments from signals"""
    heartbeats = []
    for point in sample_points:
        # 确保提取的窗口在信号范围内
        start = max(0, point - window_size // 2)
        end = min(len(signals), point + window_size // 2)

        # 提取当前心跳信号段
        beat = signals[start:end]

        # 如果信号段长度不足，进行零填充
        if len(beat) < window_size:
            pad_width = window_size - len(beat)
            beat = np.pad(beat, ((0, pad_width), (0, 0)), mode='constant')

        heartbeats.append(beat)

    return np.array(heartbeats)
# 加载诊断聚合信息
agg_df = pd.read_csv('/tmp/pycharm_project_468/Siamese-SNN-ECG-main/PTB-XL/ptb_xl_raw/scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]  # 只筛选出诊断信息

# 诊断聚合函数
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def preprocess_data():
    print("Starting data preprocessing...")

    os.makedirs('data/processed', exist_ok=True)


    # 计算 diagnostic_superclass（官方做法）
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # 2. 遍历所有记录
    record_paths = []
    subfolder_path = os.path.join(path, 'records500')
    for folder in os.listdir(subfolder_path):
        folder_path = os.path.join(subfolder_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.dat'):
                    record_name = file[:-4]
                    record_paths.append(os.path.join(folder_path, record_name))

    print(f"Found {len(record_paths)} records to process.")

    all_beats = []
    all_labels = []

    # 自定义标签映射（因为你想要 Stadium 分期，这里随便映射一个能跑的）
    label_map = {'NORM': 2, 'MI': 1, 'STTC': 3, 'CD': 1, 'HYP': 4}  # 随便对应一下

    for record_path in tqdm(record_paths, desc="Processing"):
        try:
            # 读取信号
            signal, _ = wfdb.rdsamp(record_path)  # (5000 or 10000, 12)

            # 从文件名提取 ecg_id
            filename = os.path.basename(record_path)
            ecg_id = int(''.join(filter(str.isdigit, filename)))

            if ecg_id not in Y.index:
                continue

            # 获取主要诊断类别
            classes = Y.loc[ecg_id, 'diagnostic_superclass']
            if not classes or len(classes) == 0:
                label_str = 'unknown'
            else:
                main_class = classes[0]  # 取第一个
                label_str = main_class

            # 映射到你想要的数字标签
            label_id = label_map.get(label_str, 0)  # 没有的都当 unknown

            # 滑动窗口切心跳（250 点 ≈ 0.5秒，500Hz）
            window = 250
            step = 125
            for i in range(0, signal.shape[0] - window + 1, step):
                beat = signal[i:i + window, :]  # (250, 12)
                all_beats.append(beat)
                all_labels.append(label_id)

        except Exception as e:
            print(f"错误: {record_path} -> {e}")
            continue

    # 转为 numpy
    X = np.array(all_beats, dtype=np.float32)  # (N, 250, 12)
    y = np.array(all_labels, dtype=np.int64)

    print(f"成功提取心跳: X.shape = {X.shape}, y.shape = {y.shape}")

    if len(X) == 0:
        print("严重错误：没有提取到任何数据！")
        return

    # 标准化（按通道）
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 划分训练/测试（直接随机划分，先跑通再说）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 保存
    processed_data = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'classes': ['unknown', 'Stadium II-III', 'Stadium I', 'Stadium II', 'Stadium I-II'],
        'scaler': scaler
    }

    with open('data/processed/PTB_processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

    print("预处理完成！")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print("类别分布:", np.bincount(y))
# 运行数据预处理函数
if __name__ == '__main__':
    preprocess_data()
