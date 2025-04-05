import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import glob

# 加载音频文件并提取梅尔频谱图
def load_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# 加载数据集
def load_dataset(base_path, max_length=128):
    X = []
    y = []
    for label, category in enumerate(['woman', 'man']):
        folder_path = os.path.join(base_path, category)
        if not os.path.isdir(folder_path):
            print(f"Directory {folder_path} does not exist.")
            continue
        pattern = os.path.join(folder_path, '*.mp3')
        files = glob.glob(pattern)
        print(f'Loading {category} files from {pattern}, found {len(files)} files.')
        for file_path in files:
            print(f'Trying to load file: {file_path}')
            mel_spectrogram = load_audio(file_path)
            if mel_spectrogram is not None:
                if mel_spectrogram.shape[1] < max_length:
                    mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, max_length - mel_spectrogram.shape[1])), mode='constant')
                elif mel_spectrogram.shape[1] > max_length:
                    mel_spectrogram = mel_spectrogram[:, :max_length]
                X.append(mel_spectrogram)
                y.append(label)
            else:
                print(f'Failed to load file: {file_path}')
    X = np.array(X)
    y = np.array(y)
    return X, y

# 设置数据路径
base_path = os.path.join(os.path.expanduser('~'), 'Desktop', '人声画像（区分性别')
print(f'Current working directory: {base_path}')

# 加载和预处理数据
X, y = load_dataset(base_path)
print(f'Total samples loaded: {len(X)}')

# 检查数据是否加载正确
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Please check the audio file paths and file formats.")

# 预处理数据
X = np.expand_dims(X, axis=-1)  # 添加通道维度

# 将标签转换为独热编码
y = tf.keras.utils.to_categorical(y, num_classes=2)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义卷积神经网络
input_shape = (X.shape[1], X.shape[2], 1)
num_classes = 2

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 加载和预处理单个音频文件进行测试
def predict_audio(file_path, max_length=128):
    mel_spectrogram = load_audio(file_path)
    if mel_spectrogram is not None:
        if mel_spectrogram.shape[1] < max_length:
            mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, max_length - mel_spectrogram.shape[1])), mode='constant')
        elif mel_spectrogram.shape[1] > max_length:
            mel_spectrogram = mel_spectrogram[:, :max_length]
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # 添加通道维度
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)   # 添加批次维度
        prediction = model.predict(mel_spectrogram)
        class_idx = np.argmax(prediction, axis=1)
        return 'woman' if class_idx == 0 else 'man'
    else:
        return "Error loading audio."

# 测试单个音频文件
file_path = 'test_m.mp3'  # 替换为实际的测试音频文件路径
result = predict_audio(file_path)
print(f'The audio is spoken by a(n) {result} person.')
