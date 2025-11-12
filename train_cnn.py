# --- (接續 main.py) ---
# 或者，如果您在一個新檔案中，請確保 import pickle, numpy, tensorflow 等
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

print("\n--- 步驟三：建立自訂 CNN 模型 ---")

# --- 1. 載入我們在步驟二處理好的數據 ---
print("正在載入 'processed_data.pkl'...")
try:
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        CATEGORIES = data['categories']
        IMG_SIZE = data['img_size']
        print("數據載入成功！")
except FileNotFoundError:
    print("錯誤：找不到 'processed_data.pkl' 檔案。")
    print("請先執行步驟二 (main.py 的前半部分) 來產生此檔案。")
    exit() # 如果沒有數據，就無法繼續

# --- 2. 定義 CNN 模型架構 ---
# 我們的架構將模仿論文 ，但會針對 4 個類別進行調整
# 論文架構: Conv(32) -> Pool -> Conv(64) -> Pool -> Dense(128) -> Output(2) 
# 我們的架構: Conv(32) -> Pool -> Conv(64) -> Pool -> Dense(128) -> Dropout -> Output(4)

model_cnn = Sequential()

# 輸入層 (Input Layer) + 第一個卷積/池化層
model_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

# 第二個卷積/池化層
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

# 攤平層 (Flatten)
model_cnn.add(Flatten())

# 全連接層 (Dense Layer) + Dropout
# 論文使用了 128 個神經元 [cite: 163]
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5)) # Dropout 是一種防止過擬合 (Overfitting) 的技術

# 輸出層 (Output Layer)
# 論文是 2 個類別 [cite: 163]，我們有 4 個類別
model_cnn.add(Dense(len(CATEGORIES), activation='softmax')) 

# --- 3. 編譯模型 ---
# 我們需要定義損失函數、優化器和評估指標 [cite: 173, 174]
model_cnn.compile(
    loss='categorical_crossentropy', # 多類別分類的標準損失函數
    optimizer='adam',                # 一個高效的優化器
    metrics=['accuracy']             # 我們關心的是準確率 [cite: 174]
)

# 顯示模型架構
model_cnn.summary()

# --- 4. 定義回調函數 (Callbacks) ---
# 這些是幫助我們訓練的工具
# EarlyStopping: 如果模型在 'patience' 個 epoch 內都沒有進步，就提早停止訓練
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ModelCheckpoint: 只保存訓練過程中「最好」的模型
checkpoint = ModelCheckpoint(
    'best_cnn_model.keras', # 保存模型的文件名 (使用 .keras 格式)
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# --- 5. 訓練模型 ---
print("\n開始訓練 CNN 模型...")
# 我們將訓練 30 個週期 (epochs)，但 EarlyStopping 可能會提早結束它
history_cnn = model_cnn.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30,
    validation_data=(X_val, y_val), # 使用驗證集來監控效能
    callbacks=[early_stop, checkpoint]
)

print("CNN 模型訓練完畢！")
print("最好的模型已保存為 'best_cnn_model.keras'")

# --- 6. (選做) 繪製訓練過程圖 ---
# 這對報告很有幫助，類似於論文中的圖 5 (Accuracy Gain Over Epochs) [cite: 241]
def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # 繪製準確率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro--', label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    # 繪製損失
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro--', label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()

    plt.savefig(f'{model_name}_history.png') # 保存圖片
    print(f"訓練過程圖已保存為 '{model_name}_history.png'")

plot_history(history_cnn, 'custom_cnn')