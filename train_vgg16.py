# --- (接續 main.py / 步驟三之後) ---
# 確保您已經 import 了所有需要的工具
# (如果您是開新檔案，請複製步驟三開頭的 import 語句)
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model # 我們需要這個來組合模型

print("\n--- 步驟四：建立遷移學習 (VGG16) 模型 ---")

# --- 1. 再次載入數據 (如果是在新檔案中) ---
# (如果您是接續 main.py，可以跳過這段，直接使用已載入的變數)
try:
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        CATEGORIES = data['categories']
        IMG_SIZE = data['img_size']
        print("數據載入成功！")
except FileNotFoundError:
    print("錯誤：找不到 'processed_data.pkl' 檔案。")
    exit()
except NameError: # 如果變數已存在，就略過
    print("數據已載入，繼續執行。")
    pass


# --- 2. 載入 VGG16 預訓練基礎模型 ---
# include_top=False: 這代表我們「不要」VGG16 最後的全連接分類層
# weights='imagenet': 載入在 ImageNet 上預訓練的權重
# input_shape: 我們的影像大小 (150x150x3)
base_model_vgg16 = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# --- 3. 凍結 VGG16 的權重 ---
# 我們告訴 Keras：不要在訓練期間更動 VGG16 基礎模型的權重
# 我們只把它當作一個固定的「特徵提取器」
base_model_vgg16.trainable = False

print("\nVGG16 基礎模型載入完畢，並已凍結。")

# --- 4. 建立我們自己的「分類層」 ---
# 我們要把 VGG16 的輸出 (一個 3D 張量) 接到我們自己的層
# 取得 VGG16 的輸出
x = base_model_vgg16.output

# 加上我們自己的層
x = Flatten()(x) # 攤平
x = Dense(128, activation='relu')(x) # 全連接層 (類似論文)
x = Dropout(0.5)(x) # 防止過擬合
predictions = Dense(len(CATEGORIES), activation='softmax')(x) # 我們的 4 類別輸出層

# --- 5. 組合新模型 ---
# 將「VGG16基礎模型」和我們新的「分類層」組合在一起
model_vgg16 = Model(inputs=base_model_vgg16.input, outputs=predictions)

# --- 6. 編譯模型 ---
model_vgg16.compile(
    loss='categorical_crossentropy',
    optimizer='adam', # 使用 Adam 優化器
    metrics=['accuracy']
)

# 顯示新模型的架構
# 您會看到 VGG16 的層數非常多，且大部分都是 'non-trainable' (不可訓練)
model_vgg16.summary()

# --- 7. 定義回調函數 (Callbacks) ---
# (和步驟三相同，但使用不同的檔案名)
early_stop_vgg16 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

checkpoint_vgg16 = ModelCheckpoint(
    'best_vgg16_model.keras', # 保存模型的新文件名
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# --- 8. 訓練遷移學習模型 ---
print("\n開始訓練 VGG16 (遷移學習) 模型...")

history_vgg16 = model_vgg16.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30, # 同樣設定 30，但 EarlyStopping 可能會提早停止
    validation_data=(X_val, y_val),
    callbacks=[early_stop_vgg16, checkpoint_vgg16]
)

print("VGG16 模型訓練完畢！")
print("最好的模型已保存為 'best_vgg16_model.keras'")

# --- 9. 繪製訓練過程圖 ---
# (我們可以使用步驟三中定義的 plot_history 函數)
# (如果您是開新檔案，請複製 plot_history 函數的定義)

# 檢查 plot_history 函數是否已定義
if 'plot_history' not in globals():
    def plot_history(history, model_name):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'ro--', label='Validation Accuracy')
        plt.title(f'{model_name} - Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro--', label='Validation Loss')
        plt.title(f'{model_name} - Loss')
        plt.legend()
        plt.savefig(f'{model_name}_history.png')
        print(f"訓練過程圖已保存為 '{model_name}_history.png'")

plot_history(history_vgg16, 'vgg16_transfer_learning')