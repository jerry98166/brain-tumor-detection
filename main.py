import os
import cv2  # 這是 OpenCV
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle # 用於保存處理好的數據

# --- 1. 定義常數 ---
IMG_SIZE = 150 # 將所有影像縮放到 150x150
# Kaggle 資料集已分為 'Training' 和 'Testing'，我們把它們都讀進來
DATA_DIRS = ['Training', 'Testing'] 
CATEGORIES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# 用於保存所有數據和標籤的列表
data = []
labels = []

print("開始讀取影像...")

# --- 2. 載入、縮放、正規化影像 ---
for data_dir in DATA_DIRS:
    # os.getcwd() 會獲取您當前執行 .py 檔案的資料夾路徑
    path = os.path.join(os.getcwd(), data_dir) 
    
    if not os.path.exists(path):
        print(f"警告: 找不到資料夾 {path}")
        print("請確保 'Training' 和 'Testing' 資料夾與您的 .py 檔案在同一個目錄下。")
        continue

    for category in CATEGORIES:
        class_num = CATEGORIES.index(category) # 將標籤轉為數字 (0, 1, 2, 3)
        class_path = os.path.join(path, category)
        
        if not os.path.exists(class_path):
            print(f"警告: 找不到子資料夾 {class_path}")
            continue

        for img_name in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, img_name)
                # 讀取影像 (cv2 預設以 BGR 格式讀取)
                img_array = cv2.imread(img_path) 
                # 縮放影像
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                # 添加到我們的數據列表
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                # 忽略損壞的影像（例如 .DS_Store 或其他非影像檔）
                # print(f"略過檔案： {img_path}。 錯誤： {e}")
                pass

print(f"影像讀取完畢！總共 {len(data)} 張影像。")

# --- 3. 轉換為 Numpy 陣列並正規化 ---
# 將 Python 列表轉換為 Numpy 陣列，這是模型需要的格式
data = np.array(data)
labels = np.array(labels)

# 正規化 (Normalization): 將像素值從 0-255 縮放到 0-1
data = data.astype('float32') / 255.0

# --- 4. 處理標籤 ---
# 將標籤 (例如 1, 2, 3) 轉換為 "One-Hot" 編碼
# 範例： 2 -> [0, 0, 1, 0]
# 這是 CNN 進行多類別分類所必需的
labels_one_hot = to_categorical(labels, num_classes=len(CATEGORIES))

# --- 5. 分割資料 ---
# 我們將所有讀入的資料（Training+Testing）重新分割，以建立我們自己的驗證集
# 第一次分割：將所有資料分為「訓練+驗證」集 (80%) 和「測試集」 (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    data, labels_one_hot, 
    test_size=0.20,       # 20% 作為測試集
    random_state=42,      # 確保每次分割結果都一樣
    stratify=labels_one_hot # 確保每個類別在分割後比例相同
)

# 第二次分割：將「訓練+驗證」集分為「訓練集」和「驗證集」
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=0.25, # 0.25 * 80% = 20% (所以最終比例是 60% 訓練, 20% 驗證, 20% 測試)
    random_state=42, 
    stratify=y_train_val 
)

print("\n資料分割完畢：")
print(f"訓練集 (Training):   {X_train.shape[0]} 張影像, shape: {X_train.shape}")
print(f"驗證集 (Validation): {X_val.shape[0]} 張影像, shape: {X_val.shape}")
print(f"測試集 (Test):       {X_test.shape[0]} 張影像, shape: {X_test.shape}")

# --- 6. (推薦) 保存處理好的數據 ---
# 這樣您下次就不需要再重新跑一次前處理，可以直接載入
print("\n正在保存處理好的數據到 'processed_data.pkl' ...")
with open('processed_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'categories': CATEGORIES,
        'img_size': IMG_SIZE
    }, f)

print("步驟二 (資料前處理) 已完成！")
print("您的專案資料夾中現在應該有一個 'processed_data.pkl' 檔案。")