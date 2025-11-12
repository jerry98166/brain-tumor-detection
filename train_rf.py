import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
# (修復) 我們需要 Input 來使用 Functional API
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input 
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib 

print("\n--- 步驟五：建立隨機森林 (RF) 模型 (Functional API 修復版) ---")

# --- 1. 載入我們在步驟二處理好的數據 ---
print("正在載入 'processed_data.pkl'...")
try:
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train_one_hot = data['y_train']
        X_val = data['X_val']
        y_val_one_hot = data['y_val']
        X_test = data['X_test']
        y_test_one_hot = data['y_test']
        CATEGORIES = data['categories']
        IMG_SIZE = data['img_size']
    print("數據載入成功！")
except FileNotFoundError:
    print("錯誤：找不到 'processed_data.pkl' 檔案。")
    exit()

# --- 2. (!!! 最終修復 !!!) ---
# 放棄 Sequential()，使用 Functional API 重建完全相同的架構
print("正在重新建立 CNN 模型架構 (Functional API)...")

# 1. 明確定義 Input 張量
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# 2. 像鍊條一樣手動連接每一層
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
# 3. 抓住我們想要的特徵層
features = Dense(128, activation='relu', name='dense')(x)
x_dropped = Dropout(0.5)(features)
outputs = Dense(len(CATEGORIES), activation='softmax', name='output_layer')(x_dropped)

# 4. 建立「完整模型」
full_cnn_model = Model(inputs=inputs, outputs=outputs)
# ----------------------------------------

print("架構建立完畢。正在載入權重 'best_cnn_model.keras'...")

try:
    # 將步驟三的權重載入到這個新架構中 (架構相同，所以權重可以完美匹配)
    full_cnn_model.load_weights('best_cnn_model.keras')
    print("權重載入成功！")
    
except Exception as e:
    print(f"錯誤：無法載入權重。請確保 'best_cnn_model.keras' 檔案存在且未損壞。")
    print(f"錯誤訊息: {e}")
    exit()

# --- 3. 建立特徵提取器 ---
print("正在建立特徵提取器...")
# 這次，`inputs` 和 `features` 都是已定義的張量
# 我們可以安全地建立一個只到 'dense' 層的模型
feature_extractor = Model(inputs=inputs, outputs=features)
# ----------------------------------------

# --- 4. 提取特徵 (Feature Extraction) ---
print("正在為 訓練集(X_train) 提取特徵...")
X_train_features = feature_extractor.predict(X_train, batch_size=32)
print("正在為 驗證集(X_val) 提取特徵...")
X_val_features = feature_extractor.predict(X_val, batch_size=32)
print("正在為 測試集(X_test) 提取特徵...")
X_test_features = feature_extractor.predict(X_test, batch_size=32)

print("\n特徵提取完畢！")
print(f"提取後 X_train_features shape: {X_train_features.shape}")

# --- 5. 轉換標籤 (Labels) ---
y_train_simple = np.argmax(y_train_one_hot, axis=1)
y_val_simple = np.argmax(y_val_one_hot, axis=1)
y_test_simple = np.argmax(y_test_one_hot, axis=1)

# --- 6. 訓練隨機森林 (Random Forest) ---
print("\n開始訓練隨機森林 (Random Forest) 模型...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

X_train_val_features = np.concatenate((X_train_features, X_val_features))
y_train_val_simple = np.concatenate((y_train_simple, y_val_simple))

rf_model.fit(X_train_val_features, y_train_val_simple)
print("隨機森林模型訓練完畢！")

# --- 7. 評估隨機森林模型 ---
print("\n--- 隨機森林 (RF) 評估報告 (測試集) ---")
y_pred_rf = rf_model.predict(X_test_features)
print(classification_report(y_test_simple, y_pred_rf, target_names=CATEGORIES))

# --- 8. (重要!) 討論特徵重要性 ---
print("\n--- 隨機森林 (RF) 特徵重要性 ---")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1] # 排序

print("RF 模型發現，在 CNN 提取的 128 個特徵中：")
for i in range(10): # 只顯示前 10 個最重要的
    print(f"第 {i+1} 重要特徵: 特徵編號 {indices[i]} (重要性: {importances[indices[i]]:.4f})")

# --- 9. 保存 RF 模型 ---
joblib.dump(rf_model, 'rf_model.joblib')
print("\n隨機森林模型已保存為 'rf_model.joblib'")
print("步驟五 (隨機森林) 已完成！")