import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- (!!! 這就是新加入的程式碼 !!!) ---
# 解決 Mac 上 Matplotlib 中文顯示為方塊的問題
# 我們告訴 matplotlib 使用 'PingFang TC' (蘋方-TC) 或 'STHeiti' (華文黑體)
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['PingFang TC', 'STHeiti', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
    print("已設定中文字體為 'PingFang TC' 或 'STHeiti'。")
except Exception as e:
    print(f"設定中文字體失敗: {e}")
    print("您的圖片中可能仍會顯示方塊。")
# --- (!!! 新程式碼結束 !!!) ---


print("\n--- 步驟六：最終模型評估 ---")

# --- 1. 載入數據 ---
print("正在載入 'processed_data.pkl'...")
try:
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        X_test = data['X_test']
        y_test_one_hot = data['y_test']
        CATEGORIES = data['categories']
        IMG_SIZE = data['img_size']
    print("測試集數據載入成功！")
except FileNotFoundError:
    print("錯誤：找不到 'processed_data.pkl' 檔案。")
    exit()

# 轉換標籤 (從 One-Hot [0,0,1,0] 轉為 2)
y_test_simple = np.argmax(y_test_one_hot, axis=1)

# --- 2. 輔助函數：重建模型 ---

def build_custom_cnn_and_extractor(img_size, num_classes):
    """重建步驟三的 CNN 模型 (使用 Functional API)"""
    inputs = Input(shape=(img_size, img_size, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    features = Dense(128, activation='relu', name='dense')(x)
    x_dropped = Dropout(0.5)(features)
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x_dropped)
    
    full_model = Model(inputs=inputs, outputs=outputs)
    feature_extractor = Model(inputs=inputs, outputs=features)
    return full_model, feature_extractor

def build_vgg16_model(img_size, num_classes):
    """重建步驟四的 VGG16 模型 (使用 Functional API)"""
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    """繪製並保存混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {filename}')
    plt.savefig(f'cm_{filename}.png')
    print(f"混淆矩陣圖已保存為 'cm_{filename}.png'")

# --- 3. 評估模型一：自訂 CNN ---
print("\n--- 正在評估 [模型一：自訂 CNN] ---")
model_cnn, feature_extractor = build_custom_cnn_and_extractor(IMG_SIZE, len(CATEGORIES))
model_cnn.load_weights('best_cnn_model.keras')
pred_cnn_probs = model_cnn.predict(X_test, batch_size=32)
y_pred_cnn = np.argmax(pred_cnn_probs, axis=1)
report_cnn = classification_report(y_test_simple, y_pred_cnn, target_names=CATEGORIES, output_dict=True)
plot_confusion_matrix(y_test_simple, y_pred_cnn, CATEGORIES, 'Custom_CNN')

# --- 4. 評估模型二：VGG16 遷移學習 ---
print("\n--- 正在評估 [模型二：VGG16 遷移學習] ---")
model_vgg16 = build_vgg16_model(IMG_SIZE, len(CATEGORIES))
model_vgg16.load_weights('best_vgg16_model.keras')
pred_vgg16_probs = model_vgg16.predict(X_test, batch_size=32)
y_pred_vgg16 = np.argmax(pred_vgg16_probs, axis=1)
report_vgg16 = classification_report(y_test_simple, y_pred_vgg16, target_names=CATEGORIES, output_dict=True)
plot_confusion_matrix(y_test_simple, y_pred_vgg16, CATEGORIES, 'VGG16_Transfer_Learning') # (修正檔名)

# --- 5. 評估模型三：隨機森林 (RF) ---
print("\n--- 正在評估 [模型三：隨機森林] ---")
print(" (正在為 RF 提取特徵...)")
X_test_features = feature_extractor.predict(X_test, batch_size=32)
model_rf = joblib.load('rf_model.joblib')
y_pred_rf = model_rf.predict(X_test_features)
report_rf = classification_report(y_test_simple, y_pred_rf, target_names=CATEGORIES, output_dict=True)
plot_confusion_matrix(y_test_simple, y_pred_rf, CATEGORIES, 'Random_Forest')

# --- 6. 產生最終的「比較總表」（論文的 Table 2） ---
print("\n\n" + "="*50)
print("     期中報告：最終模型性能比較總表 (測試集)")
print("="*50)

models = {
    "CNN": report_cnn,
    "VGG16 遷移學習": report_vgg16,
    "隨機森林 (RF)": report_rf
}

print(f"{'模型名稱':<18} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-"*65)

table_data = []
row_labels = []
col_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for name, report in models.items():
    row_labels.append(name) # (修復) 我們將在這裡使用中文
    
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    
    print(f"{name:<18} | {accuracy:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {f1_score:<10.4f}")
    
    table_data.append([
        f"{accuracy:.4f}", 
        f"{precision:.4f}", 
        f"{recall:.4f}", 
        f"{f1_score:.4f}"
    ])

print("="*50)

# --- 7. (修復) 將總表儲存為圖片 ---
print("\n正在將總表儲存為圖片 'final_comparison_table.png'...")

fig, ax = plt.subplots(figsize=(10, 3)) 
ax.axis('tight')
ax.axis('off')

the_table = ax.table(
    cellText=table_data,
    rowLabels=row_labels, # 這裡現在會是中文
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.2, 1.2) 

# (修復) 因為我們在頂部設定了 plt.rcParams，這裡不需額外設定字體
plt.savefig('final_comparison_table.png', bbox_inches='tight', dpi=200) 
print("總表圖片儲存完畢！")

print("\n步驟六 (評估) 已完成！")