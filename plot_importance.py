import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 解決 Matplotlib 中文顯示為方塊的問題 ---
# (和 evaluate.py 中一樣的修復程式碼)
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['PingFang TC', 'STHeiti', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
    print("已設定中文字體為 'PingFang TC' 或 'STHeiti'。")
except Exception as e:
    print(f"設定中文字體失敗: {e}")

# --- 2. 載入已訓練的隨機森林模型 ---
print("正在載入 'rf_model.joblib'...")
try:
    model_rf = joblib.load('rf_model.joblib')
    print("模型載入成功！")
except FileNotFoundError:
    print("錯誤：找不到 'rf_model.joblib' 檔案。")
    print("請確保您已成功執行 步驟五 (train_rf.py)。")
    exit()

# --- 3. 獲取特徵重要性 ---
importances = model_rf.feature_importances_

# 我們只顯示前 15 個最重要的特徵
top_n = 15
indices = np.argsort(importances)[-top_n:][::-1] # 獲取前 N 個的索引
top_importances = importances[indices]
top_labels = [f"特徵編號 {i}" for i in indices]

print(f"前 {top_n} 名最重要的特徵：")
for i in range(top_n):
    print(f"  {top_labels[i]}: {top_importances[i]:.4f}")

# --- 4. 繪製水平長條圖 ---
plt.figure(figsize=(12, 8)) # (寬, 高)

# 繪製水平長條圖
plt.barh(top_labels, top_importances, color='c') # 'c' = cyan (青色)

plt.xlabel('重要性 (Importance Score)')
plt.ylabel('由 CNN 提取的特徵編號')
plt.title(f'隨機森林 - 前 {top_n} 名特徵重要性')

# 將 y 軸反轉，讓最重要的特徵顯示在最上面
plt.gca().invert_yaxis()

# 在長條圖右側顯示數值
for index, value in enumerate(top_importances):
    plt.text(value, index, f' {value:.4f}', va='center')

# --- 5. 儲存圖表 ---
# bbox_inches='tight' 會自動裁切掉多餘的白邊
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=200)
print("\n特徵重要性圖表已保存為 'feature_importance.png'")