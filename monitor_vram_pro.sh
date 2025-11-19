#!/bin/bash

# --- 設定 ---
# 監控間隔（秒）
INTERVAL=10
# 紀錄 VRAM 使用量的日誌檔案名稱
LOG_FILE="vram_usage.log"

# --- 主程式 ---

# 檢查使用者是否有提供要執行的指令
if [ $# -eq 0 ]; then
    echo "錯誤: 請提供要執行和監控的程式指令。"
    echo "用法: ./monitor_vram_pro.sh [您的程式指令]"
    echo "範例: ./monitor_vram_pro.sh python my_app.py --user test"
    exit 1
fi

# 檢查 nvidia-smi 是否存在
if ! command -v nvidia-smi &> /dev/null
then
    echo "錯誤: nvidia-smi 指令不存在。請確認 NVIDIA 驅動已正確安裝。"
    exit 1
fi

# 清除舊的日誌檔案
rm -f $LOG_FILE

echo "VRAM 監控已啟動... (每 $INTERVAL 秒記錄一次)"

# 在背景執行監控迴圈
while true; do
    # 1. 將 nvidia-smi 的輸出存入變數
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    # 2. 使用 echo 將變數內容加上換行符後寫入檔案，確保每筆紀錄都獨立一行
    echo "$VRAM_USED" >> $LOG_FILE
    sleep $INTERVAL
done &

# 取得背景監控程序的 PID
MONITOR_PID=$!

# "$@" 會代表所有傳入腳本的參數
YOUR_PROGRAM="$@"

echo "正在執行您的指令: $YOUR_PROGRAM"
echo "程式輸出:"
echo "------------------------------------------"

# --- [新增] 紀錄開始時間 (使用 UNIX timestamp) ---
start_time=$(date +%s)

# 執行從外部傳入的完整指令
eval "$YOUR_PROGRAM"

# --- [新增] 紀錄結束時間 並 計算總秒數 ---
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "------------------------------------------"
echo "您的指令已執行完畢。"

# 停止背景的監控程序
kill $MONITOR_PID
echo "VRAM 監控已停止。"

# 使用 awk 分析日誌檔案
echo "分析 VRAM 使用量..."
awk '
# 過濾掉空行，以防萬一
/./ {
    sum += $1;
    if ($1 > max) {
        max = $1;
    }
    count++;
}
END {
    if (count > 0) {
        printf "VRAM 使用量分析結果 (針對整個執行過程):\n"
        printf " - 高峰值 (Peak): %.2f MiB\n", max
        printf " - 平均值 (Average): %.2f MiB\n", sum / count
        printf " - 總共紀錄 %d 筆有效資料\n", count
    } else {
        print "沒有紀錄到任何 VRAM 資料。"
    }
}
' $LOG_FILE

# --- [新增] 顯示總執行時間 ---
echo "程式執行時間分析:"
minutes=$((duration / 60))
seconds=$((duration % 60))
printf " - 總執行時間: %d 分 %d 秒 (共 %d 秒)\n" $minutes $seconds $duration


# 清除日誌檔案
# rm -f $LOG_FILE