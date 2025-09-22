#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
  echo "用法: $0 <命令...>"
  echo "示例: $0 python main.py --is_transfer True --test_dataset Computers --pretrained_model_name ..."
  exit 1
fi

values=()

for i in 1 2 3 4 5; do
  echo "[Run ${i}/5] 开始执行: $@"
  tmp_log=$(mktemp)
  "$@" 2>&1 | tee "$tmp_log"
  # 1) 优先使用 "Best @ epoch ... test=..." 行中的 test 值
  best_test=$(grep -E 'Best .*test=' "$tmp_log" | tail -n 1 | sed -n 's/.*test=\([0-9.][0-9.]*\).*/\1/p' | tail -n 1 || true)
  if [ -n "$best_test" ]; then
    test_val="$best_test"
  else
    # 2) 回退：使用最后一个 train/val/test 三元组中的 test 值
    last_tuple=$(grep -o 'train/val/test=[0-9.\/]\+' "$tmp_log" | tail -n 1 || true)
    if [ -z "$last_tuple" ]; then
      echo "未在输出中解析到 Best/三元组格式 (Best ... test=... 或 train/val/test=...)，请检查命令输出。"
      rm -f "$tmp_log"
      exit 1
    fi
    test_val=$(echo "$last_tuple" | awk -F'=' '{print $2}' | awk -F'/' '{print $3}')
  fi
  if [ -z "$test_val" ]; then
    echo "解析 test 值失败，请检查输出格式。"
    rm -f "$tmp_log"
    exit 1
  fi
  echo "[Run ${i}/5] 解析到 test=${test_val}"
  values+=("$test_val")
  rm -f "$tmp_log"
done

# 计算均值与标准差，并乘以100后以两位小数输出
printf '%s\n' "${values[@]}" | awk '
{
  x[NR] = $1; sum += $1; sum2 += ($1*$1);
}
END {
  n = NR;
  mean = sum / n;
  std = sqrt(sum2 / n - mean * mean);
  printf "结果(×100): %.2f±%.2f\n", mean * 100, std * 100;
}
'


