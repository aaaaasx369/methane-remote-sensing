# Finish Multisite Remaining Work

這個工具會完成：

1. GHGSat / PRISMA inventory audit 與 temporal matching
2. strict zero-shot 與 site-calibrated feature tables
3. numerical wind-table audit
4. wind record matching
5. wind-aligned Sentinel-2 features

## 安裝

```bash
cd ~/Downloads
unzip -o finish_multisite_remaining.zip -d finish_multisite_remaining

cp -f   ~/Downloads/finish_multisite_remaining/finish_multisite_remaining.py   /Users/happydoraaa/methane_release_project/
```

## 啟動環境

```bash
cd /Users/happydoraaa/methane_release_project
source .venv/bin/activate
python -m pip install pandas numpy rasterio
```

## 先跑不需要指定 wind table 的部分

```bash
python finish_multisite_remaining.py all   --project-root /Users/happydoraaa/methane_release_project
```

這會建立：

```text
outputs/43_ghgsat_prisma_inventory_audit.csv
outputs/44_ghgsat_prisma_temporal_matches.csv
outputs/40_multisite_s2_features_strict_zero_shot.csv
outputs/41_multisite_s2_features_site_calibrated.csv
outputs/45_wind_table_audit.csv
```

## 檢查 wind table

```bash
cat outputs/45_wind_table_audit_report.txt
```

如果報告中有 valid_times、valid_speed、valid_direction 都大於 0 的表，明確指定：

```bash
python finish_multisite_remaining.py wind-match   --project-root /Users/happydoraaa/methane_release_project   --wind-table outputs/你的有效風場資料.csv   --wind-max-hours 12
```

然後：

```bash
python finish_multisite_remaining.py wind-features   --project-root /Users/happydoraaa/methane_release_project
```

## 需要額外指定 GHGSat / PRISMA table

```bash
python finish_multisite_remaining.py sensor-match   --project-root /Users/happydoraaa/methane_release_project   --sensor-table outputs/ghgsat_table.csv   --sensor-table outputs/prisma_table.csv
```

## 重要限制

- GHGSat/PRISMA 配對必須有 event ID、site 或座標連結；不接受純時間巧合。
- strict zero-shot 會排除 temporal-reference 與 wind-aligned features。
- site-calibrated 會保留 temporal-reference features。
- wind direction 使用氣象學 FROM direction，plume downwind direction = direction + 180 degrees。
- 若沒有真正數值 wind speed/direction，程式會停止，不會猜測。
