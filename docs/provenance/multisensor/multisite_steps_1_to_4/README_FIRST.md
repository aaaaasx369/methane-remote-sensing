# Multisite Sentinel-2：Step 1–4

本工具會產生：

```text
outputs/36_multisite_s2_master_table.csv
outputs/37_multisite_s2_availability.csv
outputs/38_cross_sensor_temporal_matches.csv
outputs/39_multisite_s2_features.csv
```

另外也會建立 summary 與 report。

## 安裝

下載 ZIP 後，在 Terminal 執行：

```bash
cd ~/Downloads

ZIP_FILE="$(find "$HOME/Downloads" -maxdepth 1 -type f \
  -iname 'multisite_steps_1_to_4*.zip' -print -quit)"

test -n "$ZIP_FILE" || {
  echo "找不到 multisite_steps_1_to_4 ZIP"
  exit 1
}

rm -rf "$HOME/Downloads/multisite_steps_1_to_4"

unzip -o "$ZIP_FILE" \
  -d "$HOME/Downloads/multisite_steps_1_to_4"

cd "$HOME/Downloads/multisite_steps_1_to_4"

bash install_multisite_pipeline.sh
```

## 先跑本機版本

```bash
cd /Users/happydoraaa/methane_release_project
source .venv/bin/activate

python multisite_pipeline.py all \
  --project-root /Users/happydoraaa/methane_release_project
```

這會完成：

1. master table；
2. 本機 GeoTIFF availability audit；
3. 以現有 CSV 做跨感測器時間配對；
4. 特徵抽取。

## Step 1：Master table

```bash
python multisite_pipeline.py master \
  --project-root /Users/happydoraaa/methane_release_project
```

預設優先使用：

```text
outputs/548_five_site_multisource_manifest_v1.csv
outputs/530_five_site_master_manifest_v3.csv
outputs/500_multisource_canonical_table_v2.csv
```

預設尋找：

```text
outputs/309_all_exact_release_intervals_for_s2.csv
```

若你知道 wind table：

```bash
python multisite_pipeline.py master \
  --project-root /Users/happydoraaa/methane_release_project \
  --wind-table outputs/你的風場資料.csv
```

查看：

```bash
cat outputs/36_multisite_s2_master_report.txt
```

## Step 2：B11/B12、cloud、snow、invalid audit

本機影像：

```bash
python multisite_pipeline.py audit \
  --project-root /Users/happydoraaa/methane_release_project
```

本機六波段 patch 沒有 SCL，因此要正式檢查 cloud/snow，請用 Earth Engine：

```bash
export EE_PROJECT="methane-release-gee"

python multisite_pipeline.py audit \
  --project-root /Users/happydoraaa/methane_release_project \
  --use-ee \
  --ee-project "$EE_PROJECT"
```

查看：

```bash
cat outputs/37_multisite_s2_availability_report.txt
```

## Step 3：跨感測器時間配對

```bash
python multisite_pipeline.py matches \
  --project-root /Users/happydoraaa/methane_release_project
```

程式自動掃描檔名包含：

```text
methaneair
landsat
ghgsat
prisma
multisatellite
historical
```

只接受：

- event ID 相同；
- site 相同；
- 或座標在 15 km 內。

時間接近但沒有空間／事件連結，不會硬配對。

若 historical table 不在預設位置：

```bash
python multisite_pipeline.py matches \
  --project-root /Users/happydoraaa/methane_release_project \
  --extra-table outputs/你的_historical_multisatellite_table.csv
```

額外用 Earth Engine 找 Landsat 8/9：

```bash
export EE_PROJECT="methane-release-gee"

python multisite_pipeline.py matches \
  --project-root /Users/happydoraaa/methane_release_project \
  --use-ee-landsat \
  --ee-project "$EE_PROJECT"
```

查看：

```bash
cat outputs/38_cross_sensor_temporal_report.txt
```

GHGSat 或 PRISMA 為 0 時，通常是缺少同時包含 acquisition time 與 site／coordinate 的 CSV。使用 `--extra-table` 指定即可。

## Step 4：統一特徵

```bash
python multisite_pipeline.py features \
  --project-root /Users/happydoraaa/methane_release_project
```

包含：

- B11/B12；
- SWIR ratio；
- source-centered；
- background ring；
- NDVI；
- temporal target-minus-reference；
- downwind-minus-upwind；
- background-quality heuristic。

查看：

```bash
cat outputs/39_multisite_s2_features_report.txt
```

## 一次執行 Earth Engine 完整版本

```bash
export EE_PROJECT="methane-release-gee"

python multisite_pipeline.py all \
  --project-root /Users/happydoraaa/methane_release_project \
  --use-ee \
  --use-ee-landsat \
  --ee-project "$EE_PROJECT"
```

## 執行後貼回來

```bash
cat outputs/36_multisite_s2_master_report.txt
cat outputs/37_multisite_s2_availability_report.txt
cat outputs/38_cross_sensor_temporal_report.txt
cat outputs/39_multisite_s2_features_report.txt
```

## 科學限制

- MethaneAIR 三個 site 的 negatives 是 no-known-plume reference，不是 confirmed zero-emission。
- MethaneAIR rows 缺少 release interval 並不代表程式失敗。
- Temporal difference 使用同 site 的已知 label-0 reference，屬於 site-calibrated feature，不是 strict zero-shot feature。
- Wind direction 以 meteorological FROM direction 解讀。
- Local GeoTIFF band order 假設為 B2、B3、B4、B8、B11、B12。
