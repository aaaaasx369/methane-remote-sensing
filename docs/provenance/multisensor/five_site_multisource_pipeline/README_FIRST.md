# 真正完成五個 site＋不同來源

你前一次的結果只有：

- Casa Grande；
- Ehrenberg；
- 所有資料來源都被寫成 `unknown_source`。

所以它只能回答兩個 site 的 baseline。

這個新版會建立：

1. Casa Grande → `2024_AMT`
2. Ehrenberg → `2023_Scientific_Reports`
3. MethaneAIR site 1 → `MethaneAIR`
4. MethaneAIR site 2 → `MethaneAIR`
5. MethaneAIR site 3 → `MethaneAIR`

最後來源數是 3，不再是 `unknown_source`。

## 重要科學限制

前兩個 Arizona sites 有 controlled-release positive／negative ground truth。

後三個 MethaneAIR sites 使用：

- positive：MethaneAIR observational plume；
- negative：同一地點、相近季節、排除已知 plume 日期的 no-known-plume reference。

所以五-site結果應稱為：

> Exploratory five-site multisource generalization experiment

不能稱為完整的 five-site controlled-release benchmark。

---

# 1. 下載 ZIP 後安裝與準備

```bash
cd ~/Downloads

ZIP_FILE="$(find "$HOME/Downloads" -maxdepth 1 -type f \
  -iname 'five_site_multisource_pipeline*.zip' -print -quit)"

test -n "$ZIP_FILE" || {
  echo "找不到 ZIP"
  exit 1
}

rm -rf "$HOME/Downloads/five_site_multisource_pipeline"

unzip -o "$ZIP_FILE" \
  -d "$HOME/Downloads/five_site_multisource_pipeline"

cd "$HOME/Downloads/five_site_multisource_pipeline"

bash install_and_prepare.sh
```

安裝程式會：

- 複製三支程式進你的專案；
- 檢查套件；
- 跑模型 self-test；
- 從 110 張 MethaneAIR positive patches 的座標自動挑出三個有至少兩張影像的獨立 site；
- 修正 Casa Grande 與 Ehrenberg 的來源名稱。

查看：

```bash
cat /Users/happydoraaa/methane_release_project/outputs/543_five_site_prepare_report_v1.txt
```

這時應看到總共 5 個 site：2 個 controlled-release＋3 個 MethaneAIR。

---

# 2. 搜尋三個 MethaneAIR site 的 reference negatives

```bash
cd /Users/happydoraaa/methane_release_project
source .venv/bin/activate

earthengine authenticate
export EE_PROJECT="你的-Google-Cloud-project-ID"
```

先只搜尋：

```bash
python download_methaneair_reference_negatives.py \
  --project-root /Users/happydoraaa/methane_release_project \
  --search-only
```

查看候選：

```bash
cat outputs/546_methaneair_reference_negative_report_v1.txt
```

正式下載，每個新 site 預設 8 張：

```bash
python download_methaneair_reference_negatives.py \
  --project-root /Users/happydoraaa/methane_release_project
```

若候選太少，可以放寬 clear fraction：

```bash
python download_methaneair_reference_negatives.py \
  --project-root /Users/happydoraaa/methane_release_project \
  --minimum-clear-fraction 0.65 \
  --season-window-days 80
```

輸出：

```text
outputs/544_methaneair_reference_negative_candidates_v1.csv
outputs/545_methaneair_reference_negative_selected_v1.csv
outputs/546_methaneair_reference_negative_report_v1.txt
outputs/547_methaneair_reference_negative_manifest_v1.csv
patches/s2_reference_negatives_v1/
```

---

# 3. 合併成真正的五-site manifest

```bash
python prepare_and_finalize_five_sites.py \
  --project-root /Users/happydoraaa/methane_release_project \
  --finalize
```

成功條件：

```text
Unique sites: 5
Unique sources: 3
Sites with both classes: 5
```

查看：

```bash
cat outputs/550_five_site_finalize_report_v1.txt
```

最終輸入表：

```text
outputs/548_five_site_multisource_manifest_v1.csv
```

---

# 4. 訓練五-site／不同來源模型

```bash
python run_multisource_s2_model_v2.py \
  --project-root /Users/happydoraaa/methane_release_project \
  --input outputs/548_five_site_multisource_manifest_v1.csv
```

查看：

```bash
cat outputs/506_multisource_model_report_v2.txt
```

教授的問題主要看：

```text
evaluation = leave_one_site_out
model = logistic
```

它回答：

> 用另外四個 site 訓練，能不能預測完全沒看過的第五個 site？

再看：

```text
evaluation = leave_one_source_out
model = logistic
```

它回答：

> 用其他 provenance groups 訓練，能不能轉移到未見過的資料來源？

但因為來源與地點部分綁在一起，這應解讀成：

> combined source-and-domain shift

而不是純粹的 source effect。

---

# 完成後貼回這三份

```bash
cat outputs/543_five_site_prepare_report_v1.txt
cat outputs/550_five_site_finalize_report_v1.txt
cat outputs/506_multisource_model_report_v2.txt
```
