# Machine: mac

| 項目 | 值 |
|---|---|
| 機型 | MacBook Pro（MacBookPro18,3） |
| 晶片 | Apple M1 Pro（10 核：8P + 2E） |
| 記憶體 | 16 GB |
| macOS | 26.6.2（build 25G83） |
| Python | 3.12.2（conda env `navcil`，/Users/aaron/venvs/navcil） |
| PyTorch | 2.11.0（mps available: True） |
| 其他 | numpy 2.4.4、pandas 3.0.2、h5py 3.16.0、pyyaml 6.0.1、transformers 5.5.3、tokenizers 0.22.2、pytest 9.1.1 |
| CPU 執行緒數 | **8**（AMENDMENT-1 起固定：`configs/machine_mac.yaml` 的 `threads`，所有 run 以 `torch.set_num_threads(8)` 執行） |
| 其他 python 套件（NC-2 起） | scikit-learn 1.8.0、scipy 1.18.1 |
| 預設裝置 | cpu（前 50 張測速：cpu 計算中位數 0.0057 s/張，mps 0.0369 s/張） |

PyTorch 版本與 pathselect 在本機使用的 conda base 相同（Python 3.12.2、torch 2.11.0）。
執行時設 `PYTHONNOUSERSITE=1`（scripts/run_stage.sh 已設），避免 ~/.local 的套件混入。
