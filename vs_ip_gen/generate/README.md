# SOME/IP 数据集生成（应用级日志方案）

> 背景：单机 WSL 里 vSomeIP 用 UDS 内部路由，`tcpdump` 抓不到载荷。因此改用**应用级日志**：
> `signal_service`(发布) 与 `signal_client`(订阅) 把**每条通知的时间戳 + Event + 载荷字节**写进 CSV，
> 再转成视图 A/B 训练数据。这也是后续注入攻击的切入点。

## 组成
- `signal_service.cpp` — 发布服务，支持 `--attack normal|dos|fuzz|slowslow|replay`
- `signal_client.cpp`   — 订阅客户端，记录收到的通知到 CSV
- `build_samples.sh`    — 编译二者（需要 vSomeIP + Boost 1.83）
- `generate_data.sh`    — 批量跑 5 个场景，产出 `<场景>_cli.csv` + `manifest.json`
- `../../someip_ids/csv_to_records.py` — 把 CSV 转成 `f_a.npy / seq_b.npy / labels.npy`

## 用法（在 WSL 里，已装好 vSomeIP）

```bash
cd /mnt/d/mlstart/someIP/vs_ip_gen/generate
bash build_samples.sh            # 生成 signal_service / signal_client
bash generate_data.sh            # 跑 normal/dos/fuzz/slowslow/replay，产出 *_cli.csv + manifest.json

# 转成训练数据
cd /mnt/d/mlstart/someIP
python3 someip_ids/csv_to_records.py --manifest vs_ip_gen/generate/manifest.json --out data/ --len-b 128
```

## manifest.json（自动生成）
把每个场景 CSV 映射到标签：`{"normal_cli.csv":0, "dos_cli.csv":1, ...}`

## 后续训练/评估（待写）
`someip_ids/train_our.py` + `eval_our.py` 会读取 `data/f_a.npy`、`data/seq_b.npy`、`data/labels.npy`，
用 `someip_ids/model.py` 的 `MultiViewFusion` 训练，并输出 F1/PR-AUC/时延等。

## 说明
- 端口/服务 ID 沿用 `vsomeip-local.json`（服务 0x1234, 实例 0x5678, 事件 0x8778, 事件组 0x4465）。
- 若某个场景太短/抓不到数据，调大 `DUR`：`DUR=15 bash generate_data.sh`。
