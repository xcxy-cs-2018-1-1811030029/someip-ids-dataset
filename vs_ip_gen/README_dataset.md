# SOME/IP 入侵检测数据集（公开数据集说明）

> 这是论文"可复现的 SOME/IP 入侵检测数据集 + 基准"的核心交付物。数据集由开源 vSomeIP 在 Linux 多 ECU 测试床上生成，含正常流量 + 5 类攻击，并附完整生成管线可复现、可扩展。

---

## 1. 数据集构成

每个场景生成一个 `*_cli.csv`（客户端观测到的每条 SOME/IP 通知消息一行），列：

```
ts, event, session, payload_len, payload_hex
 ts       时间戳(秒, float)
 event    SOME/IP 事件 ID (如 34680 = 0x8778)
 session  会话 ID
 payload_len  载荷字节数
 payload_hex  载荷字节的十六进制串 (如 "42 43 44 ...")
```

### 场景与标签
| 场景文件 | 标签 | 说明 |
|---|---|---|
| `normal_cli.csv` | **0 (正常)** | 周期事件，信号值在正常范围内 |
| `dos_cli.csv` | 1 | 突发高频通知（资源耗尽） |
| `fuzz_cli.csv` | 1 | 畸形/随机载荷 |
| `drop_cli.csv` | 1 | **消息丢弃**：周期性扣发通知，造成时序缺口 |
| `slowslow_cli.csv` | 1 | 极低速率（低慢速） |
| `tamper_cli.csv` | 1 | **语义保持篡改**：信号值设为越界值（如速度>正常上限），但时序/结构正常 |

`manifest.json` 把每个 CSV 映射到标签（0/1），`scenario.npy` 记录每条样本来自哪个场景，用于分场景分析。

### 信号 Schema（语义合法性用）
载荷 = 12 字节 = 3 个 float32（小端）：
| 偏移 | 类型 | 信号 |
|---|---|---|
| 0 | float32 | speed（速度） |
| 4 | float32 | accel（加速度） |
| 8 | float32 | yaw（横摆角速度） |

正常范围（从正常流量学习）：speed≈[50,150]、accel≈[20,80]、yaw≈[20,119]。

---

## 2. 复现数据集（一次跑全套）

在 WSL（已装 vSomeIP + Boost 1.83）中：
```bash
cd /mnt/d/mlstart/someIP/vs_ip_gen/generate
bash build_samples.sh     # 编译 signal_service / signal_client
bash generate_data.sh     # 生成 normal/dos/fuzz/slowslow/tamper 的 *_cli.csv + manifest.json
```
- `DUR`、`CYCLE` 等参数在 `generate_data.sh` 中可调，用于控制数据量与比例。

## 3. 转成训练数据（可选，供 ML 实验）
```bash
cd /mnt/d/mlstart/someIP
python3 someip_ids/csv_to_records.py --manifest vs_ip_gen/generate/manifest.json --out data/ --len-b 128
# 产出 data/f_a.npy (行为特征)  seq_b.npy (载荷token)  labels.npy  scenario.npy
```

## 4. 基准评测（对比行为 / 语义 / 深度）
```bash
cd /mnt/d/mlstart/someIP
python3 someip_ids/pivot_experiment.py --gen vs_ip_gen/generate/
# 输出 B1(行为) / B2(语义) / FUS(融合) 的总体 + 分场景指标
```

---

## 5. 许可证与发布
- 数据集、生成器、评测代码计划公开（GitHub / Zenodo），附信号 schema 与标签 manifest。
- 数据集可**按需放大**（改 `CYCLE`/`RUNTIME` 或增减场景），方便扩展到更多 ECU/服务/攻击。

## 6. 已知局限（论文中如实写出）
- 单服务/单拓扑的合成数据；真实车辆数据留作后续。
- 语义篡改会改"值"从而也改"字节统计"，行为基线也能察觉；真正"字节统计不变"的攻击需跨信号/上下文一致性建模，属未来工作。
