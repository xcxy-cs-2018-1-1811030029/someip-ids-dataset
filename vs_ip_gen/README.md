# vSomeIP 数据生成方案（在你 WSL Ubuntu-22.04 里执行）

> 用途：为论文"多视图 SOME/IP 入侵检测"生成**大规模、含 hard-case** 的主数据集。
> 步骤：① 跑 `setup_vsomeip.sh` 装环境 → ② 搭 9-ECU 服务 → ③ 生成正常+攻击流量并采 pcap → ④ 转成训练特征。

---

## 0. 目录结构（本目录）
```
vs_ip_gen/
├── setup_vsomeip.sh      # 一键装 vSomeIP + python 依赖（在 Ubuntu-22.04 里跑）
├── README.md             # 本说明
└── generate/            # （待建）服务/ECU 配置 + 攻击注入应用
```

---

## 1. 先装环境
在 **Ubuntu-22.04** 终端执行：
```bash
bash setup_vsomeip.sh
```
装完后 `nvidia-smi` 应能看到 RTX 3050（若看不到也不影响生成，生成是 CPU 密集；GPU 只在训练时用）。

---

## 2. 搭 9-ECU 拓扑（映射文章四的网络）
9 个 ECU：`ADAS, CLU, GPS, IMU, NAV, PT, STE, TEL, VDE`。
- **数据提供者**：GPS、IMU、VDE（从公开驾驶数据读真实信号：速度/距离/偏航角等）
- **消费者**：其余 ECU
- 通信以 **publish-subscribe** 为主（周期/事件通知），并演示 request-response / fire-and-forget / getter-setter。
- 用文章四公开驾驶数据作为信号源：
  🔗 `https://dx.doi.org/10.21227/0q64-5f96`（Autonomous Vehicle Dataset for Anomaly Detection）

> vSomeIP 需要为每个 ECU 写 C++ 应用（service + 客户端），或用其 example apps 扩展。这部分你比较熟 vSomeIP，我负责把**配置模板**和**采数据脚本**给你。

---

## 3. 生成流量（正常 + 4 类攻击 + hard-case）

| 场景 | 做法 | 目标 |
|---|---|---|
| Normal | 正常周期通知 + 订阅 | 基线正常流 |
| DoS | 突发高频事件通知 / 高频 RPC | 资源耗尽 + 时序异常 |
| Fuzzing | 协议合法但 payload 畸形 / 异常 ID 组合 | 载荷异常 |
| Message-drop | 伪造 TTL=0 的 OfferService 撤销订阅 | 缺失/中断 |
| MitM | 劫持服务发现把消费者指向攻击端点 → 中继 | 重定向 + 篡改 |
| **hard-case: low-slow** | 低速率、模仿正常时序/负载分布 | 打行为检测盲区 |
| **hard-case: replay** | 复用正常载荷但改变时间节奏 | 打行为检测盲区 |
| **hard-case: semantic tamper** | 字节统计正常但语义被改 | 打行为检测盲区 |

用 `sudo tcpdump -i eth0 -w <name>.pcap 'udp port 30500 or 30490'`（端口按你服务配置）采集每个场景。

---

## 4. 转成训练特征（接 `someip_ids/` 管线）
1. 用 `scapy` 解析 pcap → 按 `(srcIP,srcPort,dstIP,dstPort)` 分成**流**，得到每条报文的 `ts / payload / length`；
2. `features.py` 的 `compute_view_a_features` 计算**视图A**行为特征（15 维）；
3. `dataset.py` 的 `payload_to_tokens` 生成**视图B**载荷 token 序列；
4. 训练/测试切分（分层 50/50），保存 `f_a.npy / seq_b.npy / labels.npy`。

> 下一步我把 `pcap_to_records.py`（scapy 解析 → 视图A/B numpy 数组）和 `train.py / eval.py / baselines.py` 写好，你就能一键训练。

---

## 5. 建议的验收点（跑通后再写论文实验）
- [ ] `setup_vsomeip.sh` 跑通，`nvidia-smi` 有 GPU in WSL
- [ ] 至少一个 Normal publish-subscribe 场景能采到 pcap
- [ ] 一个 DoS 场景能采到明显不同的 pcap
- [ ] `pcap_to_records.py` 能输出 `f_a/seq_b/labels`
- [ ] 训练 `train_our.py` 得到 >0.9 F1
