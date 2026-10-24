# Strata NVFP4 fork 0.2.0-beta-nvfp4.3 升级 A/B（2026-10-20，Roy GPU0）

**一句话结论**：升级已上线——引擎 `0.1.44.1` → `0.1.45.3`（fork `v0.2.0-beta-nvfp4.3`），**decode 稳定快 ~10%（3 次重复中位 +9.4%，全 12 档单次 +0.5%～+15.7%）**，prefill 持平，专家命中不变，输出语义一致。**生产参数除新增 `--batch-mtp` 外一字未改**，采样块保留（升级不会偷换采样）。

---

## 一、升级对象与构建（Roy，源码自建）

| 项 | 值 |
|---|---|
| 分支 tag | `sergqwer/strata-nvfp4` `v0.2.0-beta-nvfp4.3`（10-20 01:32 发布） |
| 引擎版本 | `strata 0.1.45.3`（`project(strata VERSION 0.1.45.3)`） |
| 源码来源 | codeload tar.gz（Roy 直连 GitHub 只有 ~10 KB/s，改从本机中转抓取 28 MB / 2.8 s，`gzip -t` 通过） |
| 快照 | `/home/hermes/AI/strata-0.2.0-src`，`git init` 快照 commit `3409e0e`（4460 文件） |
| 编译 | CUDA 13.0 / `CMAKE_CUDA_ARCHITECTURES=120` / Release，`BUILD_EXIT=0`，产物 `build/strata` 95 MB |
| ggml 依赖 | 复用本机既有 llama.cpp 检出 `3cf0325`（与本版 pin 的 `3cf03257…` 一致），避免重新拉取 |
| vision 二进制 | 单建 `STRATA_PORTABLE=ON / STRATA_VISION_CUDA=OFF`（与旧部署一致），`VISION_EXIT=0` |
| 服务目录 | `Strata-nvfp4-gptq-0.2.0/`（engine + serve + tools + data 全量复制，与 0.1.44 目录并存，回滚零成本） |

参数面审计：`--help` 开关 **新 146 / 旧 144，旧参数零删除**；新增 `--conversation-cache-park-gain`、`--prefill-pipe`、`--batch-mtp`。原 48 个启动参数可直接沿用（实测新引擎原样起服务）。

## 二、方法学修正（这次最重要的教训）

1. **旧基线不能是热态**：第一份基线（12:5x）取自已运行 4 小时的服务——专家已常驻、页缓存已热，**1K/4K prefill 被系统性抬高**。按"同一服务启动后第一遍"的既定口径，它不合格。补测**冷启动基线**（`before-cold-0.1.44.1.json`）作为唯一对照。
2. **单次测量的噪声大到能反转结论**：旧引擎自己 1K rep prefill 三次重复跨度 37%，4K 3.5%。第一轮"1K prefill −25.8%"的判负来自**首次请求的冷启动开销**，不是引擎退化（三次重复后 newauto 1K rep prefill 反为 +1.3%）。
3. **重复测量必须换 payload**：同 tag 二次发送命中引擎前缀缓存，prefill/TTFT 虚低。为此给 `strata_prefill_decode_bench.py` 加 `--tag-suffix`，每 rep 唯一 tag。
4. **裁定门槛改用"参考臂自身可重复性"**：`prefill 跌幅 > max(3%, 参考臂该格 min–max 跨度)` 才算退化，避免用固定 3% 切在噪声边上。

## 三、实验臂与结果

全部为**冷启动 + 每臂 3 次唯一 payload 重复**，只测 1K/4K/32K 两尾（短中长三档最能暴露回退），另有全 12 档单次跑作长上下文旁证。

### 3.1 三档 × 六格（中位数，相对旧引擎 0.1.44.1）

| 档位 | 尾 | newauto prefill/decode | +batch-mtp | parallel=1 | --prefill 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|
| 1K | nat | +2.4% / +9.8% | −0.1% / **+10.2%** | +7.8% / +15.0% | −24.4% / +16.0% | +2.1% / +12.7% | +2.0% / +12.7% |
| 1K | rep | +1.3% / +8.1% | +0.2% / +7.8% | +6.3% / +10.8% | −20.3% / +7.9% | −1.7% / +5.4% | −1.7% / +7.6% |
| 4K | nat | −1.1% / +15.8% | −1.4% / **+19.2%** | −0.6% / +17.1% | **−43.0%** / +10.3% | −25.4% / +12.4% | −17.8% / +16.6% |
| 4K | rep | −3.1% / +8.3% | 0.0% / +8.1% | −0.7% / +8.1% | **−43.4%** / +8.1% | −25.5% / +7.8% | −17.7% / +8.1% |
| 32K | nat | −0.1% / +7.7% | −0.5% / +8.8% | −0.2% / +6.4% | **−54.2%** / +4.2% | −31.5% / +5.2% | −15.9% / +1.3% |
| 32K | rep | −0.9% / +10.0% | 0.0% / +9.9% | −0.8% / +9.9% | **−55.0%** / +7.9% | −32.1% / +9.5% | −16.8% / +9.6% |

**两条硬结论**：
- **`--prefill` 必须留 `auto`**：硬指 1024/2048/4096 会让 prefill 掉 16%～55%（上游把默认提示块调大，auto 才是它自己选的合适值）。此项写入永久禁止项。
- **decode 增益对所有臂都在 +5%～+19%**，`--batch-mtp` 中位最优（+9.4% vs +9.0%），且 prefill 更稳（最差仅 −1.4%）→ 选它做生产参数。

### 3.2 全 12 档（生产臂 0.2.0+batch-mtp 单次冷跑 vs 旧引擎冷基线）

| 档位(实际pt) | nat decode | rep decode | prefill |
|---|---|---|---|
| 1K (1242) | +10.9% | +12.6% | +4.2% / −2.6% |
| 4K (4922) | +15.7% | +8.3% | +0.4% / +2.5% |
| 32K (41K) | +13.8% | +9.2% | +0.8% / +0.4% |
| 64K (82K) | +14.5% | +8.7% | +0.1% / −0.1% |
| 128K (172K) | +9.4% | +6.1% | +1.1% / +0.9% |
| 245K (325K) | +11.9% | +0.5% | −0.8% / −0.5% |

命中率 98.0–99.8%，PCIe/磁盘搬 0，与升级前同级。

### 3.3 输出一致性探针（温度 0）

| 探针 | 结果 |
|---|---|
| 长文召回（8K 结构文本中埋两个校验码） | **逐字相同**：`PLUM-OTTER-4417 SLATE-HERON-9052` |
| 代码（merge_intervals） | **逐字相同**（310 字符） |
| 短中文（天空为什么蓝） | 措辞不同（相似度 0.89），语义一致 |
| 短英文（KV cache） | 措辞不同（相似度 0.09），语义一致 |

→ 上游"同 bit"的说法**只在本机部分成立**：长文与代码逐字一致，短回答有改写。多槽批处理本身就会让贪心输出不确定，这一点必须写进结论，不能宣称完全等价。

## 四、上线状态（已生效）

- systemd drop-in `30-nvfp4-020.conf` 指向 `0.2.0/prod-batchmtp.json`，端口仍 `18083`；旧 unit 原文件未改，**删除 drop-in 即可回滚**。
- 现场：`engine 0.1.45.3`、`batch_slots=2`、`vram_free 2966 MiB`、`/v1/models` 正常、**NewAPI(3050) 端到端调用成功**（`strata-nvfp4-orcarouter-uncensored`，usage 正常入账）。
- 参数未动：`--expert-cache auto`、`--prefill auto`、512K + int8 KV + kv-resident 32768、对话缓存 16384/8、yarn×2、control-vector、`--vram-reserve-mib 3072`、sampling 块（1.0/0.95/20）。

**回滚（一条命令级）**：
```bash
sudo rm -f /etc/systemd/system/strata-nvfp4.service.d/30-nvfp4-020.conf
sudo systemctl daemon-reload && sudo systemctl restart strata-nvfp4
```

## 五、过程事故（如实记录）

| # | 事故 | 后果 | 处理 |
|---|---|---|---|
| 1 | 窗口一脚本里 `pkill -f "$NEW/engine/strata"` 匹配到脚本自身命令行 | 脚本自杀，**本机模型中断约 11 分钟**（14:34–14:46） | 改掉该匹配（用进程组 + 精确 PID），窗口二自动恢复生产 |
| 2 | 窗口三把 `--prefill` 值写成 int | server 拼 argv 报 `TypeError`，三个臂空跑 21 分钟 | 改字符串并断言全 str，窗口三 b 重跑 |
| 3 | exec 里目标服务以 detached 方式起，`kill $!` 只杀包装脚本 | 三个臂串行时第二个撞 `port in use` | 改为前台 nohup + 记 PID + 等端口/显存释放 |
| 4 | 验证臂日志路径名与配置文件名不一致 | 生产确认 bench 空跑（exit=1） | 从配置 JSON 读 `log` 字段，不靠猜名字 |

## 六、未做 / 限制

- 全 12 档只有**单次**测量（3 次重复只做了 1K/4K/32K），长档位给的是量级而非可信区间。
- `parallel=1` 只做了短三档：单发 decode 比 2 槽再快 5–7%（印证官方"槽位从专家缓存扣"的说法），但**并发语义改变，未上线**；若将来只跑单用户可再议。
- 未做长文写作质量 A/B（只做了召回与代码正确性），中文长文风格需人工目测。
- 未测 512K 满窗、未测图像输入在 0.2.0 下的实际质量。
- 未做 0.2.0 浏览器 UI（`127.0.0.1:8791`）——那是 fork 的 Windows 侧特性，本机 Linux 部署不使用。

## 七、证据文件（Roy `/home/hermes/AI/bench-20261020/`）

`before-0.1.44.1-nvfp4.2.json`（热基线，仅存档）、`before-cold-0.1.44.1.json`（冷基线）、`after-A-0.2.0-nvfp4.3.json`、`arm-{old,newauto,batchmtp,par1,newp1024,newp2048,newp4096}-S{1,2,3}.json`、`after-prod-full.json`、`probe-{before,before-cold,after-A,prod}.json`、`agg-window3.json`、`agg-window4.json`、`agg-full.json`、`choice.json`、`window{,2,3,3b,4}.log`、`prodbench.log`。
