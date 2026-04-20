# CANN 9.0.x 社区版 — 下载与安装指引

抓取时间: 2026-04-19 (playwright headless chromium, locale zh-CN)
源头: hiascend.com 社区下载门户、CANN 9.0.0-beta.2 文档站、华为云 devcloud PyPI 镜像。

## TL;DR (A3 / Ascend 910B/C, aarch64)

CANN 社区版 **9.0.0-beta.2**（仓库代号 `CANN 9.0.T511`，发布 2026/03/30）是当前最新的 9.0.x 社区版，发布节奏与我们 A3 环境匹配。官方确认 **支持 pip install**，但**仅限 Toolkit + ops 两个包**；`nnal`、`kernels` 类附加库仍需 `.run` 安装器。

目标配置 A3 (Ascend 910B/C) + aarch64 + CPython 3.7~3.13 的最短路径：

```bash
# Step 1 — 配置昇腾 PyPI 源
pip config set global.extra-index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple
pip config set global.trusted-host ascend.devcloud.huaweicloud.com

# Step 2 — 安装 Toolkit + 对应 SoC 的 ops 包
pip install ascend-cann-toolkit==9.0.0b2
pip install ascend-cann-a3-ops==9.0.0b2      # Atlas A3 系列 (A3 host)
# 910B/910C 独立用时改为:  ascend-cann-910b-ops==9.0.0b2    （Atlas A2 系列）

# Step 3 — 激活环境变量 (以非 root 安装到 $HOME 为例)
source $HOME/lib/python3.12/site-packages/ascend/cann-9.0.0-beta.2/set_env.sh
```

Pip 流程经本地验证（x86_64 whl 已从 ascend.devcloud PyPI 拉取成功，`ascend_cann_toolkit-9.0.0b2-py3-none-manylinux2014_x86_64.whl` 约 202 MB，SHA256 `b6df932a...`）。

## 版本列表（昇腾社区下载门户 `ascendgateway/ascendservice/resourceDownload/mappingList`）

| Version | 发布时间 | 文档入口 | 备注 |
|---|---|---|---|
| 9.0.0-beta.2 | 2026/03/30 | `CANNCommunityEdition/900beta2` | 最新 9.0.x，仓库 `CANN 9.0.T511` |
| 9.0.0-beta.1 | 2026/03/02 | `CANNCommunityEdition/900beta1` | 前一版 beta |
| 8.5.2 | 2026/04/13 | `CANNCommunityEdition/850` | 最新 8.5 LTS 线 |
| 8.5.0 (推荐) | 2026/01/16 | `CANNCommunityEdition/850` | 官方标「推荐」，与 verl-A3 镜像主流 CANN 贴近 |
| 8.5.1 / 8.5.0.alpha00* | 2025/Q4 | … | — |
| 8.3.RC2 / 8.3.RC1 | 2025/Q4 | … | — |

## 9.0.0-beta.2 架构 / SoC 覆盖

从 `cann/info/zh/0?versionName=9.0.0-beta.2` 返回数据（47 个包）梳理：

- cpus: **AArch64、X86_64**（两个都支持）
- packageTypeList: **rpm、run、tar.gz、zip**（**没有 deb**，对比 8.5.2 就有）
- SoC 维度，ops 算子包按产品系列划分：
  - `Ascend-cann-A3-ops` — Atlas A3 系列 (A3 host 对应)
  - `Ascend-cann-910b-ops` — Atlas A2 系列 (Ascend 910B/C)
  - `Ascend-cann-910-ops` — Atlas 训练系列 (老 910)
  - `Ascend-cann-310p-ops` / `Ascend-cann-310b-ops` — 推理卡
  - **`Ascend-cann-950-ops`**（只在安装文档示例里出现，门户列表没有；HEAD 得到 2.3 GB，推测对应 Ascend 950PR 的 ops——和 a5_ops 描述对得上）
- 其余公共包: `Ascend-cann-toolkit`（984 MB）、`Ascend-cann-nnal`（566 MB，含 ATB、SiP 加速库）、`Ascend-cann-amct`（模型压缩工具）、`Ascend-cann-device-sdk`（310/310P 驱动侧）

## 安装方法 A — pip（推荐，官方渠道）

**源索引:** `https://ascend.devcloud.huaweicloud.com/cann/pypi/simple`

官方 pip 镜像中 **实际** 发布的包（`curl simple/<pkg>/` 核对）:

| pip 包名 | 9.0.0b2 AArch64 | 9.0.0b2 X86_64 | 对应 SoC |
|---|---|---|---|
| `ascend-cann-toolkit` | yes (SHA256 `2a75656d…`) | yes (SHA256 `b6df932a…`, 202 MB) | 通用 |
| `ascend-cann-a3-ops` | yes (SHA256 `87e1ea85…`) | yes | Atlas A3 |
| `ascend-cann-910b-ops` | yes (SHA256 `c1216930…`) | yes | Atlas A2 (910B/C) |
| `ascend-cann-910-ops` | yes | yes | 老 910 训练 |
| `ascend-cann-310p-ops` | ? | ? | 推理 |
| `ascend-cann-310b-ops` | ? | ? | 200I/500 A2 推理 |
| `ascend-cann-nnal` | **not published** (404) | 404 | 需走 .run |
| `ascend-cann-950-ops` | **not published** (404) | 404 | **需走 .run** — Ascend 950PR |
| `ascend-cann-350-ops` | 404 | 404 | Atlas 350 加速卡 |

**→ 对 a5_ops 重编的启示**: a5_ops 目标 SOC = Ascend950PR，需要的是 `Ascend-cann-950-ops_9.0.0-beta.2_linux-aarch64.run` (2.33 GB, CPUAArch64, Content-Disposition 已验证 200)。**该包在 pip 索引里 404**，要从 `ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T511/` 直接下 `.run`。

### 执行步骤（对 A3 host 预演，**不要**现在真装）

```bash
# 先检查 python 环境
python3 --version                      # 需要 3.7 ~ 3.13
pip --version

# 配置源
pip config set global.extra-index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple
pip config set global.trusted-host ascend.devcloud.huaweicloud.com

# Toolkit
pip install ascend-cann-toolkit==9.0.0b2
# → 落到 <python>/site-packages/ascend/cann-9.0.0-beta.2/

# ops – A3 host 选 a3-ops; 如果是 910B/C 纯 A2 机则换 910b-ops
pip install ascend-cann-a3-ops==9.0.0b2
# 重要: 同一路径不能装多个 SoC 的 ops；多芯片混合用需要多 venv / 多 prefix

# 激活
source $(python3 -c "import site; print(site.getsitepackages()[0])")/ascend/cann-9.0.0-beta.2/set_env.sh
```

### Pip 安装的局限

文档原文（`instg_0104`）：
> Pip 方式安装的 CANN Toolkit 开发套件包，**主要用于框架侧运行训练和推理业务**。若需要开发调试场景下（算子/应用/模型的开发和编译等）的 Toolkit 包，**请选择其他安装方式**。

换言之：**pip 的 toolkit 是 runtime 子集**，缺算子开发链 / 编译工具。a5_ops 重编这种任务 **必须** 走 `.run`。

## 安装方法 B — .run 安装器（适合我们 a5_ops 重编场景）

9.0.0-beta.2 的 aarch64 `.run` 尺寸（均通过 OBS HEAD 200 核实）:

| 包 | 尺寸 | URL 模板 |
|---|---|---|
| toolkit | 984 MB | `.../CANN%209.0.T511/Ascend-cann-toolkit_9.0.0-beta.2_linux-aarch64.run` |
| A3-ops (A3 host) | 2.25 GB | `.../Ascend-cann-A3-ops_9.0.0-beta.2_linux-aarch64.run` |
| 910b-ops (A2 / 910B/C) | 2.28 GB | `.../Ascend-cann-910b-ops_9.0.0-beta.2_linux-aarch64.run` |
| **950-ops (Ascend950PR / a5_ops)** | **2.33 GB** | `.../Ascend-cann-950-ops_9.0.0-beta.2_linux-aarch64.run` |
| nnal (ATB/SiP) | 566 MB | `.../Ascend-cann-nnal_9.0.0-beta.2_linux-aarch64.run` |

URL 前缀: `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T511/`
注意仓库路径含空格，按 `%20` 或 URL 引号转义。

安装流程（文档 `instg_0008`，Ubuntu physical-machine 场景）：
```bash
# 1) 基础依赖
sudo apt-get install -y python3 python3-pip

# 2) 按顺序装 toolkit -> ops -> nnal
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T511/Ascend-cann-toolkit_9.0.0-beta.2_linux-aarch64.run"
bash ./Ascend-cann-toolkit_9.0.0-beta.2_linux-aarch64.run --install
# 默认路径: root -> /usr/local/Ascend; 非 root -> $HOME/Ascend

wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T511/Ascend-cann-950-ops_9.0.0-beta.2_linux-aarch64.run"
bash ./Ascend-cann-950-ops_9.0.0-beta.2_linux-aarch64.run --install
# 多 SoC 共存要装到不同 --install-path=<path>

# (可选) nnal
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T511/Ascend-cann-nnal_9.0.0-beta.2_linux-aarch64.run"
bash ./Ascend-cann-nnal_9.0.0-beta.2_linux-aarch64.run --install

# 3) 环境变量
source $HOME/Ascend/cann/set_env.sh             # toolkit
source $HOME/Ascend/nnal/atb/set_env.sh         # nnal ATB (可选)
```

不需要账号或点击协议就能 wget（OBS 桶直连）。

## A3 / Ascend 910B/C 推荐版本决策

- **用于 a5_ops 重编** (目标 SoC Ascend950PR): **9.0.0-beta.2 toolkit + 950-ops (.run)** — 这是唯一带 950-ops 的社区版，别的版本没 950 产品线。
- **如果只跑 A3 host 上的常规 RL 训练/推理**: **9.0.0-beta.2 toolkit + a3-ops (pip 即可)**。
- **若想尽量贴近现有 verl-A3 镜像**: 镜像里 **容器 CANN 是 8.5.1**（见 `repo/knowledge/images/...`），要重现 GPU-对照实验优先用 8.5.x；9.0.0-beta.2 是「前瞻安装」用来编 a5_ops，**不要全替换**。

## 与现有环境共存的风险

- **host CANN 8.3.RC2 + 容器 CANN 8.5.1** 的现状：装 9.0.0-beta.2 必须走独立目录 (`--install-path=` 或 venv + pip)，别覆盖 host 的 `/usr/local/Ascend`，否则可能破坏驱动侧套件。建议方案：
  - pip 路径：**新建 venv**，Toolkit 自动落到该 venv 的 `site-packages/ascend/cann-9.0.0-beta.2/`，与 host 完全隔离。
  - .run 路径：`--install-path=$HOME/Ascend-9.0.0b2` 放到用户目录；source 对应 `set_env.sh` 临时激活。
- **a5_ops (CANN 9.0.0, SOC Ascend950PR) 兼容性**：9.0.0-beta.2 的 `Ascend-cann-950-ops` 对应产品「Atlas 350 加速卡 / Atlas A3 系列」范畴里的 950 chip 家族，属同一 9.0.x 大版本，编译兼容概率很高。**关键未知**：需要在 A3 host 真机上 `./950-ops.run --extract` 翻 so/头文件验证 SoC 列表。这是下一步实验才能验。
- **ops 不能同一路径多装**：文档明确 "多个芯片的 ops 算子包暂不支持安装在同一路径下"。如果既要 950-ops（a5_ops 重编）又要 a3-ops（A3 runtime），需要两个 --install-path。

## 注意事项

- Pip 源是 **华为云 devcloud 镜像** (`ascend.devcloud.huaweicloud.com`)，海外访问要确认 GFW 不拦；本地 (x86 host) 实测可拉通。A3 host 因在 GFW 后，git/pypi 走自有路径，需另行确认（memory: a3_is_firewalled）。
- pip 元数据 `Requires-Python: >=3.7`；`site-packages/ascend/cann-9.0.0-beta.2/` 下直接含 `atc`、`atc.bin`、`bin/*_executor`、`compiler/python/`、`compat/cann-*-compat.tar.gz` — runtime + 部分编译链齐备。
- 没有点击式许可协议拦截（下载门户 `recommend` 项标灰，9.0.0-beta.2 `warning: false`），直链可下。
- 与 torch_npu 搭配时 torch_npu 会读 `ASCEND_HOME_PATH`，pip-install 路径下是 `<site-packages>/ascend/cann-9.0.0-beta.2`，**source set_env.sh 一定要做**，否则 torch_npu import 报路径缺失。

## 抓屏 / 原始资料

- `/tmp/cann-9-screens/community_download.png` — 社区下载门户初屏 (version 下拉未展开)
- `/tmp/cann-9-screens/pip_install_canonical.png` — `instg_0104.html` pip 安装指南 A3 默认视图
- `/tmp/cann-9-screens/pip_install_canonical.txt` — 上页的 inner_text
- `/tmp/cann-9-screens/info_9_0_0_beta_2.json` — 门户 info API 完整返回（47 个包元数据）
- `/tmp/cann-9-screens/ops_per_product.json` — 五类 Atlas 产品的 pip 包名映射

## Raw sources

### 主要 URL

- 社区下载门户 SPA: `https://www.hiascend.com/developer/download/community/result?module=cann`
- 版本目录 API: `https://www.hiascend.com/ascendgateway/ascendservice/resourceDownload/mappingList?category=0&offeringList=CANN`
- 单版本详情 API: `https://www.hiascend.com/ascendgateway/ascendservice/cann/info/zh/0?versionName=9.0.0-beta.2`（需要带 Referer）
- 9.0.0-beta.2 文档首页: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/index/index.html`
- **Pip 安装权威页**: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/softwareinst/instg/instg_0104.html?Mode=PmIns&OS=Ubuntu&InstallType=netpip`
- OBS 分发桶前缀: `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T511/`
- Pip 索引: `https://ascend.devcloud.huaweicloud.com/cann/pypi/simple`

### instg_0104 关键段落 (节选)

> 配置昇腾源
> 执行如下命令配置昇腾源，否则将无法安装。
>
> `pip config set global.extra-index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple`
> `pip config set global.trusted-host ascend.devcloud.huaweicloud.com`
>
> 安装Toolkit开发套件包
> …Pip方式安装的 CANN Toolkit 开发套件包，主要用于框架侧运行训练和推理业务。若需要开发调试场景下（算子/应用/模型的开发和编译等）的 Toolkit 包，请选择其他安装方式。
>
> `pip install ascend-cann-toolkit==9.0.0b2`
>
> 软件包会安装在「<Python所在路径>/site-packages/ascend」目录下…
>
> 安装ops算子包… 多个芯片的 ops 算子包暂不支持安装在同一路径下…
> Atlas A3 系列产品 → `pip install ascend-cann-a3-ops==9.0.0b2`
> Atlas A2 系列产品 → `pip install ascend-cann-910b-ops==9.0.0b2`
