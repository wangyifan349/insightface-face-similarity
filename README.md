# 🎯 InsightFace 人脸相似度 / 人脸搜索（buffalo_l · ArcFace）

<p align="center">
  <a href="README.md"><strong>🇨🇳 中文</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="README_EN.md"><strong>🇬🇧 English</strong></a>
</p>

基于 [InsightFace](https://github.com/deepinsight/insightface) 官方 `buffalo_l` 模型包的一层薄封装：
把图片里的人脸检测出来、提取成 512 维特征向量，然后算出**余弦相似度百分比**。

- 🆔 **1:1** 两张图比一次
- 🔎 **1:N** 一张查询图对一批候选（上传的多张图，或一个文件夹人脸库）排序
- 🧩 三种用法：**网页界面**（Flask + AJAX）、**命令行 / 双击**、**当 Python 库 import**
- ⚡ **纯 CPU 可跑**，自动使用 CUDA（有 N 卡且装了 `onnxruntime-gpu` 时）
- 🔒 所有照片都在**本地**处理，不上传任何第三方服务器

> ⚠️ **本工具只输出相似度，不判断"是不是同一个人"。**
> 它给你一个 0 ~ 100 的数值，要不要用它下判断、用什么阈值，由你自己决定。
> 请用**你自己**的样本（同一人 / 不同人各若干）跑一遍，看分数分布再定阈值。

---

## 📖目录

- [✨ 特点](#特点)
- [📦 仓库内容](#仓库内容)
- [🧠 模型：到底用了什么](#模型到底用了什么)
- [🔢 算法：从图片到一个百分比](#算法从图片到一个百分比)
- [❓ 为什么分数是 45~70 而不是 99](#为什么分数是-4570-而不是-99)
- [💡 一个容易踩的坑：脸在图里越小，分数越低](#一个容易踩的坑脸在图里越小分数越低)
- [🔍 取脸规则](#取脸规则)
- [🚀 快速开始](#快速开始)
- [🖥 用法一：网页版](#用法一网页版)
- [⌨ 用法二：命令行](#用法二命令行)
- [📚 用法三：当库用](#用法三当库用)
- [🔌 HTTP 接口](#http-接口)
- [🔧 环境变量](#环境变量)
- [📊 性能实测](#性能实测)
- [🐳 部署](#部署)
- [🆘 常见问题](#常见问题)
- [🙏 致谢与引用](#致谢与引用)
- [📜 许可](#许可)

---

## ✨特点

| 项目 | 说明 |
|---|---|
| 内核 | InsightFace `buffalo_l`（SCRFD-10GF 检测 + ArcFace ResNet50@WebFace600K 识别） |
| 推理 | onnxruntime（CPU / CUDA），不需要 PyTorch、不需要编译 |
| 特征 | 512 维 ArcFace 向量，已做 L2 归一化 |
| 打分 | 两个向量的余弦相似度 × 100 |
| 依赖 | Python 3.10+、insightface、onnxruntime、opencv-python、numpy（网页版另需 Flask） |
| 模型 | 首次运行自动下载官方 `buffalo_l`（约 326 MB）；只加载其中 2 个文件（182 MB） |
| 联网 | 只有第一次下载模型需要，之后可完全离线 |

---

## 📦仓库内容

```
insightface-face-similarity/
├── insightface_face_similarity.py  核心库：检测 + 特征 + 相似度（唯一真正干活的文件）
├── example_insightface.py          最小示例，6 行，只演示 1:1
├── example2_insightface.py         命令行 + 双击交互，1:1 和 1:N
├── flask_insightface_face.py       网页版，两个面板，都是 1:N
├── 使用说明_insightface.txt         完整中文 API 文档（函数、参数、返回结构全列）
├── README.md                       本文件（中文）
└── README_EN.md                    英文版，内容相同
```

四个文件彼此独立，唯一的共同依赖是 `insightface_face_similarity.py`，
所以它必须和另外三个放在同一目录（或在 `PYTHONPATH` 里）。

```
insightface_face_similarity.py
        ^
        | 被依赖
        |
  +-----+-----------+----------------+
  |                 |                |
example_insightface.py  example2_insightface.py  flask_insightface_face.py
   最小示例            命令行 / 双击          网页
```

---

## 🧠模型：到底用了什么

固定使用 InsightFace 官方的 `buffalo_l` 模型包。这个包里一共有 5 个 `.onnx` 文件，
**本工具只加载其中 2 个**：

| 文件 | 作用 | 大小 | 是否加载 |
|---|---|---|---|
| `det_10g.onnx` | SCRFD-10GF 人脸检测，输出检测框 + 5 点关键点 | 16.1 MB | ✅ |
| `w600k_r50.onnx` | ArcFace（ResNet50@WebFace600K）人脸识别，输出 512 维向量 | 166.3 MB | ✅ |
| `1k3d68.onnx` | 3D 关键点模型 | 137 MB | ❌ 跳过 |
| `2d106det.onnx` | 106 点 2D 关键点模型 | 4.8 MB | ❌ 跳过 |
| `genderage.onnx` | 性别 / 年龄模型 | 1.3 MB | ❌ 跳过 |

跳过后三个是**故意的**：算相似度根本用不到关键点和年龄。
本机实测（同机器、纯 CPU、连续测两次取稳定值）：

| | 加载耗时 |
|---|---|
| 加载全部 5 个模型 | 1.52 ~ 1.69 s |
| **只加载 2 个模型（本工具）** | **1.12 ~ 1.17 s** |

所以磁盘上模型包约 326 MB，实际进内存的权重只有 182 MB（少加载了 143 MB 的 3D 关键点权重）。
本工具在代码层面就是通过 `allowed_modules=["detection", "recognition"]` 实现的，
见 `insightface_face_similarity.py:196`。

### 📊 官方给出的 buffalo_l 识别精度

下表来自 [InsightFace 官方 Model Zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md)，
是模型在标准数据集上的评测结果（不是本工具实测）：

| 指标 | MR-ALL | African | Caucasian | South Asian | East Asian | LFW | CFP-FP | AgeDB-30 | IJB-C(E4) |
|---|---|---|---|---|---|---|---|---|---|
| buffalo_l | 91.25 | 90.29 | 94.70 | 93.16 | 74.96 | 99.83 | 99.33 | 98.23 | 97.25 |

值得留意的是 East Asian 一列明显偏低（74.96），说明模型在不同人群上的表现并不均衡。

### 🆚 和其他人脸库比，谁更准

先说结论，再给数据。

**结论**：同为「检测 + 特征 + 余弦」这一类方案，ArcFace 系列处在第一梯队；
dlib、OpenCV SFace 这类轻量方案差几个百分点但快得多；
FaceNet512 在同一张榜单上最高，ArcFace 差约 1.8 个点。
**本项目用的是 ArcFace，不是 FaceNet**。

#### ✅ 唯一可比的一张表

下面这张表来自 [DeepFace 官方 benchmark](https://github.com/serengil/deepface/tree/master/benchmarks)：
**同一个 LFW 数据集、同一套验证对、同一个 cosine 距离、都开启对齐、检测器统一用 RetinaFace**。
协议一致，所以表内各模型之间的横向比较是成立的。

| 识别模型 | LFW 准确率 | 特征维度 | 备注 |
|---|---|---|---|
| FaceNet512 | **98.4** | 512 | 本表最高 |
| ArcFace | **96.6** | 512 | **本项目使用** |
| FaceNet | 96.4 | 128 | 本表维度最小的强模型 |
| VGG-Face | 95.8 | 4096 | 特征大、推理慢 |
| SFace | 92.4 | 128 | MobileFaceNet 级别，极轻量 |
| GhostFaceNet | 90.5 | 128 | 追求极致轻量 |
| Dlib | 89.1 | 128 | 见下方「差距有多大」 |
| OpenFace | 69.4 | 128 | 已被上面几代取代 |
| DeepFace | 67.7 | 4096 | 较老的模型 |
| DeepID | 67.7 | 160 | 更早的模型 |
| *人类* | *97.5* | — | 同一批数据上的对照基线 |

#### 📏 差距有多大

在**完全相同的协议**下：

- **ArcFace 96.6% vs Dlib 89.1% → 差 7.5 个百分点**。
  换算成错误率：3.4% vs 10.9%，dlib 的错误率约为 ArcFace 的 **3.2 倍**。
- ArcFace vs 人类基线（97.5%）还差 0.9 个点。
- **检测器的影响不比模型小**：同样是 ArcFace，
  检测器从 RetinaFace 换成 OpenCV Haar 就掉到 84.6%，换成「不检测、直接用原图」只有 54.8%。
  所以别只盯着识别模型，`det_10g.onnx`（SCRFD）同样关键。
- 代价：ArcFace(R50) 权重 166 MB、512 维、CPU 明显慢于 dlib(128 维小网络)。
  要速度可以换 SFace/GhostFaceNet，接受几个百分点的损失。

#### 📋 各项目自报的数字（协议不同，不能直接比）

| 项目 | 自报 LFW | 出处 |
|---|---|---|
| InsightFace `buffalo_l`（本项目） | 99.83 | 官方 Model Zoo |
| InsightFace ArcFace R50 | 99.65+ | [arcface 仓库](https://github.com/chenggongliang/arcface) |
| OpenCV SFace | 99.40 | [opencv_zoo 模型卡](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md) |
| dlib `face_recognition_ex` | 99.38 | [dlib 官方示例](https://dlib.net/dnn_face_recognition_ex.cpp.html) |

> ⚠️ 这四个数字**不能拿来排名**。它们各自用了不同的验证子集、阈值搜索方式和
> 预处理管线，绝对值普遍比上一张表高。跨表比较只有「同一张表内」才成立。
>
> 另外注意：本工具输出的**百分比不是准确率**。那是当前两张图之间的余弦相似度，
> 见「为什么分数是 45~70 而不是 99」。

---

## 🔢算法：从图片到一个百分比

```
图片（路径 / 字节 / numpy 数组）
   │
   ├─ 1. 解码成 BGR uint8 数组
   │      路径带中文也能读：内部是 Path.read_bytes() + cv2.imdecode，
   │      不走 cv2.imread（它在 Windows 上开不了非 ASCII 路径）
   │
   ├─ 2. SCRFD 检测（固定 640x640 输入，置信度阈值 0.5）
   │      输出：若干个检测框 + 每张脸的 5 个关键点
   │      一张图有多张脸就都保留；一张脸都没有就返回空结果（不抛异常）
   │
   ├─ 3. 用检测器给的 5 个关键点 + ArcFace 标准模板做仿射对齐，裁成 112x112
   │      这一步在 insightface 内部完成（face_align.norm_crop），
   │      所以本工具不需要 2d106det 模型——它用的是检测器自带的 5 点，
   │      不是那个 106 点模型
   │
   ├─ 4. ArcFace 前向 → 512 维向量 → L2 归一化（模长 = 1.0）
   │      识别模型的输入是 112x112，按 (像素 - 127.5) / 127.5 归一化，输出 512 维
   │      向量再做一次 L2 归一化后，点积就等于余弦相似度
   │
   └─ 5. 相似度 = dot(a, b) × 100
```

代码位置：`insightface_face_similarity.py:272`（`encode_faces`）、
`insightface_face_similarity.py:326`（`cosine_similarity_percent`）。

### ⚡ 1:N 为什么快

候选图各算一次特征之后，**一次矩阵乘法**就出全部结果，不用 Python 循环：

```python
scores = (候选矩阵 @ 查询矩阵.T).max(axis=1) * 100
```

`insightface_face_similarity.py:354`（`similarity_matrix_percent`）。
`.max(axis=1)` 表示"一个候选 vs 查询图里所有脸，取最高的那一对"，
所以**查询图传合照也能正常找出最像的那个人**。

### 🔀 和 InsightFace 默认行为的唯一区别

insightface 2.0 的 `app.prepare()` 默认 `det_size=None`（Auto），
内部会同时跑 128x128 和 640x640 两次检测再合并结果。
本工具**固定只用 640x640**，这样做只有一个目的：
**让分数可复现**——同一个人在同样的两张图上，永远得到同样的分数。
代价是小尺寸人脸略吃亏（见下一节）。

---

## ❓为什么分数是 45~70 而不是 99

因为 512 维 ArcFace 向量之间的**原始余弦值本来就不接近 1**。
这是深度人脸识别的常态：模型追求的是"角度可分"，不是"角度归一化"。

本机实测参考值（buffalo_l，纯 CPU）：

| 场景 | 相似度 |
|---|---|
| 同一张图和自己比 | **100.00%** |
| 同一张图，被检测器缩放后（拼成大合照再比） | 95 ~ 97% |
| 同一个人，换角度 / 表情 / 光线 | 50 ~ 70 |
| 两个不同的人 | 38 ~ 50 |
| 两个不同的人（差别较大） | 10 ~ 30 |

所以：

- **不同的人也可能到 50%**（撞脸、长相相近、亲缘关系），高分 ≠ 同一个人；
- **同一个人也可能只有 50%**，低分 ≠ 不同人；
- 阈值没有通用值，只能拿自己的数据标定。

> 顺带一提：如果你还见过 dlib 版本的 128 维分数（"同一个人 85% 以上"），
> 那两个百分比**不能互相比较**，是两把完全不同的尺子。阈值要重新标定。

---

## 💡一个容易踩的坑：脸在图里越小，分数越低

检测器固定按 640x640 看整张图。一张 4000x3000 的大合照会被缩到 640 宽，
里面的人脸只剩几十像素，提取出来的特征就不那么稳：
**同一张脸单独放大看是 70%，塞进大合照里可能只有 50%。**

- 人脸库里的图和查询图**分辨率接近时**，分数才可比；
- 拿一张几百像素的缩略图去搜 4K 大图，命中率会明显下降；
- 遇到这种情况，把图**先裁到人脸附近**再比，会准很多。

---

## 🔍取脸规则

一张图里可能有多张脸，各处代码的取脸规则不完全一样，公开列出来免得误会：

| 入口 | 查询图 | 候选 / 人脸库 |
|---|---|---|
| 网页面板 ①（`/api/query-set`） | **所有脸**都参与，取最高的一对 | **只取最大的那张脸** |
| 网页面板 ②（`/api/query-library`） | **所有脸**都参与 | **每张脸单独一条候选**（同一文件多张脸时标成 `文件名 #2`、`#3`） |
| `example2_insightface.py`（默认） | **所有脸**都参与 | **所有脸**都参与，取最高的一对 |
| `example2_insightface.py --largest-only` | 只取最大的脸 | 只取最大的脸 |
| `face_similarity_percent()`（核心库） | 默认只取最大的脸 | 默认只取最大的脸，加 `compare_all_faces=True` 改为任意配对 |

命令行和网页虽然走的是两条不同的打分路径（命令行用 `best_pair_percent`，
网页用 `similarity_matrix_percent`），但**取脸**这一步是同一个函数
`encode_faces`，它保证两条路径拿到的都是同一套按面积排序好的向量，
所以不会出现"网页能搜到、命令行搜不到"的取脸差异。

---

## 🚀快速开始

### 1. 环境要求

- **Python 3.10 或更新**（insightface 2.0 起的要求）
- Windows / Linux / macOS
- 直接装在当前环境里，不需要虚拟环境
- 首次运行需要联网下载模型；之后可完全离线

### 2. 安装

```bash
git clone https://github.com/wangyifan349/insightface-face-similarity.git
cd insightface-face-similarity

python -m pip install -U pip
```

> 下面所有命令都在**当前环境**里执行。用了 `python -m pip` 而不是裸 `pip`，
> 就是为了确保包装到当前这个 Python 里（Windows 上 `pip` 有时会指向另一个解释器）。

#### 一行检测有没有 NVIDIA GPU

PowerShell / cmd / bash 都能直接粘这一行（输出用 ASCII，避免 Windows 控制台编码问题）：

```bash
python -c "import shutil,subprocess; g=shutil.which('nvidia-smi') and subprocess.run(['nvidia-smi'],capture_output=True).returncode==0; print('[GPU ] NVIDIA GPU found -> install onnxruntime-gpu' if g else '[CPU ] no NVIDIA GPU found -> install onnxruntime')"
```

打印 `[GPU ]` 就走下面的 GPU 那一行，打印 `[CPU ]` 就走 CPU 那一行。

#### 按检测结果条件化安装

同样一行，检测到 GPU 就装 GPU 版，否则装 CPU 版：

```bash
python -c "import shutil,subprocess,sys; g=shutil.which('nvidia-smi') and subprocess.run(['nvidia-smi'],capture_output=True).returncode==0; p=(['insightface','onnxruntime-gpu'] if g else ['insightface','onnxruntime'])+['opencv-python','numpy','flask']; print('pip install -U '+' '.join(p)); sys.exit(subprocess.call([sys.executable,'-m','pip','install','-U']+p))"
```

想看清每一步、或者要复现到别处，就分开写：

```bash
# 有 N 卡
python -m pip install insightface onnxruntime-gpu opencv-python numpy flask

# 无 N 卡
python -m pip install insightface onnxruntime opencv-python numpy flask
```

`insightface` 本身就会带上 `onnxruntime`、`opencv-python`、`numpy`，
上面写全是为了让意图清楚、少踩坑。

> ⚠️ `nvidia-smi` 能跑通 **不等于** GPU 版能跑起来。
> `onnxruntime-gpu` 还要对上本机 CUDA / cuDNN 版本，否则会自动退回 CPU。
> 验证方法见「GPU 部署」一节的 `info` 命令。
>
> macOS 没有 `nvidia-smi`，永远走 CPU 依赖（CoreML 可手动指定）。

### 3. 确认装好了

```bash
python insightface_face_similarity.py info
```

输出类似：

```
model=buffalo_l dim=512 providers=not loaded yet det=640x640@0.5 models=C:\Users\<你>\.insightface\models\buffalo_l
```

- `providers` 是实际使用的推理后端；刚跑 `info` 还没加载模型，所以显示 `not loaded yet`，
  真正比对过一次之后（或用 `python -c "import insightface_face_similarity as m; m.warm_up(); print(m.describe_configuration())"`）才会显示实际后端
- `models` 是模型文件所在位置
- 如果显示的是"需要下载"，说明第一次运行会去拉取约 326 MB 的 `buffalo_l`

### 4. 跑起来

```bash
# 最快验证
python insightface_face_similarity.py a.jpg b.jpg

# 网页版
python flask_insightface_face.py
```

---

## 🖥用法一：网页版

```bash
python flask_insightface_face.py                     # 默认人脸库 = 脚本同级 face_library\
python flask_insightface_face.py --port 5100         # 指定起始端口
python flask_insightface_face.py --dir D:\faces      # 指定人脸库文件夹
python flask_insightface_face.py --library-info      # 只统计人脸库有多少图多少人脸，然后退出
```

启动时先加载模型（约 1 ~ 2 秒），然后打印真实地址：

```
Open http://127.0.0.1:5000 in your browser
```

端口被占用时会自动往后顺延（最多试 20 个），**以终端打印的地址为准**。

同一个页面里有两个面板，都用 AJAX 提交，不整页刷新：

**① 查询图 vs 候选图片（1:N）**
选 1 张查询图 + 多张候选图（浏览器里按住 Ctrl 多选）。只比这次上传的候选，**完全不读**本地人脸库。

**② 查询图 vs 人脸库目录（1:N）**
选 1 张查询图 + 填一个文件夹路径，和该文件夹里**全部**图片逐个比，递归扫描子目录。
文件夹有三级优先级：**网页输入框 > `INSIGHTFACE_FACE_LIBRARY_DIR` > 脚本同级的 `face_library\`**，
所以换个库只要在页面上粘个路径，不用改环境变量、不用重启。

人脸库规则：

- 递归扫描，识别 `.jpg .jpeg .png .bmp .webp`（大小写不敏感）
- 一张图里有多张脸时，第一张用文件名，其余记成 `文件名 #2`、`文件名 #3`
- 编码结果按「文件路径 + 修改时间」缓存：文件没动就不重复算，
  文件被改动或删除会自动失效 —— **改了库里的图不用重启**
- 上传体积上限：单次请求 64 MB，超出返回 HTTP 413

界面上会显示当前人脸库有多少张图、多少张脸，② 的目录输入框失焦时自动重新统计。

---

## ⌨用法二：命令行

```bash
# 1:1，两张图比一次
python example2_insightface.py a.jpg b.jpg

# 1:N，一张查询图逐个比所有候选，按相似度降序排名
python example2_insightface.py 查询这是谁.jpg 1用户1.jpg 2用户2.jpg 3帅哥3.jpg

# 1:N，候选可以直接给文件夹（递归收图）
python example2_insightface.py 查询这是谁.jpg face_library -n 5

# 输出 JSON，便于脚本处理
python example2_insightface.py 查询这是谁.jpg face_library --json

# 双击本文件（不带任何参数运行）进入交互模式
python example2_insightface.py
```

| 参数 | 作用 |
|---|---|
| `-n, --top N` | 最多显示前 N 条，`0` 或不写表示全部 |
| `--largest-only` | 只比每张图里最大的一张脸，而不是任意配对 |
| `--no-recursive` | 候选文件夹只看第一层 |
| `--no-bar` | 不画相似度条 |
| `--json` | 输出 JSON |
| `-h, --help, info` | 打印帮助 |

交互模式菜单：`1)` 1:1、`2)` 1:N、`0)` 退出。路径可以直接粘贴，
带引号、相对路径、`%变量%` 都能识别；相对路径找不到时会自动到脚本所在目录再找一次。
看完结果按回车回到主菜单，可以连着比很多轮。

输出细节：进度条固定 10 格；多个候选时相似度条按**本次结果的最低分到最高分缩放**
（ArcFace 分数本来就挤在 40~70 这一段，用固定 0~100 画每条都一样长），
表格下方会打印实际缩放区间，**左边的数值始终是真实相似度**；
只有一个结果、或所有结果差不到 0.5 时，改用真实的 0~100 尺度。
中文按两个字符宽度对齐，表格控制在 78 列内不折行；
终端编码不支持的符号会自动降级（`█` `░` `─` `…` → `#` `.` `-` `...`）。

退出码：`0` 正常完成；`1` 有一张图检测不到人脸或运行出错；`2` 用法错误 / 路径不存在 / 没有可用候选。

---

## 📚用法三：当库用

```python
from insightface_face_similarity import face_similarity_percent

percent = face_similarity_percent("a.jpg", "b.jpg")
if percent is None:          # 有一张图没检测到人脸
    print("未检测到人脸")
else:
    print(f"{percent:.2f}%")
```

1:N 的标准写法：查询图只算一次，候选各算一次，之后全靠矩阵乘法。

```python
from insightface_face_similarity import encode_faces, similarity_matrix_percent

query = encode_faces("查询.jpg")          # [query_vec, ...]
labels, rows = [], []
for path in 候选列表:
    faces = encode_faces(path)
    if not faces:
        continue
    labels.append(path)
    rows.append(faces[0])                 # 只取这张图最大的脸

scores = similarity_matrix_percent(query, rows)
排名 = sorted(zip(labels, scores), key=lambda r: r[1], reverse=True)
```

主要公开接口：

| 函数 | 作用 |
|---|---|
| `encode_faces(image)` | 每张脸返回一个 512 维 float32 向量（已 L2 归一化），**按检测框面积从大到小排序**，所以 `[0]` 是最大那张脸；无人脸返回 `[]` |
| `encode_faces_with_scores(image)` | 同上，每项额外带 `det_score`（检测置信度）和 `area` |
| `face_similarity_percent(a, b, array_is_bgr=True, compare_all_faces=False)` | 两张图的相似度百分比，无人脸返回 `None` |
| `cosine_similarity_percent(a, b)` | 直接比两个已取好的向量 |
| `best_pair_percent(a, b, largest_only=False)` | 两组向量之间最高的一对 |
| `similarity_matrix_percent(query, candidates)` | 1:N 搜索走这条路径，返回 1 维百分比数组 |
| `to_bgr_array(image, array_is_bgr=True)` | 把路径 / 字节 / 数组统一成 BGR uint8 数组 |
| `get_app(det_threshold=None, det_size=None)` | 拿到缓存的 `FaceAnalysis`，第一次调用才真正加载模型，线程安全 |
| `warm_up()` | 预热并返回耗时毫秒数，服务端启动时调一次 |
| `find_model_directory()` | 按查找顺序找 `buffalo_l`，找不到返回 `None`（= 需要下载） |
| `describe_configuration()` | 一行配置摘要，排查环境问题用 |

`image` 支持：文件路径 / `pathlib.Path` / 图片字节 / numpy 数组
（数组可以是 BGR、RGB 或灰度；浮点数组最大值不超过 1.0 会自动乘 255）。

完整的参数、返回结构、异常清单见 `使用说明_insightface.txt`。

---

## 🔌HTTP 接口

全部返回 JSON，网页版可直接当后端调用。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 网页本身 |
| `GET` | `/api/library?dir=...` | 人脸库状态：`{ok, path, images, faces}`；目录不存在返回 400 |
| `POST` | `/api/query-set` | 面板 ①。表单：`query`（单文件）、`candidates`（可多个同名字段）、`top_results` |
| `POST` | `/api/query-library` | 面板 ②。表单：`query`、`library_dir`（留空用默认）、`top_results` |

`POST /api/query-set` 成功返回：

```json
{
  "ok": true,
  "mode": "query-set",
  "query": "查询.jpg",
  "query_faces": 1,
  "count": 5,
  "compared": 4,
  "matches": [
    { "rank": 1, "kind": "候选图片", "candidate": "2用户2.jpg",
      "similarity": 63.21, "bar_low": 41.5, "bar_high": 63.21, "best": true }
  ],
  "skipped": [ { "name": "坏图.jpg", "status": "未检测到人脸" } ],
  "elapsed_ms": 812
}
```

`skipped` 里的原因只有两种：「未检测到人脸」和「处理失败：…」。
**某个候选坏了不会影响其它候选照常排名。**

`top_results` 不传、传空串或传非数字时按 10 处理；能转成整数时限制在 1~100
（0 和负数变成 1，超过 100 按 100 截断）。

`POST /api/query-library` 成功返回 `{ok, mode, query, query_faces, library, library_images, library_faces, matches, elapsed_ms}`。

失败统一返回 HTTP 400 + `{"ok": false, "error": "中文错误信息"}`，
例如「请选择一张查询图片」「人脸库目录不存在: …」「人脸库里没有检测到人脸: …」。
上传超过 64 MB 返回 HTTP 413。

curl 示例：

```bash
curl -F "query=@查询.jpg" \
     -F "candidates=@1.jpg" -F "candidates=@2.jpg" \
     -F "top_results=5" \
     http://127.0.0.1:5000/api/query-set
```

---

## 🔧环境变量

| 变量 | 作用 | 默认值 |
|---|---|---|
| `INSIGHTFACE_MODEL_ROOT` | 模型根目录 | `~/.insightface` |
| `INSIGHTFACE_MODEL_DIR` | 直接指定放着 `.onnx` 的目录，跳过自动查找 | 空 |
| `INSIGHTFACE_PROVIDER` | onnxruntime 执行后端，如 `CPUExecutionProvider` | 自动（检测到 CUDA 就用 CUDA） |
| `INSIGHTFACE_FACE_LIBRARY_DIR` | 网页版人脸库目录 | 脚本同级的 `face_library\` |
| `PORT` | 网页版起始端口 | `5000` |

### 📂 模型查找顺序

按顺序找，找到就直接用，**整个过程不联网**：

1. `INSIGHTFACE_MODEL_DIR` 指向的目录（或它的上一级加 `buffalo_l`）
2. 脚本同级的 `insightface_models\buffalo_l\` 和 `models\buffalo_l\`
3. 脚本同级的 `buffalo_l\`
4. 脚本所在目录下的所有子目录（递归查找 `buffalo_l`）
5. `INSIGHTFACE_MODEL_ROOT\models\buffalo_l\`
6. `~/.insightface\models\buffalo_l\`

以上 6 处都找不到，才走 InsightFace 官方的下载流程。

**离线部署最省事的做法**：把整个 `buffalo_l` 文件夹拷到
`insightface_face_similarity.py` 同级目录：

```
insightface_face_similarity.py
     buffalo_l\det_10g.onnx
     buffalo_l\w600k_r50.onnx
```

（只放这 2 个文件就够，另外 3 个不会被加载。）
放好后**不联网也能跑**。也可以指向别处：

```bash
# Windows
set INSIGHTFACE_MODEL_DIR=D:\models\buffalo_l
# Linux / macOS
export INSIGHTFACE_MODEL_DIR=/opt/models/buffalo_l
```

---

## 📊性能实测

本机环境：纯 CPU（无 CUDA），Python 3.14，onnxruntime 1.30，insightface 2.0。

| 项目 | 数值 |
|---|---|
| 模型加载（只加载 2 个模型） | 1.1 ~ 1.9 s（磁盘缓存冷热有差别） |
| 单张 2400x1080 图片 | 0.13 ~ 0.23 s |
| 500 张人脸库首次搜索 | 约 1 分钟（每张 0.1 ~ 0.3 s） |
| 之后每次搜索 | 矩阵乘法，几十毫秒 |
| 1:1 冷启动（含 Python 启动 + 加载模型） | 约 2.7 s |
| 1:1 已加载模型 | 约 0.6 ~ 0.7 s（两张图各约 0.3 s） |

大库第一次慢是因为要逐张过一遍特征。之后走缓存，改了图会自动重算那一张。

---

## 🐳部署

### 1. 最简单的部署（本机 / 单机）

```bash
python flask_insightface_face.py --port 5000 --dir D:\faces
```

> ⚠️ **网页版默认只监听 `127.0.0.1`**，也就是只有本机能访问。
> 这是刻意的：这是个**没有登录、没有鉴权**的服务，而且会读取你指定的
> 任意文件夹里的图片。**不要**直接暴露到公网。
>
> 需要给同事在内网用的话，有两种做法：
>
> - **推荐**：用 Nginx / Caddy 反代，加 HTTPS 和访问控制，Flask 仍只听 127.0.0.1；
> - 或者改 `flask_insightface_face.py:908` 的 `app.run(host="127.0.0.1", ...)`
>   为 `host="0.0.0.0"`，**同时**确保前面有防火墙规则和鉴权层。
>
> 任何情况下都不要把 `/api/query-library` 指向含有敏感照片的目录后暴露出去，
> 因为这个接口能按需读取服务端任意目录里的图片。

### 2. GPU 部署

有 N 卡就把 onnxruntime 换成 GPU 版：

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

- **两个版本不要同时装**；
- 装 / 升级 insightface 可能会把 CPU 版 `onnxruntime` 装回来，N 卡机器上要再换一次；
- `onnxruntime-gpu` 需要与本机 CUDA / cuDNN 版本对得上，否则会退回 CPU。

确认是否真的用上了 GPU：

```bash
python insightface_face_similarity.py info
```

看到 `providers=CUDAExecutionProvider,...` 才算成功；
如果还是 `CPUExecutionProvider`，说明 GPU 版没装上或版本不匹配。
本工具只在检测到 CUDA 时才自动启用 GPU；
macOS 上不会自动选 CoreML，需要的话用 `INSIGHTFACE_PROVIDER` 手动指定。

```bash
# 强制只用 CPU
export INSIGHTFACE_PROVIDER=CPUExecutionProvider
```

### 3. 离线 / 内网部署

1. 在有网机器上跑一次 `python insightface_face_similarity.py info`，让它下好模型；
2. 把 `buffalo_l` 文件夹（至少 `det_10g.onnx` + `w600k_r50.onnx`）拷到内网机器；
3. 放到脚本同级目录，或设 `INSIGHTFACE_MODEL_DIR` 指向它；
4. `python insightface_face_similarity.py info` 确认 `models=` 指向本地路径。

Python 依赖本身也要提前离线准备：

```bash
# 有网机器（GPU 机器把 onnxruntime 换成 onnxruntime-gpu）
python -m pip download insightface onnxruntime opencv-python numpy flask -d wheels

# 内网机器
python -m pip install --no-index --find-links=wheels insightface onnxruntime opencv-python numpy flask
```

### 4. 做成常驻服务

`app.run()` 是 Flask 自带的开发服务器，适合本机和小规模内网使用。
要长期跑、或者给更多人用，建议套一层生产级 WSGI 服务器：
Linux 用 Gunicorn，Windows 用 waitress。两者都不能直接用
`flask_insightface_face:app` 的形式启动——那样会跳过 `main()` 里的模型预热，
第一个请求要自己承担 1~2 秒的加载。

**本仓库没有内置 `wsgi.py`**，先在仓库根目录建一个（会先预热模型再服务）：

```python
# wsgi.py
import flask_insightface_face as web
from insightface_face_similarity import warm_up

warm_up()                 # 进程启动就把模型加载掉
application = web.app
```

```bash
python -m pip install waitress
waitress-serve --listen=127.0.0.1:5000 wsgi:application
```

Linux 上用 Gunicorn 同理：

```bash
gunicorn --bind 127.0.0.1:5000 --workers 1 --threads 8 wsgi:application
```

> `--workers` 建议保持 1：每个 worker 都会各自加载一份模型（各占约 300 MB 内存），
> 人脸库特征缓存也**不跨进程共享**。要压榨 CPU，用 `--threads` 比加 worker 更划算。

systemd 示例：

```ini
# /etc/systemd/system/face-search.service
[Unit]
Description=InsightFace face similarity web
After=network.target

[Service]
WorkingDirectory=/opt/insightface-face-similarity
Environment=INSIGHTFACE_MODEL_DIR=/opt/models/buffalo_l
Environment=INSIGHTFACE_FACE_LIBRARY_DIR=/data/faces
ExecStart=/usr/local/bin/waitress-serve --listen=127.0.0.1:5000 wsgi:application
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now face-search
```

Windows 上用「任务计划程序」创建一个"计算机启动时"运行的计划任务：

- 程序：`python`
- 参数：`-m waitress --listen=127.0.0.1:5000 wsgi:application`
- 起始于：仓库目录

用 `python -m waitress` 而不是 `waitress-serve`，就不用去猜它装到哪个 Python 里了。

### 5. 容器部署（未在本机验证）

本机没有 Docker，所以下面这份 `Dockerfile` **没有实际跑过**，
首次使用请自行验证；本仓库实测可用的路径是上面 1~4 节。

```dockerfile
FROM python:3.11-slim

# opencv-python 需要这两个系统库，slim 镜像里默认没有
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir insightface onnxruntime opencv-python numpy flask waitress

# 本仓库不含 wsgi.py，容器里现建一个并预热模型
RUN printf 'import flask_insightface_face as web\nfrom insightface_face_similarity import warm_up\nwarm_up()\napplication = web.app\n' > /app/wsgi.py

# 模型不入库，容器首次启动时自动下载到 ~/.insightface（需要联网）。
# 离线环境请改成 volume 挂载已下载好的 buffalo_l，并设 INSIGHTFACE_MODEL_DIR。
VOLUME /root/.insightface

EXPOSE 5000

# 必须用 WSGI 服务器监听 0.0.0.0：
# 直接跑 flask_insightface_face.py 只听 127.0.0.1，容器外根本连不上。
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "wsgi:application"]
```

```bash
docker build -t face-search .
# 只映射到本机回环地址；确认安全后再考虑 -p 5000:5000
docker run --rm -p 127.0.0.1:5000:5000 -v /data/faces:/data/faces face-search
```

---

## 🆘常见问题

**Q 报 `No face was detected in one of the images`**
有一张图里没检测到人脸。极端侧脸、小脸、模糊都可能这样。
可以把检测阈值调低（`get_app(det_threshold=0.3)`），或先把图裁到人脸附近再试。

**Q 第一次运行卡住，然后报网络错误**
在下载 `buffalo_l` 模型包（约 326 MB），需要联网。
下好后放在 `~/.insightface/models/buffalo_l/`，之后就不需要网了。
也可以手动下载（官方 Model Zoo 表格里有 `buffalo_l` 的 Download 链接），
解压出 `buffalo_l` 文件夹放到脚本同级目录。

**Q 怎么确认模型装好了**
`python insightface_face_similarity.py info`，会打印模型名、维度、执行后端、模型位置。

**Q 装了 CUDA 版 onnxruntime还是很慢**
跑 `python insightface_face_similarity.py info` 看 `providers` 是不是
`CUDAExecutionProvider`。显示 `CPUExecutionProvider` 说明 `onnxruntime-gpu`
没装上、或者被装 insightface 时又换回了 CPU 版、或者和 CUDA 版本对不上。

**Q 网页打开是 5001 不是 5000**
5000 被别的程序占了，程序会自动顺延，**地址以终端打印的为准**。也可以 `--port` 指定。

**Q 人脸库改了文件要重启吗**
不用。库按「文件路径 + 修改时间」缓存，变了会自动重算；
目录输入框改路径也不需要重启。

**Q 人脸库第一次搜索很慢**
第一次要把整个目录逐张过一遍特征，每张 0.1 ~ 0.3 秒，500 张大约 1 分钟。
之后都走缓存，几十毫秒。终端会打印「载入人脸库 12/500」这样的进度。

**Q 双击后窗口一闪而过**
用不带参数的方式运行会停在主菜单等你输入。
若确实闪退，说明 Python 文件关联有问题，建议在命令行里运行看报错信息。

**Q 分数都很接近，看不出谁更谁**
相似度条和命令行表格都已经按本次的分数区间缩放了，**左边的数值才是真实分数**，
直接看数字就行。

**Q 同一张脸单独看 70%，放进大合照里搜就只剩 50%**
正常。检测器按 640x640 看整张图，图越大里面的人脸像素越少，特征就没那么稳。
把图裁到人脸附近再比，分数会回到 70%。

**Q 不同的人为什么也能到 50%**
ArcFace 就是这样，撞脸的人分数会很高。所以本工具**不替你判断**是不是同一个人，
阈值要拿自己的数据标定。

**Q 网页版有手机版界面吗**
有，页面用了 Bootstrap 5，手机浏览器能正常用，桌面和手机都是同一套界面。

**Q 会把照片传到网上吗**
不会。解码、检测、特征提取、比对全部在本地进程内完成，
网页版也是本机 Flask 直接处理上传的文件。insightface 在导入 onnxruntime 前
会设置 `ORT_DISABLE_TELEMETRY=1`。

---

## 🙏致谢与引用

本项目**没有实现任何人脸算法**，全部算法能力来自 InsightFace 官方开源项目。
没有它就没有这个仓库，非常感谢 Jia Guo、Jiankang Deng 等维护者。

- **InsightFace 官方仓库**：<https://github.com/deepinsight/insightface>
- 官方网站：<https://insightface.ai>
- PyPI：<https://pypi.org/project/insightface/>
- Model Zoo（模型包、下载链接、官方精度）：<https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md>
- 运行环境与执行后端说明：<https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md>
- 相关论文：
  - SCRFD（本工具的检测器）—
    Guo, Deng, Lattas, Zafeiriou. *Sample and Computation Redistribution for
    Efficient Face Detection.* [arXiv:2105.04714](https://arxiv.org/abs/2105.04714)
  - ArcFace（本工具的特征提取器）—
    Deng, Guo, Yang, Xue, Kotsia, Zafeiriou. *ArcFace: Additive Angular Margin
    Loss for Deep Face Recognition.* [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)

也感谢 [ONNX Runtime](https://onnxruntime.ai)（推理后端）和
[Flask](https://flask.palletsprojects.com/)（网页）。

欢迎提 Issue、Pull Request、翻译，或把你自己的阈值标定经验贴出来一起讨论。

---

## 📜许可

- **本仓库代码：MIT License**，允许商用、修改、闭源分发，只需保留版权声明。
  发布时请补上 `LICENSE` 文件。
  需要明确的专利授权条款就换 Apache-2.0（同样允许商用，两者都比 GPL 宽松）。
- **模型权重不在本仓库里**：InsightFace 代码是 MIT，但官方预训练权重
  **仅限非商业研究用途**。商用请联系官方授权，或换成自己有权使用的模型。
  详见 [Model licenses](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md#model-licenses)。

> 免责：本工具**不做人脸身份认定**，分数只是余弦相似度，
> 阈值与后果由使用者自行判断；处理他人人脸请遵守当地隐私法规并取得同意。
