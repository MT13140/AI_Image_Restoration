# models 目录说明

本目录用于放置 AI 修复模型权重。项目已经放好 LaMa 权重，**可以离线直接使用**。

---

## 1. 默认模型：LaMa (big-lama)

| 项目 | 说明 |
| --- | --- |
| 模型名称 | LaMa / big-lama（Large-Mask inpainting） |
| 论文 | *Resolution-robust Large Mask Inpainting with Fourier Convolutions*（WACV 2022） |
| 官方实现 | <https://github.com/advimman/lama> |
| 本项目使用的权重 | `big-lama.pt`（TorchScript，来自 `simple-lama-inpainting` v0.1.0 官方 release） |
| 下载地址（真实可用） | <https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt> |
| 文件大小 | 196.3 MB（205,803,670 字节） |
| SHA-256 | `7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c` |
| 放置位置 | `models/big-lama.pt`（本目录） |
| CPU 支持 | ✅ 支持（`--index-url .../whl/cpu` 安装的 PyTorch 即可运行，速度较慢） |
| GPU 推荐 | ✅ 强烈推荐。CUDA 12.1 版 PyTorch + 8GB 显存即可流畅处理 2000px 级图片 |
| 输入要求 | 尺寸需为 8 的倍数（程序内部自动反射填充，用户无需关心） |

### 手动下载 / 重新下载

方式 A：直接下载后放到本目录，文件名保持 `big-lama.pt`

```bat
curl -L -o models\big-lama.pt ^
  https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

方式 B：什么都不做，直接点界面上的「🚀 开始 AI 修复」或「⬇️ 预加载模型」，
程序检测到 `models/big-lama.pt` 缺失时会自动下载（支持断点续传）到本目录。

方式 C：自定义权重位置/文件名——`
把 TorchScript 权重命名为以下任一名字放进本目录即可被自动识别：
`big-lama.pt`、`big_lama.pt`、`big_lama_fp32.pt`、`lama_fp32.jit`、`big-lama.jit`。
也可以通过环境变量 `LAMA_MODEL_URL` 指定其它下载地址。

### 程序如何加载

`core/inpainting.py` 中的 `LaMaBackend`：

1. 优先在本目录寻找上述文件名，用 `torch.jit.load(文件对象)` 加载
   （用文件对象而非路径，是为了规避 PyTorch JIT 在 Windows 上无法打开
   含中文/非 ASCII 路径的问题，例如用户名是中文的情况）；
2. 未找到则用纯 Python 实现的断点续传下载到本目录后再加载；
3. 仍然失败时，才回退到 `simple-lama-inpainting` 包自带封装；
4. 全部失败时，界面会明确提示原因，并可选择使用 OpenCV Telea（传统算法）兜底。

---

## 2. OCR 模型：PP-OCRv4（随依赖包分发，无需手动下载）

| 项目 | 说明 |
| --- | --- |
| 模型名称 | PP-OCRv4 det / cls / rec（PaddleOCR 的检测、方向分类、识别模型） |
| 来源 | PaddleOCR 官方模型，由 `rapidocr-onnxruntime` 转换为 ONNX 并随 wheel 一起分发 |
| 实际位置 | `.venv\Lib\site-packages\rapidocr_onnxruntime\models\` |
| 文件 | `ch_PP-OCRv4_det_infer.onnx`、`ch_PP-OCRv4_rec_infer.onnx`、`ch_ppocr_mobile_v2.0_cls_infer.onnx` 等 |
| 大小 | 约 15 MB（合计） |
| CPU 支持 | ✅ 支持（默认使用 CPU 版 onnxruntime，速度快且无需 CUDA 依赖） |
| GPU | 可选：安装 `onnxruntime-gpu` 可加速，但需要匹配的 CUDA/cuDNN，默认不启用 |
| 语言 | 中文 + 英文 + 数字（PP-OCRv4 中文模型同时支持英文与数字） |

> 说明：本项目默认**不**安装完整 PaddlePaddle。PaddleOCR 的原生 Windows 安装
> 依赖较重且容易与已有环境冲突，因此采用官方推荐的 ONNXRuntime 路径（RapidOCR），
> 使用的是同一套 PP-OCRv4 模型。若你已安装 PaddleOCR，可设置环境变量
> `AI_RESTORE_OCR=paddle` 切换到 PaddleOCR 后端（`core/ocr_detector.py` 已实现两种后端）。

---

## 3. 人脸检测模型（第四轮新增，用于人脸保护）

| 项目 | 说明 |
| --- | --- |
| 模型 | **YuNet**（OpenCV 官方轻量人脸检测器，带 5 点关键点：双眼/鼻尖/嘴角） |
| 文件 | `models/face_detection_yunet_2023mar.onnx`（227 KB） |
| 来源 | OpenCV Zoo `face_detection_yunet`（本机通过 hf-mirror 镜像取得同一文件） |
| 作用 | 检测人脸 → 判断水印是否压到五官 → 启用"人脸保护 / 五官硬保护 / 局部高分辨率重建" |
| 依赖 | 只需 `opencv-python`（已安装），**不新增任何 Python 依赖**，不需要 CUDA |
| 缺失时 | 自动退回 OpenCV 自带的 Haar 级联（完全离线，但没有人脸关键点，保护粒度略粗） |
| GPU 支持 | 不需要（CPU 检测一次约 20~60ms） |

> 该模型只用于"保护原始人脸"，不参与图像生成；`models/big-lama.pt` 仍是唯一的修复模型。

---

## 4. 本目录中**没有**的模型（避免误解）


* **通用 Logo 检测模型**：本项目未内置任何 Logo 深度学习检测模型。
  界面中的「检测 Logo 候选」是基于颜色/形状的**启发式**候选提取，
  结果需要人工确认（`core/watermark_detector.py::LogoDetector`）。
  该类已独立封装，方便日后替换为真实模型。
* **Stable Diffusion / MAT 等其它 Inpainting 模型**：未内置。
  `core/inpainting.py` 按后端接口设计，可在其基础上扩展新后端。

---

## 5. 磁盘占用参考

| 内容 | 大小 |
| --- | --- |
| `big-lama.pt` | 196 MB |
| PyTorch（CUDA 12.1 版） | 约 2.4 GB（安装在 `.venv` 中） |
| 其它依赖（Gradio / OpenCV / OCR 等） | 约 400 MB |
| 合计 | 约 3 GB |
