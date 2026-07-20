# 16×16 Pressure-Mat Posture Recognition System

基于16×16薄膜压力传感器阵列的坐姿识别系统，支持CSV数据回放、占用状态判断、人体与物品区分、多阶段坐姿识别、Boundary安全回退以及桌面GUI可视化。

## 功能概览

- 读取16×16压力矩阵帧
- 判断 EMPTY、低负载、OBJECT、UNKNOWN和HUMAN
- 仅在稳定人体状态下进入坐姿识别
- 使用随机森林、Multi-prototype和时间平滑完成文件级识别
- 支持后靠坐姿二阶段细分类
- 支持候选侧向姿势局部解析器
- 16×16实时热力图显示
- CSV播放、暂停、进度显示与结果导出
- 导出模型版本和artifact哈希，保证结果可追溯
- 不确定时使用Boundary安全回退，避免强制错误分类

## 项目结构

```text
posture-recognition-system/
├── posture_csv_app.py
├── recognizer_api.py
├── API_DOCUMENT.md
├── CURRENT_MODEL.md
├── recognizer/
│   ├── recognizer_api.py
│   ├── occupancy_detector.py
│   ├── seat_detector.py
│   ├── seat_analyzer.py
│   ├── feature_extractor.py
│   ├── rf_recognizer.py
│   ├── prototype_bank.py
│   ├── smoothing.py
│   ├── csv_gui.py
│   ├── csv_gui_core.py
│   ├── models/
│   └── tests/
└── posture_dataset_v2/
    └── scripts/
```

原始CSV、开发数据和生成报告默认保存在本地，不提交到GitHub。

## 环境要求

推荐使用 Python 3.10 或更高版本。

主要依赖包括：

```text
numpy
pandas
scikit-learn
joblib
tkinter
```

安装Python依赖：

```bash
python3 -m pip install numpy pandas scikit-learn joblib
```

macOS自带的Python环境通常包含Tkinter；如果GUI无法启动，请检查当前Python是否支持Tk。

macOS实时串口读取依赖可以使用项目提供的运行依赖文件安装：

```bash
python3 -m pip install -r requirements-macos.txt
```

## 启动桌面软件

进入项目目录：

```bash
cd "/path/to/posture-recognition-system"
```

启动当前默认模型：

```bash
python3 posture_csv_app.py
```

当前默认模型由以下文件决定：

```text
recognizer/models/default_model.json
```

目前默认版本为：

```text
v2_2_candidate
```

GUI显示名称：

```text
V2.2（H3闭卷通过）
```

## 显式加载模型版本

```bash
python3 posture_csv_app.py --model-version v1
python3 posture_csv_app.py --model-version v2_candidate
python3 posture_csv_app.py --model-version v2_1_candidate
python3 posture_csv_app.py --model-version v2_2_candidate
python3 posture_csv_app.py --model-version v2_3_candidate
python3 posture_csv_app.py --model-version v2_3_1_candidate
python3 posture_csv_app.py --model-version v2_4_candidate
python3 posture_csv_app.py --model-version v2_4_1_candidate
python3 posture_csv_app.py --model-version v2_4_2_candidate
python3 posture_csv_app.py --model-version v2_4_3_candidate
```

除默认版本外，其他版本主要用于实验、诊断和回归测试。

## Windows免Python版本

项目提供GitHub Actions自动构建的Windows文件夹版桌面程序。Windows电脑无需安装Python、pip或任何Python依赖。

使用步骤：

1. 打开GitHub仓库 `Jennifer288/posture-recognition-system`。
2. 进入Actions页面，运行 `Build Windows App`。
3. 构建完成后下载Artifact：`PostureRecognition-Windows-V243`。
4. 解压下载的ZIP文件。
5. 双击 `PostureRecognition-V243.exe` 启动。

该Windows程序会显式加载：

```text
v2_4_3_candidate
```

请保留解压后的整个 `PostureRecognition-V243/` 文件夹，不要只单独复制EXE。

## 统一识别API

硬件端只需持续提供一个16×16压力帧：

```python
from recognizer_api import Recognizer

recognizer = Recognizer()

frame = read_frame()  # numpy.ndarray, shape=(16, 16)
result = recognizer.predict(frame)
```

显式加载候选模型：

```python
recognizer = Recognizer(model_version="v2_4_3_candidate")
```

重置时间状态：

```python
recognizer.reset()
```

更多接口说明见：

```text
API_DOCUMENT.md
```

## 当前默认模型：V2.2

V2.2以V2.1为父模型，并增加后靠坐姿二阶段分类器，用于区分：

- 后仰靠背坐
- 后靠/瘫坐类

细分类证据不足时安全回退为：

```text
后靠坐姿
```

H3 external holdout结果：

```text
correct_accept:   3/4
correct_fallback: 1/4
wrong_accept:     0/4
gate_miss:        0/4
safe_resolution:  4/4
```

因此V2.2目前仍是默认运行版本。

## 侧向姿势候选版本

后续候选版本探索了侧向姿势识别。

当前候选标签体系将：

```text
标准侧坐 + 侧身倚靠坐
```

合并为：

```text
侧向坐姿
```

同时保留：

```text
斜跨坐
```

不确定时回退为：

```text
侧向姿势
```

这些版本目前属于实验候选，没有替换默认V2.2。可使用以下命令体验最新候选：

```bash
python3 posture_csv_app.py --model-version v2_4_3_candidate
```

## GUI说明

桌面GUI支持：

- 选择并加载CSV
- 播放、暂停和重置
- 16×16正方形压力热力图
- 前、后、左、右方向标识
- 当前Occupancy状态
- 父模型和局部分类结果
- Boundary与安全回退状态
- 最终显示标签
- 手动查看可滚动模型详情
- 导出逐帧预测、姿势片段和汇总结果

CSV播放完成后只更新状态栏，不会自动弹出超长模型JSON窗口。

## 导出文件

GUI可导出：

```text
frame_predictions.csv
posture_segments.csv
summary.json
```

导出内容包括：

- 模型版本
- 父模型版本
- 子模型版本
- 最终姿势标签
- Boundary与fallback信息
- 模型artifact哈希
- 文件级统计结果

## 自动化测试

运行核心测试：

```bash
python3 -m unittest recognizer.tests.test_recognizer_core
```

测试覆盖：

- Occupancy与物品拦截
- 模型版本加载
- 后靠二阶段识别
- 侧向候选解析器
- Boundary安全回退
- 非侧向回归保护
- GUI热力图几何
- CSV播放完成逻辑
- 模型详情窗口
- 导出字段完整性

## 数据与验证说明

为避免数据泄漏：

- 训练和验证按完整CSV文件分组
- 不采用同一CSV窗口随机拆分
- 已使用的development数据不能再次作为闭卷数据
- 已使用的external holdout不能加入训练或调参
- 模型发生修改后，需要新的独立holdout才能重新宣称闭卷通过

大型原始数据、报告和运行输出通过 `.gitignore` 保留在本地。

## 当前限制

- 系统主要面向普通办公椅和16×16座面压力垫
- 座面传感器无法直接观察肩膀、上半身角度或靠背接触
- 躺卧类暂不适用于普通办公椅场景
- 候选侧向模型尚未替换正式默认版本
- 当前仓库主要提供CSV回放接口；实时硬件接入可通过统一Recognizer API完成

## License

本项目目前用于个人作品集、研究与演示。
