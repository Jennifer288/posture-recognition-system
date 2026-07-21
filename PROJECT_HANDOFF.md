# Posture Recognition System 项目交接

## 1. 项目位置

本地目录：

/Users/yongqizhang/Documents/posture-recognition-system

GitHub：

Jennifer288/posture-recognition-system

当前分支：

main

## 2. 当前完成状态

项目已经完成：

1. FlexPressureVision CSV 坐姿回放识别
2. 16×16 压力热力图
3. Occupancy 检测
4. 坐姿分类
5. Boundary 判断
6. Prototype 辅助诊断
7. 时间平滑
8. macOS 串口协议解析
9. macOS 实时串口读取
10. macOS 实时坐姿识别 GUI
11. 空载校准
12. 串口方向调整
13. 安全连接、断开和异常处理

## 3. 当前实时识别流程

压力坐垫
→ macOS USB 串口
→ SerialFrameReader
→ PressurePacketParser
→ 16×16 压力矩阵
→ Recognizer.predict(frame)
→ 实时热力图和坐姿结果

模型运行在 Mac 本地，不在 GitHub 云端运行。

## 4. 当前模型

实时 GUI 显式使用：

v2_4_3_candidate

识别接口：

```python
result = recognizer.predict(frame)

frame 要求：

frame.shape == (16, 16)
5. 串口设备

当前 Mac 端口示例：

/dev/cu.usbserial-130

串口参数：

Baudrate: 460800
Data bits: 8
Parity: None
Stop bits: 1
Flow control: None

运行实时软件前必须关闭 CoolTerm，避免串口被占用。

CoolTerm 不是日常运行必需品，只用于串口调试。

6. 串口协议

完整帧：

55 AA | 01 01 | 01 | 256 bytes payload | checksum | 5A

说明：

帧头：55 AA
长度：01 01，小端，等于257
功能码：01
payload：256个uint8压力点
checksum：当前通常为00，不强制校验
帧尾：5A
总长度：263字节

矩阵按照逐列顺序传输，解析方式：

values.reshape((16, 16), order="F")
7. 关键文件

CSV 软件：

posture_csv_app.py
recognizer/csv_gui.py
recognizer/csv_gui_core.py

实时串口软件：

posture_serial_app_macos.py
recognizer/serial_gui.py
recognizer/serial_gui_core.py
recognizer/frame_reader.py
recognizer/serial_protocol.py

测试：

recognizer/tests/test_serial_protocol.py
recognizer/tests/test_frame_reader.py
recognizer/tests/test_serial_gui_core.py
recognizer/tests/test_recognizer_core.py

Mac 依赖：

requirements-macos.txt
8. 关键 Git 提交

275ed25 Add macOS pressure serial protocol and reader

a099dfd Add macOS live serial posture recognition GUI

9. Mac 启动方式
cd "/Users/yongqizhang/Documents/posture-recognition-system"
source .venv/bin/activate
python posture_serial_app_macos.py
10. 测试命令
python -m unittest discover -s recognizer/tests -v

最近本机结果：

Ran 123 tests
OK
skipped=3
11. 实际设备验证结果

已经验证：

串口可以连接
接收字节持续增加
有效帧持续增加
FPS接近20
错误帧接近0
热力图实时变化
GUI不卡顿
坐姿可以实时识别
起身后可以恢复空载
断开操作正常
12. 不要破坏的内容

后续任务不要随意修改：

现有 CSV 回放功能
posture_csv_app.py
posture_csv_app_windows.py
requirements-windows.txt
GitHub Actions
default_model.json
模型 artifact
标签体系
Boundary逻辑
Prototype逻辑
v2_4_3_candidate模型行为
13. 数据说明

旧 CSV 数据仍然保留，用于：

模型训练
回归测试
数据质量分析
CSV回放

实时识别不需要先导出CSV或BIN文件，而是直接读取坐垫正在发送的串口字节流。

14. 新账号接手要求

新 Codex 账号进入项目后，首先：

阅读 README.md
阅读 PROJECT_HANDOFF.md
查看 git log --oneline
查看 git status
运行全部测试
不要重新搭建项目
不要重新实现已经存在的串口解析器和实时GUI
修改前先提交稳定节点

保存后提交：

```bash
git add PROJECT_HANDOFF.md
git commit -m "Add project handoff documentation"
git push origin main