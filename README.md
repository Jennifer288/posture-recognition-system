# 食材保鲜管家

这是一个冰箱食材管理作品集 Demo，包含两个交付形态：

- Web Demo：可直接通过 GitHub Pages 打开。
- Android App：真正的 Android WebView 外壳，可构建 debug APK 安装到手机。

在线网页 Demo：

[https://jennifer288.github.io/fridge-food-android-demo/](https://jennifer288.github.io/fridge-food-android-demo/)

## 功能概览

- 冰箱拍照识别模拟流程
- `全部 / 肉蛋奶 / 蔬果 / 饮品 / 冷冻` 分类筛选
- 食材新鲜度、剩余天数、储存区域和状态展示
- 今天、明天、已过期、新鲜度低的提醒列表
- 食材详情底部面板
- `food_freshness_logic.c` 展示嵌入式/底层 C 逻辑组织方式

## Web Demo

网页源码位于：

```text
fridge-food-android-demo/
```

本地打开：

```bash
cd fridge-food-android-demo
python3 -m http.server 8080
```

然后访问：

```text
http://localhost:8080
```

## Android WebView App

Android 工程位于：

```text
android-app/
```

App 信息：

- App 名称：`食材保鲜管家`
- Package name：`com.jennifer.fridgefood`
- minSdk：`23`
- 实现方式：Android WebView
- 本地入口：`file:///android_asset/fridge-food-android-demo/index.html`
- 不申请相机权限；“拍照识别”仍使用 demo 模拟数据
- 不依赖网络即可打开 App 主界面

WebView 加载的离线资源位于：

```text
android-app/app/src/main/assets/fridge-food-android-demo/
```

## 构建 APK

### 使用 Android Studio

1. 打开 Android Studio。
2. 选择 `Open`。
3. 打开本仓库中的 `android-app/` 目录。
4. 等待 Gradle Sync 完成。
5. 选择 `Build > Build Bundle(s) / APK(s) > Build APK(s)`。
6. 构建完成后，APK 位于：

```text
android-app/app/build/outputs/apk/debug/app-debug.apk
```

### 使用命令行

需要本机安装 Android SDK，并设置 `ANDROID_HOME` 或 `ANDROID_SDK_ROOT`。

```bash
cd android-app
./gradlew assembleDebug
```

生成的 APK 路径：

```text
android-app/app/build/outputs/apk/debug/app-debug.apk
```

## GitHub Actions 下载 APK

仓库包含 workflow：

```text
.github/workflows/build-android-apk.yml
```

每次 push 到 `main` 后会自动运行 `Build Android APK`，构建 debug APK 并上传 artifact。

下载方式：

1. 打开仓库的 `Actions` 页面。
2. 选择最新的 `Build Android APK` run。
3. 在页面底部 `Artifacts` 区域下载：

```text
fridge-food-android-debug-apk
```

解压后可得到：

```text
app-debug.apk
```

## 安装到 Android 手机

### 方法一：手机直接安装

1. 从 GitHub Actions 下载 `fridge-food-android-debug-apk`。
2. 解压得到 `app-debug.apk`。
3. 把 APK 传到 Android 手机。
4. 在手机上打开 APK。
5. 如果系统提示，允许“安装未知来源应用”。
6. 安装完成后打开 `食材保鲜管家`。

### 方法二：ADB 安装

手机开启 USB 调试后运行：

```bash
adb install -r android-app/app/build/outputs/apk/debug/app-debug.apk
```

## C 逻辑说明

核心 C 展示文件：

```text
fridge-food-android-demo/food_freshness_logic.c
```

Android App 当前不编译这个 C 文件；它作为嵌入式/Native 层逻辑展示，用 `enum`、`struct`、bit flag 和纯函数表达食材识别结果、新鲜度状态、临期提醒和分类筛选。

## 验证

网页逻辑测试：

```bash
node fridge-food-android-demo/tests/logic.test.js
```

C 语法检查：

```bash
clang -fsyntax-only fridge-food-android-demo/food_freshness_logic.c
```

Android 项目结构检查：

```bash
python3 android-app/tests/verify_android_project.py
```

Android APK 构建：

```bash
cd android-app
./gradlew assembleDebug
```

## 16×16坐垫CSV识别软件

本仓库同时包含本地电脑端坐姿识别软件。默认启动命令：

```bash
python3 posture_csv_app.py
```

当前默认模型由 `recognizer/models/default_model.json` 指向
`v2_2_candidate`，GUI 显示为 `V2.2（H3闭卷通过）`。

历史版本仍可显式回退：

```bash
python3 posture_csv_app.py --model-version v1
python3 posture_csv_app.py --model-version v2_candidate
python3 posture_csv_app.py --model-version v2_1_candidate
python3 posture_csv_app.py --model-version v2_2_candidate
python3 posture_csv_app.py --model-version v2_3_candidate  # V2.3候选：侧向三类局部解析，未闭卷
```

`v2_2_candidate` 是当前默认运行模型：父模型仍为 V2.1，仅在后靠相关
窗口上进一步区分“后仰靠背坐”和“后靠/瘫坐类”；无法安全细分时显示
安全回退标签“后靠坐姿”。

V2.2 在 H3 external holdout 晋级测试中：

- correct_accept：3/4
- correct_fallback：1/4
- wrong_accept：0/4
- gate_miss：0/4
- safe_resolution：4/4

V2.1 在 holdout_batch_02 Phase 1 闭卷测试中：

- Boundary-aware 文件准确率：10/12 = 83.33%
- V2 candidate：75.00%
- V1：16.67%
- wrong accepted files：0
- object pressure entering posture model：0

限制：H1/H2 是 V2.2 development 数据；H3 已被用于晋级决策，不能再作为未来未见 holdout，也不能进入训练或调参。未来任何 V2.2 改动都必须使用全新的 H4 或后续批次做闭卷验证。

CSV GUI 导出的 `frame_predictions.csv`、`posture_segments.csv` 和
`summary.json` 会记录 `model_version`、`model_artifact_sha256`、
`metadata_sha256`、`runtime_config_sha256`，以及 V2.2 的 submodel
hash，便于追溯每份识别结果使用的模型。

### V2.3 Candidate

`v2_3_candidate` is candidate-only. It keeps V2.2 as the parent recognizer and adds a local lateral resolver for `标准侧坐` / `斜跨坐` / `侧身倚靠坐`, with `侧向坐姿` as the safe Boundary fallback. It is not the default model and still requires a fresh closed-book holdout before promotion.
