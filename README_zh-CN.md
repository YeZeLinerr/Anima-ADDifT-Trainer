# Anima ADDifT Trainer

[English](README.md)

这是一个只用于 Anima ADDifT 配对图片 LoRA 训练的独立工具。

训练后端使用从 Anima Standalone Trainer 最近未提交更改中提取的
`paired_difference_mode` 实现。WebUI 采用专用的 Source → Target 布局，
不包含无关的训练模式。

## 效果展示

下面是训练使用的四组配对图片。左侧为 Source（睁眼），右侧为 Target
（闭眼）。

<img src="images/source_target_pairs.png" alt="Source and Target training pairs" width="600">

使用上述四组图片训练 100 步后，LoRA 已能在未参与训练的角色上生成闭眼
效果：

<img src="images/output.png" alt="Result after 100 training steps" width="420">

所有展示图片均已移除 PNG 工作流、提示词及其他元数据。

## 已知问题

- 训练闭眼 LoRA 会导致闭眼线条颜色变淡。生成的眼皮或睫毛线条可能比
  Target 图片中对应的深色硬线更浅。
- 如果在每个 Target 角色的鼻子上添加一个硬边圆，训练得到的 LoRA 往往
  会生成带羽化效果的软边圆，无法完整保留 Target 中的硬边。

## 启动

便携环境：

```text
start_training_ui_portable.bat
```

项目虚拟环境或系统 Python：

```text
start_training_ui.bat
./start_training_ui.sh
```

打开 `http://127.0.0.1:3001`。

## 数据方向

```text
Source 图片（变化前） → Target 图片（变化后）
```

两侧图片必须使用相同文件名 stem：

```text
source/001.png
target/001.png
```

Caption 放在 Target 图片旁边，例如 `target/001.txt`。

正倍率应用 Target 中的效果，负倍率移除该效果。训练后端会自动交替正反
方向，不需要额外开启 Reverse Pair。

## UI 参数

- **Slider Scale：**训练时使用的正负 LoRA 倍率。
- **Timestep：**局部装饰或结构变化建议 `500-1000`；颜色或风格变化可从
  `200-400` 开始。
- **Soft Difference Mask：**强调 Source/Target 真正发生变化的 latent
  区域。
- **Mask Area Normalize：**避免小面积编辑的 loss 被整张图片平均稀释。
- **Background Weight：**未变化区域保留的最低监督权重。

## 主要文件

- `anima_train_addift.py`：强制启用 ADDifT 的专用训练入口。
- `tools/anima_addift_webui.py`：单页 UI、图片配对检查和训练进程管理。
- `anima_train_network.py`：ADDifT 预测匹配训练实现。
- `train_network.py`：共享训练循环。

详细算法说明见 `PAIRED_DIFFERENCE_TRAINING.md`。

## 开源协议

本项目自行编写的新增代码使用 [MIT License](LICENSE) 开源。继承或修改自
上游项目的代码仍遵循 [Apache License 2.0](LICENSE-APACHE-2.0.md)。
归属信息见 [NOTICE](NOTICE)。

## 参考项目

- [tukisuwa/sd-scripts](https://github.com/tukisuwa/sd-scripts)
- [hako-mikan/sd-webui-traintrain](https://github.com/hako-mikan/sd-webui-traintrain)
- [gazingstars123/Anima-Standalone-Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer)
