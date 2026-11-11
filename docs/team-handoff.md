# Part 1 与 Part 2 交接

Wenhan 负责 NYU 米制深度和 VDA 视频深度。Yongru 负责 Middlebury 双目结果
和逐帧 DA V2。双目和 NYU 的指标各自进入报告。过山车视频用同一源帧拼成
RGB、DA V2、VDA 三列。

## Yongru 的提交

| 内容 | 放在哪里 |
|---|---|
| 双目方法、参数和观察 | 新增 `docs/part1.md` |
| 双目结果 | `results/stereo/per_scene.csv`、`results/stereo/metrics.json` 和有代表性的视差、无效区域、深度图 |
| 逐帧 DA V2 推理 | 可沿用 `scripts/video.py` 中的 DA V2 路径，或提交自己的 Part 1 运行入口和参数说明 |
| 逐帧深度交接 | 本地的 `results/video/dav2_depths.npz` 及同名 `.json` |

现有 `scripts/stereo.py` 是共享参考。提交时记录自己实际运行的参数与结果。
Wenhan 的 NYU 指标在 `results/nyu/baseline/`，VDA 交接数组为
`results/video/vda_depths.npz` 及同名 `.json`。

可以把下面的话交给 Yongru 的 Codex：

> 从含交接说明的最新版分支开始，在自己的 Part 1 分支上工作。按这页的目录
> 提交 Middlebury 指标和代表性图，并提供逐帧 DA V2 的复现方式。两套视频数组
> 齐备后，用 `scripts/compose_video.py` 生成三列对比并检查对齐。用 PR 合入结果。

协作版合入后以 `main` 为起点。数据集、模型权重、`.npy`、`.npz` 由本地
下载、推理或组内交换。PR 只提交代码、复现说明、轻量指标和选定的展示文件。

## 视频文件约定

两边使用同一份 RGB 源文件、原帧率和从 0 开始的连续帧。每个 NPZ 有一个
`depths` 键，存 `float32` 的 `(帧数, 高, 宽)` 数组。保存模型的原始相对深度，
保留完整画面及宽高比。当前官方文件的两组数组都是 `(70, 360, 640)`，帧率
为 24 FPS。若课程提供新视频，两边用该文件重新生成对应数组。

`scripts/video.py` 在生成新的 NPZ 时会附带同名 JSON，里面有
`source_sha256`、`fps`、`frame_indices`。使用自己的推理脚本时，确认
写入的元数据对应实际读取的源文件和帧序。可在仓库根目录调用同一个写入函数：

```python
from pathlib import Path
from depth_project.video_render import source_sha256, write_depth_metadata

source = Path("third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4")
write_depth_metadata(Path("results/video/dav2_depths.npz"), source_sha256(source), fps=24.0, count=70)
```

例子里的 FPS 和帧数是当前源视频的实测值。新视频用本次实际推理的值。
元数据记录生成者使用的帧序。合成器核对这个记录和源视频，因此生成数组时
要保持帧序与记录一致。

## 合成三列视频

从仓库根目录运行。已有原始 RGB 和两份数组时，合成只用 CPU。

```bash
python scripts/compose_video.py --source third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4 --dav2 results/video/dav2_depths.npz --vda results/video/vda_depths.npz --output tmp/rgb_depth_comparison.mp4
```

两份 JSON 与各自 NPZ 放在一起。合成器检查源文件指纹、帧率、帧编号、
数组形状和数值有效性，并对输出视频重新解码，确认帧数和尺寸。
两个模型使用同一调色板，每个模型各自固定整段的数值范围。
查看栏杆、车厢和遮挡处在连续帧中的表现。

之前生成的缓存可能只有 NPZ。确认它们使用同一源视频、原帧率与相同的连续帧后，
可在命令末尾加 `--allow-legacy`。程序仍检查形状和解码结果，输出会标记缺少
可核验元数据的输入。推荐交换新结果时一起带上 JSON。

现有 [三列示例](../results/video/rgb_depth_comparison.mp4) 可用于第一次对齐时参考。
合成输出先放在 `tmp/`，双方看过成片后再选用到报告与结果包。
