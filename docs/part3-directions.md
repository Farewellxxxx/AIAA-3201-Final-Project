# Part 3 拓展方向

课程 Project 3 PDF 第 5.3 节列出六个备选方向。小组选一种轻量改进，与原始方法对比。

| 方向 | 作业中的拓展思路 |
|---|---|
| A · Flow-Guided Fusion | 用 RAFT 或 Farnebäck 光流把上一帧深度对齐到当前帧。用前后向一致性筛除遮挡，再按置信度融合。 |
| B · Metric Scale Calibration | 在 NYU 训练或开发集拟合一个全局尺度，固定后用于测试集，分析效果与泛化范围。 |
| C · Edge-Aware Refinement | 用 RGB 梯度、语义边界或分割掩膜细化深度轮廓，检查是否把图像纹理带进了深度图。 |
| D · Uncertainty-Aware Smoothing | 用模型置信度、测试时增强或模型间差异估计不确定性，有选择地平滑不稳定区域。 |
| E · Efficient Deployment | 试用 ONNX、Core ML、量化或动态分辨率，比较速度、内存和准确率。 |
| F · Depth-Aware Application | 改善背景虚化或物体遮挡排序等效果，和使用原始深度的同一应用做对照。 |
