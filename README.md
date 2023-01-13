# 目标检测工具包

## DETR

使用 `torch.hub` 中提供的 `detr_resnet50` 模型

```python
detr = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
```

**结果图:**

![DETR](./figures/detr_result.png)