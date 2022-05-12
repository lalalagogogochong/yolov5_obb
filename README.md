
# Yolov5 for Oriented Object Detection

## Usage

```shell
原版yolov5_obb,更改kldloss之后运行相应文件即可detect_kld
CUDA_VISIBLE_DEVICES=1 python detect.py \
    --source ./images/input \
    --weights ./models/weights/best.pt \
    --conf 0.4 \
    --iou-thres 0.6 \
    --img-size 640 \
    --output ./images/output \
    --augment
```


#### 参数说明

```shell
--source 输入图片路径
--output 测试结果保存路径
--weights 训练好的模型权重路径
--conf 置信度阈值
--iou-thres iou阈值
--img-size 输入图片尺寸
--output 模型输出结果保存路径
--augment 多尺度测试
```

