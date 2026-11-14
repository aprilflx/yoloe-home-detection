#test_yoloe.py
from ultralytics import YOLOE

# 加载 YOLOE 预训练模型（会自动下载权重，第一次稍慢）
model = YOLOE("yoloe-11s-seg.pt")

# 用文本提示指定想检测的物体
names = ["cup", "book", "toy", "bottle","chair","table"]
#names = ["books"]
model.set_classes(names, model.get_text_pe(names))

# 对图片做推理
#results = model.predict("D:/myyoloe/myimages/ertongyizi.jpg", save=True)
results = model.predict("D:/myyoloe/myimages", save=True)

# 打印检测到的类别和置信度
for r in results:
    for box in r.boxes:
        print(names[int(box.cls)], round(float(box.conf), 3))
