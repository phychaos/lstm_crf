# lstm_crf

基于tensorflow，使用lstm神经网络和crf进行序列标注。


### 训练模型
>python3 train
>
### 测试
>python3 test
>
>
>也可以更改main函数中的train，test进行训练和测试。
>
### 数据配置路径
>原始数据：data/data.data
>
>模型: model_path
>
>输出数据: output/output.data
>
>日志: loss_log
>
>配置文件: config.config
>
>
### 训练数据
>
>迈	B
>
>向	E
>
>充	B
>
>满	E
>
>希	B
>
>望	E
>
>的	S
>
>新	S
>
>世	B
>
>纪	E
>
>—	B
>
>—	E
>
>一	B
>
>九	M
>
>九	M
>
>八	M
>
>年	E
>
>新	B
>
>年	E
>
>讲	B
>
>话	E
>
>（	S
>
>附	S
>
>图	B
>
>片	E
>
>１	S
>
>张	S
>
>）	S
>
>
>每句以空行分割，字与标签用\t分割.
>
### 输出结果
>
>既往青霉素、链霉素、磺胺类药物过敏史<@>既_往_青霉素_、_链_霉素_、_磺_胺类_药物_过敏史
>
>对“鸡蛋”等多种食物过敏<@>对_“_鸡蛋_”_等_多种_食物_过敏
>
>对降脂药“非诺贝特”过敏<@>对_降脂_药_“_非诺贝特_”_过敏
>
>有“青霉素”过敏史、食物过敏史<@>有_“_青霉素_”_过敏史_、_食物_过敏史
>
>对磺胺类、青霉素类及巴比妥类药物过敏<@>对_磺胺类_、_青霉_素类_及_巴比_妥类_药物_过敏
>