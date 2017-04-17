# lstm_crf


基于tensorflow，使用lstm神经网络和crf进行序列标注。


训练模型: python3 train

测试: python3 test

也可以更改main函数中的train，test进行训练和测试。


原始数据：data/data.data

模型: model_path

输出数据: output/output.data

日志: loss_log

配置文件: config.config


训练数据格式：
可	O

疑	O

碘	E

过	O

敏	O


无	O

食	O

物	O

过	O

敏	O

史	O


有	O

药	O

物	O

、	O

食	O

物	O

过	O

敏	O

史	O

：	O

青	B

霉	I

素	I

类	I

药	I

物	E

过	O

敏	O


对	O

鱼	O

虾	O

等	O

海	O

产	O

品	O

过	O

敏	O


对	O

青	B

霉	I

素	E

及	O

阿	O

司	O

匹	O

林	O

等	O

药	O

物	O

过	O

敏	O


发	O

现	O

心	O

房	O

纤	O

颤	O

1	O

个	O

月	O

、	O

食	O

物	O

过	O

敏	O

史	O


有	O

“	O

磺	B

胺	I

类	I

药	I

物	E

”	O

过	O

敏	O

史	O


...


每句以空行分割，字与标签用\t分割.

输出结果格式为：

可疑碘过敏，无食物过敏史<@>OOEOOOOOOOOO<@>碘

有药物、食物过敏史：青霉素类药物过敏<@>OOOOOOOOOOBIIIIEOO<@>青霉素类药物

对鱼虾等海产品过敏<@>OOOOOOOOO<@>

对青霉素及阿司匹林等药物过敏<@>OBIEOOOOOOOOOO<@>青霉素

发现心房纤颤1个月、食物过敏史<@>OOOOOOOOOOOOOOO<@>

有“磺胺类药物”过敏史<@>OOBIIIEOOOO<@>磺胺类药物

对“磺胺类药物”过敏<@>OOBIIIEOOO<@>磺胺类药物

有青霉素及头孢类抗生素过敏史<@>OBIEOBIIIIEOOO<@>青霉素*&*头孢类抗生素

对磺胺类药物过敏；否认食物过敏史<@>OBIIIEOOOOOOOOOO<@>磺胺类药物

对青霉素注射液过敏<@>OBIIIBEOO<@>射液

对青霉素及茴香过敏<@>OBIEOOOOO<@>青霉素

青霉素皮试阳性；否认食物过敏史<@>BIEOOOOOOOOOOOO<@>青霉素

“磺胺类药物”过敏史，无食物过敏史<@>OBIIIEOOOOOOOOOOO<@>磺胺类药物

芒果过敏<@>OOOO<@>

药物应用对“黄连素”过敏<@>OOOOOBIIEOOO<@>“黄连素

有海鲜、牛奶、茶水、青霉素过敏史<@>OOOOOOOOOOBIEOOO<@>青霉素

对紫外线过敏<@>OOOOOO<@>

既往有青霉素类药物过敏史<@>OOOBIIIIEOOO<@>青霉素类药物

刺五加药物过敏史<@>BIIIEOOO<@>刺五加药物

对青霉素类、磺胺类、头孢类抗生素过敏<@>OBIIEOBIEOBIIIIEOO<@>青霉素类*&*磺胺类*&*头孢类抗生素


输出格式为：str1<@>str2<@>str3
str1： 原始数据
str2: 数据标签
str3：识别关键词，关键词以*&*分割.
