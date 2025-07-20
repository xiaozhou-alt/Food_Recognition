# 【AI入门系列】美食侦探：食物声音识别学习赛



<table><tr><td>赛题与数据</td><td>文档</td><td>大小</td><td>操作</td><td>ossutil命令</td></tr><tr><td>排行榜</td><td>test_a.zip</td><td>zip(1GB)</td><td>下载</td><td>复制命令</td></tr><tr><td>代码规范</td><td>test_b.zip</td><td>zip(1GB)</td><td>下载</td><td>复制命令</td></tr><tr><td>学习建议</td><td>train_sample.zip</td><td>zip(515MB)</td><td>下载</td><td>复制命令</td></tr><tr><td>获奖名单</td><td>train.zip</td><td>zip(3GB)</td><td>下载</td><td>复制命令</td></tr></table>

# 赛题描述及数据说明



数据集来自Eating Sound Collection，数据集中包含20种不同食物的咀嚼声音，赛题任务是给这些声音数据建模，准确分类。作为零星音识别的新大赛，本次任务不涉及复杂的声音模型、语言模型，希望大家通过两种baseline的学习能体验到语音识别的乐趣。

- train文件夹：完整的训练集；
- train_sample文件夹：部分训练集；
- test文件夹：测试集；目前由于天池实验室存储限制，在DSW上参与本场比赛建议使用压缩后的数据集

# 赛题包含的类别：

aloe ice- cream ribs chocolate cabbage candied_fruits soup jelly grapes pizza gummies salmon wings burger pickles carrots fries chips noodles drinks

# 评估标准

赛题使用准确率（Accuracy）衡量选手结果与真实标签的差异性。

# 结果提交

提交前请确保预测结果的格式与sample_submit.csv中的格式一致，以及提交文件后缀名为csv。

注意事项：

- 第一列为语音文件名称，第二列为类别；- 提交语音文件顺序不作要求；

# 提交格式样例：

name,Table1 

DP2R8P7KJK.wav.cabbage 

3ITH5DYEUI.wav.grapes 

09119CHZBZ.wav.noodles 

AH2AXWMRAP.wav.ice- cream 

v9A67TZ9RF.wav.carrots 

C096FBDJ2L.wav.chips

# 比赛规则

为了比赛公平公正，所有参赛选手不允许使用任何外部数据集。同时所有参赛选手不允许使用任何非公开的预训练模型，公开的预（如ImageNet和COCO）可以使用。为了比赛趣味性，不建议选手使用伪标签操作，同时建议选手保存好代码，最终比赛程序代码需要完整复现。
