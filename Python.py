1、柱状图：bar
import matplotlib.pyplot as plt
# 样本数据
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [25, 40, 30, 50]
# 创建柱状图，第一个参数是柱子的标签或类别，第二个参数是柱子的高度或值。
plt.bar(categories, values, color='skyblue')
# 添加标题和标签
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')
# 显示图表
plt.show()



2、直方图：hist
import matplotlib.pyplot as plt
import numpy as np
# 生成随机样本数据
data = np.random.randn(1000)
# 创建直方图
plt.hist(data, bins=20, color='skyblue', edgecolor='black')
# 添加标题和标签
plt.title('Histogram Example')
plt.xlabel('Values')
plt.ylabel('Frequency')
# 显示图表
plt.show()




3、折线图：plot
import matplotlib.pyplot as plt
# 样本数据
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 15, 11]
# 创建折线图, label表示这一系列数据的名称，用来添加图例时用
plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Data')
# 添加标题和标签
plt.title('Line Chart Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# 添加图例
plt.legend()
# 显示图表
plt.show()




4、面积图：fill_between
#面积图：（累积值为纵坐标，以累积值构成的折线图作为图的上边缘，通过面积来表示数值的大小）
import matplotlib.pyplot as plt

# 样本数据
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 15, 11]

# 创建面积图
plt.fill_between(x, y, color='skyblue', alpha=0.6)

# 添加标题和标签
plt.title('Area Chart Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图表
plt.show()




5、饼图
import matplotlib.pyplot as plt

# 样本数据
labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [25, 40, 30, 50]
# 创建饼图，autopct 是 plt.pie() 函数的一个参数，用于控制百分比值的显示方式。
#%1.1f：这部分是一个浮点数格式化指令，表示显示一个浮点数，保留一位小数。
#%%：这是一个转义字符，用于显示百分比符号 %。
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'])
# 添加标题
plt.title('Pie Chart Example')
# 显示图表
plt.show()




6、环形图：先绘出一个饼图，然后再绘出一个空白的圆，将空白的圆添加到饼图中
import matplotlib.pyplot as plt

# 样本数据
labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [25, 40, 30, 50]

# 创建环形图，plt.subplots()为绘制子图。此处为将全局图命名为fig，其中一个子图命名为ax。创建子图不是必须的。
#下方代码创建的为单样本集的环形图，非复合环形图。
# wedgeprops={'width': 0.4}表示内圆的直径为外圆直径的40%
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'],
       wedgeprops={'width': 0.4})

# 添加中心圆（可选)，plt.Circle((0, 0), 0.2, color='white')表示创建一个圆心为原点，直径为0.2，颜色为白色的圆
center_circle = plt.Circle((0, 0), 0.2, color='white')
ax.add_artist(center_circle) #表示将创建的白色圆添加到子图中

# 添加标题
plt.title('Donut Chart Example')

# 显示图表
plt.show()




7、散点图：scatter
import matplotlib.pyplot as plt

# 样本数据
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 15, 11]

# 创建散点图
plt.scatter(x, y, c='blue', marker='o', label='Data')

# 添加标题和标签
plt.title('Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 添加图例
plt.legend()

# 显示图表
plt.show()




8、气泡图：plot
#即直接对三维数据绘制散点图即可
#edgecolors='black' 和 linewidths=1.0：这两个参数用于设置散点边界的颜色和线宽，使气泡图的边界更加清晰。

import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 15, 11]
bubble_size = [100, 200, 300, 400, 500]  # 气泡的大小，代表第三个维度的数据

# 创建气泡图
plt.scatter(x, y, s=bubble_size, c='blue', alpha=0.6, edgecolors='black', linewidths=1.0)

# 添加标题和标签
plt.title('Bubble Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图表
plt.show()



9、散点矩阵图：sns.pairplot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 创建示例数据（DataFrame）
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 12, 8, 15, 11],
    'C': [20, 18, 25, 22, 27],
    'D': [5, 10, 15, 20, 25]
})

# 绘制散点矩阵图
sns.pairplot(data, 
             palette='Set1',  # 设置颜色调色板
             markers=['o', 's'],  # 设置不同的标记样式
             diag_kind='kde',  # 对角线上的图形类型为核密度估计
             plot_kws={'alpha': 0.7, 's': 50},  # 透明度设置为0.7， 圆点的大小设置为50，值越大则圆点越大。
             diag_kws={'color': 'purple'})  # 传递对角线图的绘图参数

# 显示图表
plt.show()




10、雷达图
import numpy as np
import matplotlib.pyplot as plt

# 创建示例数据
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
data = [4, 3, 2, 5, 4, 4]  # 请注意，data 列表包含6个元素，这是因为雷达图是一个封闭的形状，需要将最后一个值重复一次，以使图形封闭。

# 计算角度
num_categories = len(categories)
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
#np.linspace() 函数生成了一个从0到2π（一圈的角度范围）之间的均匀分布的 num_categories 个角度
#endpoint=False 表示不包括终点值2π，以确保最后一个角度与第一个角度不重叠。
#.tolist()：将生成的角度数组转换为Python列表，以便后续使用。
angles += angles[:1]  #通过将 angles[:1]（第一个元素）添加到 angles 列表的末尾来实现角度列表闭合，从而闭合图形

# 创建一个雷达图
plt.figure(figsize=(6, 6)) #该函数每被调用一次，即会创建一个图形窗口。如果不调用的话，会创建一个默认尺寸的图形窗口。
ax = plt.subplot(111, polar=True)  
#plt.subplot(111)表示创建一个1*1的子图网格，并且子图位于第1个位置，等价于plt.subplot(1, 1, 1.）
#polar=True表示为创建一个极坐标图，此时y轴为极径轴（体现为圆环），x轴为角度轴（体现为躯干）

# 绘制雷达图
plt.xticks(angles[:-1], categories, color='grey', size=12)  
#plt.xticks() 是Matplotlib库中用于设置x轴刻度标签的函数。
#x轴为骨架，y轴为圆环
plt.yticks(np.arange(1, 6), color='grey', size=12)  # 设置极径刻度

ax.fill(angles, data, 'b', alpha=0.1)  # 在各类别的值围成的区域，填充指定的颜色
ax.set_ylim(0, 5)  # 用于设置 y 轴（极径轴）的取值范围，一般为数据值中的最大值和最小值

# 添加标题
plt.title('Radar Chart Example', size=16, color='black', y=1.1)

# 显示雷达图
plt.show()

11、箱线图：plt.boxplot
import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
data = [np.random.normal(0, 1, 100),  # 第一个数据集
        np.random.normal(1, 1, 100),  # 第二个数据集
        np.random.normal(2, 1, 100)]  # 第三个数据集

# 绘制箱线图
plt.boxplot(data,            # 数据集列表
            labels=['A', 'B', 'C'],  # 箱线图的标签
            notch=True,      # 是否显示缺口
            vert=True,       # 竖直方向绘制箱线图
            patch_artist=True,  # 是否填充箱体颜色
            showmeans=True,  # 是否显示均值
            meanline=True,   # 是否在箱线图上显示均值线
            whis=1.5)        # 离群值识别范围

# 添加标题和标签
plt.title('Box Plot Example')
plt.xlabel('Groups')
plt.ylabel('Values')

# 显示箱线图
plt.show()





12、词云图：需要先创建一个WordCloud实例，然后再通过imshow()方法
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 示例文本数据
text = "Python is a popular programming language. It is used for web development, data analysis, machine learning, and more."

# 创建词云对象，返回的对象是一个 WordCloud 类的实例
wordcloud = WordCloud(width=800, height=400,
                      background_color='white',  # 背景颜色
                      colormap='viridis',          # 颜色地图
                      stopwords=['is', 'a', 'it', 'and'],  # 停用词
                      max_words=50,                # 最大显示的单词数量
                      min_font_size=10,            # 最小字体大小
                      contour_width=2,             # 边界线宽度
                      contour_color='steelblue')   # 边界线颜色

# 生成词云图
wordcloud.generate(text)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
#显示图像数据、热图、矩阵等二维数据时，需要使用imshow()方法
#因为词云图与热图、二维数据的可视化类似，都是以视觉的方式来展示数据的分布或重要性，因此也使用imshow()方法
plt.axis('off')  # 不显示坐标轴
plt.title('Word Cloud Example')
plt.show()




13、热图：imshow()
#一个二维图，直接将二维表中的数据值，用颜色深浅来表示，每一个数据值用一个颜色小块来表示，整个图即为由许多不同深浅的颜色构成的二维表
import matplotlib.pyplot as plt
import numpy as np

# 创建一个示例的二维数据矩阵
data = np.random.rand(5, 5)  # 生成一个随机的5x5矩阵，代表数据分布

# 绘制热图
plt.imshow(data, cmap='coolwarm', interpolation='nearest')
#cmap：颜色映射参数，用于指定颜色的映射方式。'coolwarm' 是一个常用的颜色映射，数据值低的区域为冷色，数据值高的为暖色
plt.colorbar()  # 添加颜色条

# 设置轴标签
plt.xticks(np.arange(5), ['A', 'B', 'C', 'D', 'E'])
plt.yticks(np.arange(5), ['1', '2', '3', '4', '5'])

# 添加标题
plt.title('Heatmap Example')

# 显示图像
plt.show()




14、核密度图（展现样本概率分布情况）
import seaborn as sns
import matplotlib.pyplot as plt

# 创建示例数据（正态分布）
#"tips"为seaborn库中内置的示例数据集，load_dataset()为加载该类数据集的函数
data = sns.load_dataset("tips")

# 绘制核密度图
#表示是否在核密度估计曲线下方填充阴影，以增加可读性。
sns.kdeplot(data=data["total_bill"], shade=True, color="blue", label="Total Bill")

# 添加标题和标签
plt.title("Kernel Density Plot Example")
plt.xlabel("Total Bill")
plt.ylabel("Density")

# 显示图形
plt.show()




15、网络图:
#展示各元素之间的联系


