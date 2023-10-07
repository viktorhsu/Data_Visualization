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

11、箱线图

12、热图：（一个二维图，直接将二维表中的数据值，用颜色深浅来表示，每一个数据值用一个颜色小块来表示，整个图即为由许多不同深浅的颜色构成的二维表）

13、词云图

14、网络图（展示各元素之间的联系）

15、核密度图（展现样本概率分布情况）
