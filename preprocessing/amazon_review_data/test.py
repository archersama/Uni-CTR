import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator

# 创建场景字典
scenario_dict = {
    0: 'Amazon Fashion',
    1: 'Digital Music',
    2: 'Musical Instruments',
    3: 'Gift Cards',
    4: 'All Beauty'
}

# 加载数据
def load_data(file_path):
    return pd.read_csv(
        file_path,
        # sep='\t'
    )

# 计算字符串长度
def calculate_content_length(data):
    data['content_length'] = data['content'].apply(len)
    return data


def plot_combined_distribution_3d(data, scenarios, colors, save_path, x_min, x_max):

    n = 3
    fig = plt.figure(figsize=(n * 4, n * 2))
    ax = fig.add_subplot(111, projection='3d')

    scale_x = 2
    scale_y = 2
    scale_z = 1

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

    # 设置图表的视角
    ax.view_init(elev=20, azim=-35)

    # 仅选择落在新的x轴范围内的数据点
    data = data[(data['content_length'] >= x_min) & (data['content_length'] <= x_max)]
    bins = np.linspace(x_min, x_max, 50)  # 重新计算bins以匹配新的x_min
    bin_width = bins[1] - bins[0]  # 计算bins宽度

    # 计算直方图数据
    hist_data = [np.histogram(data[data['scenario'] == scenario]['content_length'], bins=bins) for scenario in scenarios]

    # 设置每个场景的柱状图的位置
    xpos = bins[:-1] + bin_width / 2  # 中心对齐
    ypos = np.arange(len(scenarios))
    xpos, ypos = np.meshgrid(xpos, ypos)

    xpos = xpos.flatten()
    ypos = ypos.flatten() * 2   # 控制场景之间的间隔
    zpos = np.zeros_like(xpos)

    # 设置柱状图的宽度和高度
    dx = bin_width * 0.5  # 柱体的宽度稍微小于bin宽度
    dy = 0.2  # 每个场景的宽度

    # 绘制柱状图
    for idx, (counts, _) in enumerate(hist_data):
        ax.bar3d(xpos[idx * len(bins[:-1]):(idx + 1) * len(bins[:-1])],
                 ypos[idx * len(bins[:-1]):(idx + 1) * len(bins[:-1])]*2,
                 zpos[idx * len(bins[:-1]):(idx + 1) * len(bins[:-1])],
                 dx, dy, counts, color=colors[idx], zsort='max')

    # 设置图表的标题和坐标轴标签
    ax.set_xlabel('Content Length')
    # 设置标签水平和位置
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Frequency', labelpad=40, rotation=0)

    # 设置y轴的标签为场景名称
    ax.set_yticks(ypos[::len(bins[:-1])] + dy / 2)
    ax.set_yticklabels([scenario_dict[scenario] for scenario in scenarios], va='center', ha='left')

    # 设置x轴的显示范围
    ax.set_xlim([x_min - bin_width / 2, x_max + bin_width / 2])

    # 设置z轴的刻度位置
    ax.zaxis.set_major_locator(FixedLocator(ax.get_zticks()))
    ax.set_zticklabels([f'{tick:.1e}' for tick in ax.get_zticks()], va='center', ha='left')

    # 保存并显示图像
    plt.savefig(save_path, format='pdf')
    plt.show()

# 主函数，整合以上步骤
def main(file_path, save_path, x_min, x_max):
    data = load_data(file_path)
    data = calculate_content_length(data)
    scenarios = [3, 1, 4, 0, 2]  # 按照场景的顺序指定颜色
    colors = ['#1f77b4',  # 深蓝色
              '#ff7f0e',  # 橙色
              '#2ca02c',  # 绿色
              '#d62728',  # 红色
              '#9467bd']  # 紫色

    plot_combined_distribution_3d(data, scenarios, colors, save_path, x_min, x_max)

# 设置文件路径和保存路径
file_path = "../../datasets/amazon_review_data/hybrid_data/hybrid_5_title_10000.csv"  # 这里填写您数据集的路径
save_path = 'path_to_save_plot.pdf'  # 这里填写您希望保存图表的路径

# 设置x轴的范围
x_min = 0
x_max = 1000

# 运行主函数
main(file_path, save_path, x_min, x_max)
