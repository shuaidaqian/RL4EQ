"""Matplotlib 中文显示配置模块。
在其他脚本中通过 from utils.matplotlib_zh import * 导入即可使用中文。

用法：
1. 在绘图脚本开头 import
2. 绘图时直接使用中文，无需额外设置
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 全局设置中文字体
_font = fm.FontProperties(family="SimHei")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
plt.rcParams["font.size"] = 10


def get_font(size: int = 10):
    """获取指定大小的 SimHei 字体属性对象，用于 title/label/suptitle 等。"""
    return fm.FontProperties(family="SimHei", size=size)
