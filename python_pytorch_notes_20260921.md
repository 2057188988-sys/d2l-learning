# Python + PyTorch 学习笔记（2026-09-21）

本次学习围绕「张量操作 → Python 切片 → NumPy → CSV → 路径处理 → 环境优化」这条线，从 PyTorch 报错出发，一路补到工程实践。

---

## 一、PyTorch 张量操作

### 1. `reshape`：形状的乘积必须等于元素总数

```python
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))   # ✅ 3×4 = 12
X = torch.arange(12, dtype=torch.float32).reshape((2, 4))   # ❌ 2×4 = 8 ≠ 12
```

报错信息：

```
RuntimeError: shape '[2, 4]' is invalid for input of size 12
```

**规律**：`reshape` 各维相乘必须等于元素总个数，多一个少一个都不行。

| 写法 | 乘积 | 结果 |
|------|------|------|
| `torch.arange(12).reshape(3, 4)` | 12 | ✅ |
| `torch.arange(12).reshape(2, 6)` | 12 | ✅ |
| `torch.arange(12).reshape(2, 4)` | 8 | ❌ |
| `torch.arange(8).reshape(2, 4)` | 8 | ✅ |

### 2. `torch.cat`：只有拼接的那一维可以不同

```python
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])   # (3, 4)

torch.cat((X, Y), dim=0)   # 结果 (6, 4)  行数相加
torch.cat((X, Y), dim=1)   # 结果 (3, 8)  列数相加
```

**核心规则**：**除了拼接维度，其余所有维度必须严格相等。**

反例（会报错）：

```python
X = torch.arange(12, dtype=torch.float32).reshape((2, 6))   # (2, 6)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # (3, 4)
torch.cat((X, Y), dim=0)   # ❌ 第 1 维 6 vs 4 不匹配
```

```
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 6 but got size 4 for tensor number 1 in the list.
```

| 方向 | 要求 | X(3,4) + Y(3,4) 结果 |
|------|------|----------------------|
| `dim=0` | 第 1 维必须相同 | (6, 4) |
| `dim=1` | 第 0 维必须相同 | (3, 8) |

### 3. 原地操作与 `id()`

```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))     # 地址 A
Z[:] = X + Y
print('id(Z):', id(Z))     # 还是地址 A
```

- `torch.zeros_like(Y)`：创建形状/dtype/device 与 `Y` 一致、值全为 0 的**新张量**
- `id()`：返回对象的**内存地址标识**
- `Z[:] = X + Y`：**原地写入**，把 `X+Y` 的数值拷进 `Z` 原有内存，**地址不变**

| 写法 | 地址是否改变 | 说明 |
|------|-------------|------|
| `Z = X + Y` | ✅ 变 | 变量重新绑定到新张量 |
| `Z[:] = X + Y` | ❌ 不变 | 原地写入，复用原内存 |

**注意**：`Z[:] = X + Y` 中间那个临时张量**还是被创建了**。它省的是「Z 的重复分配」，不是「完全不产生临时内存」。

---

## 二、Python 切片（slice）语法

### 1. 基本形式

```
obj[start:stop:step]
```

| 部分 | 含义 | 默认值 |
|------|------|--------|
| `start` | 起始索引（**包含**） | 0（正步长）/ -1（负步长） |
| `stop` | 结束索引（**不包含**） | len(obj)（正步长）/ 到开头（负步长） |
| `step` | 步长，负数表示倒序 | 1 |

```python
a = [0, 1, 2, 3, 4, 5]

a[1:4]      # [1, 2, 3]      含头不含尾
a[:3]       # [0, 1, 2]
a[3:]       # [3, 4, 5]
a[:]        # [0, 1, 2, 3, 4, 5]
a[::2]      # [0, 2, 4]
a[1::2]     # [1, 3, 5]
```

**索引示意图**

```
正向:   0    1    2    3    4    5
值:     0    1    2    3    4    5
负向:  -6   -5   -4   -3   -2   -1
```

### 2. 负索引与倒序

```python
a[-1]        # 5
a[-3:]       # [3, 4, 5]
a[:-2]       # [0, 1, 2, 3]

a[::-1]      # [5, 4, 3, 2, 1, 0]   整个反转
a[4:1:-1]    # [4, 3, 2]
```

**注意**：负步长时 `start` 要**大于** `stop`，否则得到空列表。

### 3. 越界不报错（重要特性）

```python
a[2:100]     # [2, 3, 4, 5]   stop 超界自动截断
a[100:]      # []             start 超界返回空
a[2]         # 单元素索引超界则报 IndexError
```

**切片容错，单元素索引不容错。**

### 4. 切片赋值

```python
a[:] = [9, 9]            # 切片赋值（原地修改）
b = a[:]                 # 浅拷贝一份新列表
a[1:3] = [10, 20, 30]    # 可以改变长度
```

### 5. `[:]` 是原地操作的标志吗？

**不是。** `[:]` 只是「取全部元素的切片」，是否原地取决于它出现在等号哪一边。

| 写法 | 是否原地 | 原因 |
|------|---------|------|
| `Z = expr` | ❌ | 变量名重新绑定 |
| `Z[:] = expr` | ✅ | 触发 `__setitem__`，写入原对象内存 |
| `Z[0] = expr` | ✅ | 同样是 `__setitem__` |
| `Z.add_(Y)` | ✅ | 方法名带下划线 |

**结论**：
- **Python 层面**：「索引/切片赋值」（`Z[:] = ...`）是原地
- **PyTorch 层面**：方法名带 `_`（`add_`、`relu_`、`mul_`）是原地
- `[:]` 单独出现不是原地标志，它只是「全部元素」的意思

### 6. 不只列表能用

字符串、元组、`range`、NumPy 数组、`torch.Tensor` 规则完全一致。

```python
s = "PyTorch"
s[0:2]      # "Py"
s[::-1]     # "hcroT yP"
```

---

## 三、NumPy 是什么

**一句话：NumPy 是 Python 做科学计算的基础库，核心是 `ndarray`（N 维数组）。**

### 为什么需要它

Python 原生 `list` 做数值计算又慢又啰嗦：

```python
# 纯 Python：要写循环
c = [a[i] + b[i] for i in range(len(a))]

# NumPy：直接相加，底层 C 实现，快几十倍
c = np.array([1, 2, 3]) + np.array([4, 5, 6])   # array([5, 7, 9])
```

这叫**向量化（vectorization）**——不写循环，整块数组一起算。

### 核心用法

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])   # 形状 (2, 3)

x.shape        # (2, 3)
x.dtype        # int64
x.ndim         # 2
x.T            # 转置 (3, 2)
x.sum(); x.mean()
x[0, :]        # 第 0 行
x[:, 1]        # 第 1 列

np.zeros((2, 3))
np.arange(12)
np.random.randn(2, 3)
```

### 与 PyTorch 的关系

| | NumPy | PyTorch |
|---|---|---|
| 核心类型 | `ndarray` | `Tensor` |
| API | `np.array`, `np.zeros` | `torch.tensor`, `torch.zeros` |
| 形状/切片/广播 | 完全一致 | 完全一致 |
| 自动求导 | ❌ 无 | ✅ 有（autograd） |
| 用 GPU | ❌ 只能 CPU | ✅ 支持 CUDA |

**PyTorch 基本就是把 NumPy 那套语法搬过来，再加上「自动求导 + GPU」。** 理解 NumPy 就等于理解了 PyTorch 数据操作的那一半。

互相转换：

```python
t = torch.from_numpy(x)    # ndarray → Tensor（共享内存）
n = t.numpy()              # Tensor → ndarray
```

---

## 四、CSV 文件

**一句话：CSV 是一个纯文本文件，用逗号把数据隔开，一行代表一条记录。**

（Comma-Separated Values，逗号分隔值）

```
姓名,数学,英语
张三,90,85
李四,78,92
```

用 Excel 打开就变成表格：

| 姓名 | 数学 | 英语 |
|---|---|---|
| 张三 | 90 | 85 |
| 李四 | 78 | 92 |

**逗号是竖线（分列），换行是横线（分行）。**

### 三条基本规则

1. **第一行通常是表头**（列名）
2. **值里有逗号 → 用双引号包起来**：`张三,"上海市,宝山区"`
3. **值里有双引号 → 写成两个**：`A,"他说""你好"""`

### 与 Excel 的区别

| | CSV | .xlsx |
|---|---|---|
| 是什么 | 纯文本 | 二进制压缩包 |
| 公式/颜色 | ❌ | ✅ |
| 多工作表 | ❌ | ✅ |
| 记事本能看 | ✅ | ❌ |
| 体积 | 小 | 大 |
| 适合 | 程序读取、传数据 | 人手工编辑 |

### 常见的坑

1. **中文乱码**——编码问题（UTF-8 vs GBK），最常见
2. **分隔符不一定是逗号**——有些地区用 `;`，用 Tab 的叫 TSV
3. **没有数据类型**——全是文字，`001` 的前导 0 可能丢
4. **空值/换行处理不一致**——跨软件传数据要留意

### Python 读写

```python
import pandas as pd

df = pd.read_csv("score.csv", encoding="utf-8")
df.to_csv("out.csv", index=False)
```

```python
import csv

with open("score.csv", encoding="utf-8") as f:
    for row in csv.reader(f):
        print(row)   # ['张三', '90', '85']
```

**与深度学习的关系**：很多数据集以 CSV 形式给出（图片路径 + 标签），`pd.read_csv` 读入后转成 numpy 数组或 Tensor 喂给模型，是最常见的数据入口。

---

## 五、路径处理

### 1. `os.path.join`

**把几段路径拼成完整路径，并自动用对当前系统的分隔符。**

```python
import os

os.path.join("data", "train", "a.csv")
# Windows → 'data\\train\\a.csv'
# Linux   → 'data/train/a.csv'
```

解决的问题：**Windows 用 `\`，Linux/Mac 用 `/`**。

**常见用法**

```python
# 与 __file__ 搭配（标准做法）
BASE = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE, "data", "score.csv")
```

**三个坑**

| 坑 | 例子 | 结果 |
|---|---|---|
| 遇到绝对路径会截断前面 | `os.path.join("data", "/etc/passwd")` | `/etc/passwd` |
| 空字符串参数被跳过 | `os.path.join("data", "", "a.csv")` | `data/a.csv` |
| 参数必须是 str | `os.path.join("data", 2024)` | ❌ TypeError |

### 2. 绝对路径 vs 相对路径

| | 例子 | 特点 |
|---|---|---|
| **绝对路径** | `C:\sites\pytorch\data\score.csv` | 从根开始，任何地方都能找到 |
| **相对路径** | `data/score.csv`、`../data/score.csv` | 相对于「当前所在位置」，换地方就找不到 |

**最大的坑：先分清两个概念**

| | 意思 | 获取方式 |
|---|---|---|
| **脚本在哪** | 这个 .py 文件本身的位置 | `__file__` |
| **我在哪** | 你敲命令时所在的目录（cwd） | `os.getcwd()` |

```python
project/
├── main.py      ← 里面写 open("data/score.csv")
└── data/
    └── score.csv
```

- 在 `project/` 下运行 `python main.py` → ✅ 找得到
- 在 `c:/sites/` 下运行 `python project/main.py` → ❌ FileNotFoundError

**Jupyter 更坑**：cwd 是你**启动 Jupyter 的那个目录**，不是 .ipynb 文件所在目录。

### 3. 拿到别人代码的路径排查清单

按报错概率排序：

| # | 问题 | 说明 |
|---|------|------|
| 1 | **硬编码绝对路径** | 搜 `C:\`、`/home/`、`/Users/`，搜到就要改 |
| 2 | **相对路径不知道相对谁** | 先 `print(os.getcwd())` |
| 3 | **反斜杠被当转义字符** | `"C:\Users\new"` 里 `\n`、`\t` 会坏掉，用 `r"..."` 或 `/` |
| 4 | **分隔符 `\` `/` 混用** | 建议统一用 `os.path.join` |
| 5 | **大小写敏感** | Windows 不管，Linux 严格区分 |
| 6 | **数据文件根本不存在** | 按报错路径去文件管理器确认 |
| 7 | **中文/空格路径** | 老库可能处理不了，项目路径建议全英文无空格 |
| 8 | **编码问题** | `encoding="utf-8"` 或 `"gbk"` |
| 9 | **`__file__` 在 Jupyter 里不存在** | 报 NameError，改用 `os.getcwd()` |
| 10 | **`~` 没被展开** | 用 `os.path.expanduser("~/data/a.csv")` |

**一分钟自查**

```python
import os
print("当前工作目录:", os.getcwd())
print("目录下有什么:", os.listdir("."))
print("这个文件在吗:", os.path.exists("data/score.csv"))
```

> **核心结论**：别人的代码出错，先看路径，再看数据文件在不在，最后才是代码逻辑。前两类占绝大多数。

### 4. 标准改造

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # 项目根目录
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "score.csv"

print(BASE_DIR)          # 先打印确认
df = pd.read_csv(CSV_PATH)
```

**为什么这样就稳**：`CSV_PATH` 完全靠 `main.py` 自身位置推出来，**和 cwd 彻底解耦**，在任何目录运行结果都一样。

Notebook 里改成：

```python
BASE_DIR = Path.cwd()   # 或手写 Path("c:/sites/pytorch/project")
```

### 5. 关键字逐个解释

假设文件是 `c:\sites\pytorch\project\main.py`：

```python
Path(__file__).resolve().parent
```

| 写法 | 返回 | 意思 |
|------|------|------|
| `__file__` | `str` | 脚本文件的路径（字符串），**和在哪运行无关** |
| `Path(__file__)` | `Path` | 包装成 Path 对象 |
| `.resolve()` | `Path` | 归一化成绝对路径，去掉 `..` |
| `.parent` | `Path` | 上一级目录（**属性，不加括号**） |
| `BASE / "data"` | `Path` | 拼接路径 |
| `Path.cwd()` | `Path` | 当前工作目录 |

逐层展开：

```python
Path(__file__)      # WindowsPath('C:/sites/pytorch/project/main.py')
  .resolve()        # WindowsPath('C:/sites/pytorch/project/main.py')   归一化
  .parent           # WindowsPath('C:/sites/pytorch/project')          取出所在文件夹
```

**老写法对照**：

```python
os.path.dirname(os.path.abspath(__file__))   # 等价于 Path(__file__).resolve().parent
```

学长代码里前者居多，见到不用慌。

### 6. Path 对象是什么

**对象 = 一份数据 + 一堆附带的「功能」。**

`Path` 对象就是：**一种专门表示文件路径的数据，并且自带一堆操作路径的功能。**

```python
from pathlib import Path

p = Path("c:/sites/pytorch/data/score.csv")
print(p)        # c:\sites\pytorch\data\score.csv
print(type(p))  # <class 'pathlib.WindowsPath'>
```

**看起来像字符串，但不是字符串。**

**类比：纸条 vs 导航仪**

| | 字符串 | Path 对象 |
|---|---|---|
| 本质 | 一张写着地址的**纸条** | 一个会干活的**路径管家** |
| 能干嘛 | 只能看、只能拼字符 | 可以问它各种问题 |

```python
p.name          # 'score.csv'
p.stem          # 'score'
p.suffix        # '.csv'
p.parent        # .../data
p.parents[1]    # 往上两级
p.exists()      # 存在吗
p.is_file()     # 是文件吗
p.absolute()    # 绝对路径
p.with_suffix(".txt")     # 换后缀
p.read_text(encoding="utf-8")
```

**常用功能速查**

```python
p.parts                    # ('c:\\', 'sites', 'pytorch', 'data', 'score.csv')
p.with_name("b.csv")       # 换文件名
p.mkdir(parents=True, exist_ok=True)   # 建目录
```

**三个注意点**

1. Windows 上是 `WindowsPath`，Linux 上是 `PosixPath`，用法一样，**自动适配分隔符**
2. **Path 对象 ≠ 文件真的存在**，要判断得显式调 `.exists()`
3. 传给老库可能要 `str(p)` 转换（报 `TypeError: expected str` 时就是这原因）

**`BASE_DIR / "data"` 里的 `/`**：这是 `Path` 重载的除法运算符，专用来拼路径，等价于 `os.path.join`，只是拼完还是 Path 对象。

| 对比 | `os.path.join` | `pathlib.Path` |
|---|---|---|
| 拼接 | `os.path.join(a, b)` | `Path(a) / b` |
| 取父目录 | `os.path.dirname(p)` | `p.parent` |
| 取文件名 | `os.path.basename(p)` | `p.name` |
| 是否存在 | `os.path.exists(p)` | `p.exists()` |
| 返回类型 | `str` | `Path` |

### 7. `resolve()` 的「干净唯一」

真实运行结果：

```
原始写法                                     resolve 之后
c:\sites\pytorch\data\..\data\a.csv    →    C:\sites\pytorch\data\a.csv
c:\sites\pytorch\data\a.csv            →    C:\sites\pytorch\data\a.csv
C:\SITES\pytorch\data\a.csv            →    C:\sites\pytorch\data\a.csv
```

| 词 | 意思 |
|---|---|
| **干净** | 没有 `..`、`.`、多余斜杠这些**绕路成分** |
| **唯一** | 一个文件只对应**一种规范写法**，各种写法都收敛到它 |

**验证**

```python
str(p1) == str(p2)            # False  ← 原始字符串不一样
p1.resolve() == p2.resolve()  # True   ← 规范化后一样
```

**`.resolve()` 具体做的事**

| 动作 | 例子 |
|---|---|
| 相对 → 绝对 | `data/a.csv` → `C:\sites\pytorch\data\a.csv` |
| 消除 `..` 和 `.` | `data\..\data\a.csv` → `data\a.csv` |
| 合并重复分隔符 | `data//a.csv` → `data\a.csv` |
| 解析软链接 | 顺链跟到真实文件 |
| 统一盘符和大小写 | `C:/SITES/...` → `C:\sites\...`（去问硬盘真实大小写） |

**补充**：构造 `Path()` 时已经顺手干掉了 `.` 和多余的 `/`，但**它不去动 `..`**。能消掉 `..` 的只有 `.resolve()`。

**类比写地址**：

```
「上海市宝山区XX路100号」
「XX路100号，宝山区，上海市」
「中国上海，宝山区，XX路100号」
```

都是同一个地方，但系统会当成三个不同字符串。`.resolve()` 就是**统一成官方标准写法**，这样比较、当字典 key、做缓存才不会出错。

---

## 六、环境小技巧：一键启动 Jupyter

**痛点**：每次要开 Git Bash → `cd /c/sites/pytorch` → 输入 `jupyter notebook`。

**解决方案**：[启动Jupyter.bat](file:///c:/sites/pytorch/启动Jupyter.bat)

```bat
@echo off
title Jupyter Notebook - c:\sites\pytorch
cd /d "%~dp0"
"C:\Users\yangbukun\miniconda3\Scripts\jupyter-notebook.exe"
```

| 行 | 作用 |
|---|---|
| `@echo off` | 不显示命令本身，窗口干净 |
| `title ...` | 给窗口起名，任务栏一眼认出 |
| `cd /d "%~dp0"` | 切到 **bat 文件自己所在的目录**（`%~dp0` 就是这个目录） |
| 最后一行 | 用全路径启动，不依赖 PATH |

**`%~dp0` 的好处**：脚本跟着文件夹走，整个目录改名/挪盘都不用改脚本。

**使用方式**
1. 桌面快捷方式，双击即可（已创建）
2. 右键任务栏图标 → 固定到任务栏

**注意**：弹出的黑窗口是服务端，**别关**。停止按 `Ctrl+C` 或关掉窗口。

**换环境 / 换目录**

```bat
@echo off
call "C:\Users\yangbukun\miniconda3\Scripts\activate.bat" oldfilm
cd /d "%~dp0"
jupyter-notebook
```

---

## 七、张量沿 axis 求和（2026-09-28 补充）

### 1. 一句话定义

**`sum(axis=k)` 就是把第 k 个维度「压扁」相加，结果里那个维度消失。**

`axis` 就是**要被消掉的那个维度**。

### 2. 2D 例子

```python
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])   # 形状 (2, 3)
```

`A.sum(axis=0)` —— 沿第 0 维（行）压，**每列竖着加**：

```
  1  2  3
  4  5  6
  ↓  ↓  ↓
  5  7  9
```

```python
A.sum(axis=0)     # tensor([5, 7, 9])   形状 (3,)   行没了
A.sum(axis=1)     # tensor([6, 15])     形状 (2,)   列没了
```

| axis | 方向 | 消掉谁 |
|---|---|---|
| `axis=0` | 从上往下压 | 行 |
| `axis=1` | 从左往右压 | 列 |

### 3. 3D 例子（重点）

```python
X = torch.arange(24).reshape(2, 3, 4)   # 形状 (2, 3, 4)
```

拆成两层 3×4 的表：

```
第 0 层                        第 1 层
 0   1   2   3                 12  13  14  15
 4   5   6   7                 16  17  18  19
 8   9  10  11                 20  21  22  23
```

**给它语义：2 个班级 × 3 名学生 × 4 门课**

| 写法 | 在图上 | 每个数怎么来 | 结果形状 |
|------|--------|-------------|---------|
| `sum(axis=0)` | 层与层之间 | 两层对应位置相加 | `(3, 4)` |
| `sum(axis=1)` | 每层内竖直方向 | 一列里 3 个数相加 | `(2, 4)` |
| `sum(axis=2)` | 每层内水平方向 | 一行里 4 个数相加 | `(2, 3)` |

对应语义：

- `axis=0` → 两个班合并
- `axis=1` → 3 名学生合并（每班 4 门课的单科总分）
- `axis=2` → 4 门课合并（每个学生的总分）

**结论**：`sum(axis=k)` 的结果 = **原形状删掉第 k 位**。

| 写法 | 结果形状 |
|---|---|
| `X.sum(axis=0)` | `(3, 4)` |
| `X.sum(axis=1)` | `(2, 4)` |
| `X.sum(axis=2)` | `(2, 3)` |
| `X.sum(axis=2)` 的值 | `tensor([[6, 22, 38], [54, 70, 86]])` |

### 4. 必须记住的坑

**`axis=0` 消掉的是「最外面那一层」，不是「上面那一行」。**

2D 时 `axis=0` 消掉行，容易误以为 `axis=0` = 「行方向」；到 3D，`axis=0` 消掉的是**层**。

> 别记「axis=0 是行」，要记「**axis=0 是第 0 位**」。按位置数，永远不会错。

`axis=-1` 表示最后一维（3D 里等价于 `axis=2`），**深度学习里非常常见**——经常要对最后一维（特征、类别概率）求和。

### 5. `keepdims=True`

```python
A.sum(axis=1)                        # tensor([6, 15])       形状 (2,)
A.sum(axis=1, keepdims=True)         # tensor([[6], [15]])   形状 (2, 1)
```

| 写法 | 形状 | 说明 |
|---|---|---|
| 默认 | `(2,)` | 维度被删掉 |
| `keepdims=True` | `(2, 1)` | 维度保留，缩成 1 |

**用途**：给 softmax 做归一化时，必须用它才能正确广播。

```python
A / A.sum(axis=1, keepdims=True)   # (2,3) / (2,1) 可广播，每行之和 = 1
```

### 6. 别和 `reshape` 混

```python
A.sum(axis=0)        # 元素数 6 → 3，真实做了加法（聚合）
A.reshape(3, 2)      # 元素数 6 → 6，只是重排（不变）
```

`sum` 是**减少元素**，`reshape` 是**元素不变只重排**。

---

## 八、向量求导与梯度（2026-09-29 补充）

### 1. 先破除恐惧

**`∂y/∂x` 没有做任何新运算，只是把一堆普通偏导数装进一个向量。**

假设 `y = x₁² + x₂² + x₃²`，`x = [1, 2, 3]`，则 `y = 1 + 4 + 9 = 14`。

```
∂y/∂x₁ = 2x₁ = 2      （只拧第 1 个旋钮）
∂y/∂x₂ = 2x₂ = 4      （只拧第 2 个旋钮）
∂y/∂x₃ = 2x₃ = 6      （只拧第 3 个旋钮）

∂y/∂x = [2, 4, 6]     ← 这就是「梯度」
```

**「偏」= 只看其中一个变量**，单独看就是一个普通的一元导数。

`∂y/∂xᵢ` 的含义：**把其他变量按住不动，只轻轻动一下 xᵢ，y 涨多快？**

### 2. 形状规律（唯一需要记的）

| 求谁 | 结果是什么 | 结果形状 |
|------|-----------|---------|
| 标量 → 标量 | 一个数 | 标量 |
| **标量 → 向量** | **向量** | **与 x 同形** |
| 标量 → 矩阵 | 矩阵 | 与 X 同形 |
| 向量 → 向量 | 矩阵（雅可比） | `(y的维度, x的维度)` |

**最常用的规律：标量对向量求导，结果形状 = 那个向量的形状。**

原因：每个分量对应一个偏导数。

### 3. 为什么书上的形状对不上

d2l 2.4 写的是：

```
∂y/∂x = [∂y/∂x₁, ..., ∂y/∂x₄]ᵀ     形状 (4, 1)
```

而 PyTorch 里：

```python
x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)      # = 2(x₁²+x₂²+x₃²+x₄²)
y.backward()
print(x.grad)                # tensor([0., 4., 8., 12.])   形状 (4,)
```

**是 `(4,)` 不是 `(4,1)`。** 这个差异纯属**记号约定**：数学书写列向量是为了矩阵乘法对齐，PyTorch 一维张量不区分行列。

> **看到 `ᵀ` 不用慌，那只是「摆正姿势」，不影响数值。** 只关心「哪几个数、按什么顺序排」。

### 4. 亲手验证

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()          # y = 1 + 4 + 9 = 14
y.backward()

print(y.item())             # 14.0
print(x.grad)               # tensor([2., 4., 6.])  ← 与手算 2xᵢ 一致
```

改一个数，梯度立刻跟着变：

```python
x = torch.tensor([5.0, 2.0, 3.0], requires_grad=True)
((x ** 2).sum()).backward()
print(x.grad)               # tensor([10., 4., 6.])  ← 2×5 = 10
```

### 5. 梯度的真正用途

```python
w = w - lr * w.grad      # 往梯度反方向走一小步，损失下降
```

**梯度指向 y 增长最快的方向**，想让它变小就往反方向走——这就是「梯度下降」。

这解释了**为什么 `∂y/∂x` 的形状必须和 `x` 一样**：要做 `x = x - lr * grad` 这个减法，两者必须逐元素对应。**形状规律不是数学家的洁癖，是被「参数更新」这个用途逼出来的。**

### 6. 放下心理包袱

| 你会做的 | 你不会做的 |
|---|---|
| 知道梯度 = 偏导数排成队 | 手推 `∂y/∂X` 的展开式 |
| 知道结果形状和谁一致 | 背矩阵求导的十条法则 |
| 会用 `.backward()` 和 `.grad` | 求雅可比矩阵的每个元素 |
| 会用链式法则连乘 | 手算多层复合的导数 |

李沐在 2.4 节讲矩阵求导，是为了让你**看得懂论文里的公式**（论文到处是 `∂L/∂W`），不是要你手算。2.5 节的自动微分会把这一切交给 `backward()`。

**目标**：看到 `∂y/∂x` 能翻译成「对 x 的每个分量分别求偏导，排成向量」，就够了。

---

## 附：核心记忆点

| 主题 | 一句话记住 |
|------|-----------|
| `reshape` | 各维乘积 = 元素总数 |
| `torch.cat` | 只有拼接维可以不同，其余必须相等 |
| 原地操作 | Python 看「索引赋值」，PyTorch 看方法名带 `_` |
| 切片 | `start:stop:step`，含头不含尾，越界不报错 |
| NumPy | PyTorch 的数据操作就是 NumPy + 自动求导 + GPU |
| CSV | 用逗号当格子线的纯文本表格 |
| 路径 | `BASE_DIR = Path(__file__).resolve().parent`，与 cwd 解耦 |
| `resolve()` | 把任意写法归一化成干净、唯一的绝对路径 |
| **`sum(axis=k)`** | **消掉第 k 维，结果形状 = 原形状删掉第 k 位** |
| **`keepdims=True`** | 求和后保留那一维（缩成 1），为了能正确广播 |
| **梯度 `∂y/∂x`** | 就是「每个分量的偏导数排成队」，形状与 x 相同 |
| **形状规则的来源** | 不是为了好看，是为了 `x = x - lr * grad` 能逐元素相减 |
| **矩阵求导** | 看得懂就行，手算交给 `.backward()` |