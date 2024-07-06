import pandas as pd
import xlrd
import xlwt
import math

from matplotlib.ticker import MultipleLocator
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import KernelPCA
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
import random
import classifier as cl
import warnings
from scipy.io import loadmat  #导入 loadmat, 用于对 mat 格式文件进行操作
import matplotlib.pyplot as plt  #导入绘图操作用到的库
import smote_variants
from sklearn.decomposition import FastICA
from sklearn import manifold
import umap

p_means = []
q_means = []
def chpoint(X, k):
    Pointnum = []
    for i in range(X.shape[0]):
        a = 0
        num = 0
        y = []
        distence = []
        for z in range(X.shape[0]):
            if i != z:
                distence.append(np.linalg.norm(X[i] - X[z], ord=2))
        distence.sort()
        for j in range(X.shape[0]):
            if np.linalg.norm(X[i] - X[j], ord=2) < np.mean(distence[0:k]):
                y.append(j)
                a = a + 1
        for m in range(len(y)):
            num = num + math.exp(-(np.linalg.norm(X[i] - X[y[m]], ord=2)/a)**2)

        Pointnum.append([i, num])

    Pointnum1 = np.array(Pointnum)

    num = np.lexsort(-np.matrix(Pointnum1).T)
    fake_data = Pointnum1[num[0], :]

    fake_data_dis = fake_data[int(fake_data.shape[0] * 0.7):]
    return fake_data_dis


def means(X):
    """
    启发式的选取两个点

    参数
    ----------
    X : 特征矩阵

    返回
    ----------
    两个向量点
    """


    iteration_steps = 20
    count = X.shape[0]
    fake_data_disx = chpoint(X, 12)
    count1 = fake_data_disx.shape[0]
    i = int(fake_data_disx[0][0])
    j = int(fake_data_disx[count1 - 1][0])

    # 保证 i\j 不相同
    j += (j >= i)
    ic = 1
    jc = 1
    p = X[i]
    q = X[j]
    for l in range(iteration_steps):
        k = np.random.randint(0, count)
        di = ic * distance(p, X[k])
        dj = jc * distance(q, X[k])
        if di == dj:
            continue
        if di < dj:
            p = (p * ic + X[k]) / (ic + 1)
            ic = ic + 1
        else:
            q = (q * jc + X[k]) / (jc + 1)
            jc = jc + 1
    p_means.append(i)
    q_means.append(j)
    return p, q


def distance(a, b):
    """
    计算距离

    参数
    ----------
    a : 向量 a

    b : 向量 b

    返回
    ----------
    向量 a 与 向量 b 直接的距离
    """
    return np.linalg.norm(a - b)


class annoynode:
    """
    Annoy 树结点
    """

    def __init__(self, index, size, w, b, left=None, right=None):
        # 结点包含的样本点下标
        self.index = index
        # 结点及其子结点包含的样本数
        self.size = size
        # 分割超平面的系数
        self.w = w
        # 分割超平面的偏移量
        self.b = b
        # 左子树
        self.left = left
        # 右子树
        self.right = right

    def __lt__(self, other):
        # 结点大小比较
        return self.size < other.size


class annoytree:
    """
    Annoy 树算法实现

    参数
    ----------
    X : 特征矩阵

    leaf_size : 叶子节点包含的最大特征向量数量，默认为 10
    """

    def __init__(self, X, leaf_size=10):
        def build_node(X_indexes):
            """
            构建结点

            参数
            ----------
            X_indexes : 特征矩阵下标
            """
            # 当特征矩阵小于等于指定的叶子结点的大小时，创建叶子结点并返回
            if len(X_indexes) <= leaf_size:
                return annoynode(X_indexes, len(X_indexes), None, None)
            # 当前特征矩阵
            _X = X[X_indexes, :]
            # 启发式的选取两点
            p, q = means(_X)
            # 超平面的系数
            w = p - q
            # 超平面的偏移量
            b = -np.dot((p + q) / 2, w)
            # 构建结点
            node = annoynode(None, len(X_indexes), w, b)
            # 在超平面“左”侧的特征矩阵下标
            left_index = (_X.dot(w) + b) > 0
            if left_index.any():
                # 递归的构建左子树
                node.left = build_node(X_indexes[left_index])
            # 在超平面“右”侧的特征矩阵下标
            right_index = ~left_index
            if right_index.any():
                # 递归的构建右子树
                node.right = build_node(X_indexes[right_index])
            return node

        # 根结点
        self.root = build_node(np.array(range(X.shape[0])))


class annoytrees:
    """
    Annoy 算法实现

    参数
    ----------
    X : 特征矩阵

    n_trees : Annoy 树的数量，默认为 10

    leaf_size : 叶子节点包含的最大特征向量数量，默认为 10
    """

    def __init__(self, X, n_trees=10, leaf_size=10):
        self._X = X
        self._trees = []
        # 循环的创建 Annoy 树
        for i in range(n_trees):
            self._trees.append(annoytree(X, leaf_size=leaf_size))

    def query(self, x, k=1, search_k=-1):
        """
        查询距离最近 k 个特征向量

        参数
        ----------
        x : 目标向量

        k : 查询邻居数量

        search_k : 最少遍历出的邻居数量，默认为 Annoy 树的数量 * 查询数量
        """

        # 创建结点优先级队列
        nodes = PriorityQueue()
        # 先将所有根结点加入到队列中
        for tree in self._trees:
            nodes.put([float("inf"), tree.root])
        if search_k == -1:
            search_k = len(self._trees) * k
        # 待查询的邻居下标数组
        nns = []
        # 循环优先级队列
        while len(nns) < search_k and not nodes.empty():
            # 获取优先级最高的结点
            (dist, node) = nodes.get()
            # 如果是叶子结点，将下标数组加入待查询的邻居中
            if node.left is None and node.right is None:
                nns.extend(node.index)
            else:
                # 计算目标向量到结点超平面的距离
                dist = min(dist, np.abs(x.dot(node.w) + node.b))
                # 将距离做为优先级的结点加入到优先级队列中
                if node.left is not None:
                    nodes.put([dist, node.left])
                if node.right is not None:
                    nodes.put([dist, node.right])
        # 对下标数组进行排序
        nns.sort()
        prev = -1
        # 优先级队列
        nns_distance = PriorityQueue()
        for idx in nns:
            # 过滤重复的特征矩阵下标
            if idx == prev:
                continue
            prev = idx
            # 计算特征向量与目标向量的距离做为优先级
            nns_distance.put([distance(x, self._X[idx]), idx])
        nearests = []
        distances = []
        # 取前 k 个
        for i in range(k):
            if nns_distance.empty():
                break
            (dist, idx) = nns_distance.get()
            nearests.append(idx)
            distances.append(dist)
        return nearests, distances

    def function(self):
        nodes = PriorityQueue()
        # 先将所有根结点加入到队列中
        for tree in self._trees:
            print(tree.root)
            nodes.put([float("inf"), tree.root])

def load_data(path):

    # Step 1. 导入训练数据
    # 划分少数类和多数类
    data_0 = pd.read_excel(path, sheet_name='major',
                           header=None)  # Read most types of data
    data_1 = pd.read_excel(path, sheet_name='minor',
                           header=None)  # Read minority data
    print(np.shape(data_0))  # 187,66
    print(np.shape(data_1))  # 15,66
    # X_0为多数类，X_1为少数类
    X_0, Y_0 = data_0.iloc[:, 1:].values, data_0.iloc[:, 0].values
    X_1, Y_1 = data_1.iloc[:, 1:].values, data_1.iloc[:, 0].values
    X_0 = pd.DataFrame(X_0)
    X_1 = pd.DataFrame(X_1)
    Y_0 = pd.DataFrame(Y_0)
    Y_1 = pd.DataFrame(Y_1)

    dataset = np.vstack((X_0, X_1))

    Y = np.vstack((Y_0, Y_1))

    # 把负类的标签全部用-1表示
    for i in range(np.shape(Y)[0]):
        if Y[i] == 0:
            Y[i] = -1

    dataArr, testArr, LabelArr, testLabelArr = train_test_split(dataset, Y, test_size=0.2, random_state=1, stratify=Y)

    return dataArr,testArr,LabelArr,testLabelArr

def get_num_leaf(root):
    """获取二叉树叶子节点"""
    if root==None:
        return 0 #当二叉树为空时直接返回0
    elif root.left==None and root.right==None:
        return 1 #当二叉树只有一个根，但是无左右孩子时，根节点就是一个叶子节点
    else:
        return (get_num_leaf(root.left)+get_num_leaf(root.right))  #其他情况就需要根据递归来实现

def Partitioned_subspace_matrix(at1, leaf, data):
    '''
    划分子空间矩阵
    '''
    prepare_list = locals()
    # 将原样本矩阵划分为多个子空间样本矩阵：
    for i in range(get_num_leaf(at1.root)):

        prepare_list['subspace_matrix' + str(i)] = []
        for j in range(len(leaf[i])):
            prepare_list['subspace_matrix' + str(i)].append(data[leaf[i][j], :].tolist())

        np.mat(prepare_list['subspace_matrix' + str(i)])

        # print(np.shape(prepare_list['subspace_matrix' + str(i)]))
        # print(prepare_list['subspace_matrix' + str(i)])
    # 将数组转换成矩阵
    for i in range(len(leaf)):
        prepare_list['subspace_matrix' + str(i)] = np.mat(prepare_list['subspace_matrix' + str(i)])

    return prepare_list

def inorder(node):
    if node is None:
        return

    inorder(node.left)
    if node.left == None and node.right == None:
        leaf.append(node.index)
    inorder(node.right)

def import_excel_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = np.zeros((row, col))  # 生成一个nrows行*ncols列的初始矩阵
    for i in range(col):  # 对列进行遍历
        cols = np.matrix(table.col_values(i))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, i] = cols  # 按列把数据存进矩阵中
    return datamatrix

def xlsxToTxt(xls_name, txt_name):
    data = pd.read_excel(xls_name, header=None)
    data.to_csv(txt_name, sep='\t', index=False, header=False)

def save(matrix, path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = matrix.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, matrix[i, j])
    f.save(path)

def amplification(N, P, Z, B):
    count1 = 0   #扩增的样本计数

    i = 0  #从第一行开始找距离最近值
    while count1 < int(1.5 * B):        # 正样本扩增的数量
        if i == A:
            i = 0
        a = P[i]
        a1 = a.flatten()
        min = 999  # 距离最小值
        max = 0  # 距离最大值
        for j in range(A):

            b_min = P[j]
            b_min1 = b_min.flatten()
            per_min = np.linalg.norm(a1 - b_min1)
            if per_min < min and per_min != 0:
                min = per_min
                k = j  #找到离目标距离最小的那一行
        # print("min ========", min)
        for n in range(A):

            b_max = P[n]
            b_max1 = b_max.flatten()
            per_max = np.linalg.norm(a1 - b_max1)
            if per_max > max and per_max != 0:
                max = per_max
                m = n  # 找到离目标距离最小的那一行
        c = random.uniform(0, 0.5)
        d = random.uniform(0, 0.5)
        # print("max ========", max)
        Z[count1] =d*P[m] + c*P[k] + (1-c-d)*P[i]   #Z是扩增后的矩阵
        count1 = count1 + 1
        i = i + 1

    return Z

def filter(Z, F, P, PZ, PF, k):

    Pointnum = []
    Point = []
    for i in range(Z):
            a = 0
            num = 0
            y = []
            distence = []
            for z in range(P.shape[0]):
                if i != z:
                    distence.append(np.linalg.norm(PZ[i] - P[z], ord=2))
            distence.sort()
            for j in range(P.shape[0]):
                if np.linalg.norm(PZ[i] - P[j], ord=2) < np.mean(distence[0:k]):
                    y.append(j)
                    a = a + 1
            for m in range(len(y)):
                num = num + math.exp(-(np.linalg.norm(PZ[i] - P[y[m]], ord=2)/a)**2)

            Pointnum.append([i, num])

    Pointnum1 = np.array(Pointnum)

    num = np.lexsort(-np.matrix(Pointnum1).T)
    fake_data = Pointnum1[num[0], :]

    fake_data_dis = fake_data[int(fake_data.shape[0] * 0.02):int(fake_data.shape[0] * 0.9), :]

    L = random.sample(range(0, fake_data_dis.shape[0]), F)


    for n in range(F):
        Point.append(PZ[int(fake_data_dis[L[n]][0])])
    return Point

def filter1(Z, F, PZ, PF):

    H = []  # 权重距离矩阵
    H1 = np.zeros(shape=(int(Z)))  # 同类距离矩阵
    H2 = np.zeros(shape=(int(Z)))  # 不同类距离矩阵

    for i in range(F):
        a = PF[i]
        a1 = a.flatten()
        per1 = 0
        per2 = 0
        for j in range(F):  # 考虑同类距离
            c = PF[j]  # P 是原始矩阵
            c1 = c.flatten()
            per1 = per1 + np.linalg.norm(a1 - c1)
        H1[i] = per1

        for k in range(Z):  # 考虑不同类距离
            b = PZ[k]  # P 是原始矩阵
            b1 = b.flatten()
            per2 = per2 + np.linalg.norm(a1 - b1)
        H2[k] = per2  # H是距离矩阵


    for i in range(H1.shape[0]):
        m = 0.1 * H1[i] + 0.9 * H2[i]
        H.append([i, m])

    Pointnum1 = np.array(H)

    num = np.lexsort(-np.matrix(Pointnum1).T)
    fake_data = Pointnum1[num[0], :]

    fake_data_dis = fake_data[0:F-20, :]

    X = []
    for i in range(F-100):

        X.append(PF[int(fake_data_dis[i][0])])

    return X



def showDataSet_New(featureMat, labelMat, h):
    #创建标签为1的样本列表
    data_one = []
    #创建标签为0的样本列表
    data_zero = []
    #遍历特征矩阵featureMat，i是特征矩阵featureMat的当前行
    #特征矩阵featureMat的两个特征列，正好是散点图的数据点的x轴坐标和y轴坐标
    for i in range(len(featureMat)):
        #如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为1
        if labelMat[i] == 1:
            #将当前特征矩阵featureMat[i]行添入data_one列表
            data_one.append(featureMat[i])
        #如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为0
        elif labelMat[i] == 0:
            #将当前特征矩阵featureMat[i]行添入data_zero列表
            data_zero.append(featureMat[i])
    #将做好的data_one列表转换为numpy数组data_one_np
    data_one_np = np.array(data_one)
    #将做好的data_zero列表转换为numpy数组data_zero_np
    data_zero_np = np.array(data_zero)
    plt.figure(dpi=800)
    plt.xticks(fontsize=14, color='black')
    plt.yticks(fontsize=14, color='black')

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    x_major_locator = MultipleLocator(1)  # 以每15显示
    y_major_locator = MultipleLocator(1)  # 以每3显示
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    #根据标签为1的样本的x坐标（即data_one_np的第0列）和y坐标（即data_one_np的第1列）来绘制散点图
    plt.scatter(data_one_np[:, 0]/10, data_one_np[:, 1]/10, s=2, label='Positive Sample')
    #根据标签为0的样本的x坐标（即data_zero_np的第0列）和y坐标（即data_zero_np的第1列）来绘制散点图
    plt.scatter(data_zero_np[:, 0]/10, data_zero_np[:, 1]/10, s=2, label='Negative Sample')
    #显示画好的散点图

    plt.legend()
    plt.savefig('./imbalance/D008881/D008881_png/' + str(h) + '.png')

def showDataSet1(featureMat, labelMat):
    # 创建标签为1的样本列表
    data_one = []
    # 创建标签为0的样本列表
    data_zero = []
    # 遍历特征矩阵featureMat，i是特征矩阵featureMat的当前行
    # 特征矩阵featureMat的两个特征列，正好是散点图的数据点的x轴坐标和y轴坐标
    for i in range(len(featureMat)):
        # 如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为1
        if labelMat[i] == 1:
            # 将当前特征矩阵featureMat[i]行添入data_one列表
            data_one.append(featureMat[i])
        # 如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为0
        elif labelMat[i] == 0:
            # 将当前特征矩阵featureMat[i]行添入data_zero列表
            data_zero.append(featureMat[i])
    # 将做好的data_one列表转换为numpy数组data_one_np
    data_one_np = np.array(data_one)
    # 将做好的data_zero列表转换为numpy数组data_zero_np
    data_zero_np = np.array(data_zero)
    plt.figure(dpi=250)
    # plt.xticks(fontsize=14, color='black')
    # plt.yticks(fontsize=14, color='black')
    #
    # plt.xlim(-40, 40)
    # plt.ylim(-60, 80)
    # x_major_locator = MultipleLocator(20)  # 以每15显示
    # y_major_locator = MultipleLocator(20)  # 以每3显示
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)

    # 根据标签为1的样本的x坐标（即data_one_np的第0列）和y坐标（即data_one_np的第1列）来绘制散点图
    plt.scatter(data_one_np[:, 0], data_one_np[:, 1], s=2, label='Positive Sample')
    # 根据标签为0的样本的x坐标（即data_zero_np的第0列）和y坐标（即data_zero_np的第1列）来绘制散点图
    plt.scatter(data_zero_np[:, 0], data_zero_np[:, 1], s=2, label='Negative Sample')
    # 显示画好的散点图
    plt.legend()
    plt.savefig('./imbalance/D008881/8881/4_13.png')
    # plt.show()

def showDataSet(featureMat, labelMat, h):
    # 创建标签为1的样本列表
    data_one = []
    # 创建标签为0的样本列表
    data_zero = []
    # 遍历特征矩阵featureMat，i是特征矩阵featureMat的当前行
    # 特征矩阵featureMat的两个特征列，正好是散点图的数据点的x轴坐标和y轴坐标
    for i in range(len(featureMat)):
        # 如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为1
        if labelMat[i] == 1:
            # 将当前特征矩阵featureMat[i]行添入data_one列表
            data_one.append(featureMat[i])
        # 如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为0
        elif labelMat[i] == 0:
            # 将当前特征矩阵featureMat[i]行添入data_zero列表
            data_zero.append(featureMat[i])
    # 将做好的data_one列表转换为numpy数组data_one_np
    data_one_np = np.array(data_one)
    # 将做好的data_zero列表转换为numpy数组data_zero_np
    data_zero_np = np.array(data_zero)

    plt.figure(dpi=800)
    plt.xticks(fontsize=14, color='black')
    plt.yticks(fontsize=14, color='black')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    x_major_locator = MultipleLocator(1)  # 以每15显示
    y_major_locator = MultipleLocator(1)  # 以每3显示
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    # 根据标签为1的样本的x坐标（即data_one_np的第0列）和y坐标（即data_one_np的第1列）来绘制散点图
    plt.scatter(data_one_np[:, 0]/10, data_one_np[:, 1]/10, s=2, label='Positive Sample')
    # 根据标签为0的样本的x坐标（即data_zero_np的第0列）和y坐标（即data_zero_np的第1列）来绘制散点图
    plt.scatter(data_zero_np[:, 0]/10, data_zero_np[:, 1]/10, s=2, label='Negative Sample')
    # 显示画好的散点图

    plt.legend()
    plt.savefig('./imbalance/D008881/D008881/' + str(h) + '.png')

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    warnings.filterwarnings("ignore")
    dataset = '02318'
    number = '01'
    data_file_1 = f"transformfile/D0{dataset}/{number}"\
                  "" \
                  "/m.xls"  # 去标签的训练集矩阵分解后的结果

    data_file_2 = f"imbalance/D0{dataset}/{number}" \
                  "" \
                  "/D0" + str(dataset) + "trainL.xls"  # 训练集标签

    data_file_3 = f"imbalance/D0{dataset}/{number}" \
                  "" \
                  "/D0" + str(dataset) + "test.xls"  # 测试集

    data_file_4 = f"transformfile/D0{dataset}/{number}"\
                  "" \
                  "/l.xls"  # 去标签的训练集矩阵分解后的结果

    data_file_5 = f"imbalance/D0{dataset}/{number}" \
                  "" \
                  "/D0" + str(dataset) + "train.xls"  # 训练集

    matrix_U = import_excel_matrix(data_file_1)  # 转换成矩阵（调用函数）
    U = np.array(matrix_U) #转换为矩阵   训练集
    print('矩阵分解训练集：' + str(U.shape))
    matrix_Label = import_excel_matrix(data_file_2)  # 转换成矩阵（调用函数）
    Label = np.array(matrix_Label) #训练集标签
    print('训练集标签：' + str(Label.shape))
    matrix_test = import_excel_matrix(data_file_3)  # 转换成矩阵（调用函数）
    test = np.array(matrix_test)  # 测试集
    print('测试集：' + str(test.shape))
    matrix_l = import_excel_matrix(data_file_4)  # 转换成矩阵（调用函数）
    llist = np.array(matrix_l)
    print('矩阵分解：' + str(llist.shape))

    trainnew = np.array(import_excel_matrix(data_file_5)) #原始训练集
    print("原始训练集：" + str(trainnew.shape))
    print('--------------------------------数据输入----------------------------------------')
    X_tsne = TSNE(n_components=2, n_iter=300).fit_transform(np.mat(trainnew))
    showDataSet1(X_tsne, Label)
    # plt.scatter(np.transpose(U)[0], np.transpose(U)[1], c='k', zorder=1)
    # plt.show()
    # trans_data = manifold.Isomap(n_neighbors=4, n_components=2, n_jobs=-1).fit_transform(np.mat(trainnew))
    # showDataSet(trans_data, Label)
    #print(chpoint(U))

    trainS = np.concatenate([Label, U], axis=1)#原始训练集，带标签
    print(trainS.shape)

    NumZ = 0
    NumF = 0
    for i in range(trainS.shape[0]):
        if int(trainS[i][0]) == 1:
            NumZ = NumZ + 1
        else:
            NumF = NumF + 1

    Y = [] #存储扩增样本的列表
    Y_label = []
    Y_train = []
    Y1 = []
    at = annoytree(U, 250) #空间划分，每个子空间最多200个样本
    leaf = [] #存储叶子节点
    inorder(at.root) #遍历annoy二叉树
    prepare_list = Partitioned_subspace_matrix(at, leaf, trainS)#保存每一个子空间

    for i in range(get_num_leaf(at.root)):
        a = 0  # 记录子空间正样本数量
        b = 0  # 记录子空间负样本数量
        V = np.array(prepare_list['subspace_matrix' + str(i)])  # 当前需要处理的子空间

        # V_label = np.array(V[:, 0])
        # V_te = np.delete(V, 0, 1)
        # X_tsne = TSNE(n_components=2, n_iter=2500, random_state=23).fit_transform(np.mat(V_te))
        # showDataSet(X_tsne, V_label, i)

        for j in range(V.shape[0]):
            if prepare_list['subspace_matrix' + str(i)][j, 0] == 1:
                a = a + 1
            else:
                b = b + 1
        print("第" + str(i) + "个子空间正样本数：", a)
        if a <= 2 or b == 0:  # 设置空间扩增条件
            print("---------------------第" + str(i) + "个子空间不能扩增-------------------------")
        else:
            fl = b / (a + b)
            Z = np.zeros(shape=(int(1.5 * b * fl ), trainS.shape[1]))  # 对U进行扩增的部分 1.5 * 代价因子 * 原始正样本数
            X = np.zeros(shape=(int(b * fl ), trainS.shape[1]))  # 筛选后的正样本  代价因子*原始正样本数
            A = a  # 正样本的数量
            nX = amplification(A, V, Z, int(b * fl ))  # 扩增函数
            for m in range(len(nX)):
                Y.append(nX[m])

            # tu = np.row_stack((V, nX))
            # tu_label = np.array(tu[:, 0])
            # tu_te = np.delete(tu, 0, 1)
            # X_tsne = TSNE(n_components=2, random_state=23).fit_transform(np.mat(tu_te))
            # showDataSet_New(X_tsne, tu_label, i)

    I = np.array(Y)

    P = np.row_stack((trainS, I))  # 将原始的和扩增后的合并

    PZ = []
    PF = []
    NumZ = 0
    NumF = 0
    for i in range(P.shape[0]):
        if int(P[i][0]) == 1:
            NumZ = NumZ + 1
            PZ.append(P[i])
        elif int(P[i][0]) == 0:
            NumF = NumF + 1
            PF.append(P[i])
    PZ_new = np.array(PZ)
    PF_new = np.array(PF)

    PF_new = PF_new[PF_new[:, 0] == 0]
    PF_new = PF_new[PF_new[:, 1] != 0]

    newI = np.array(filter(NumZ, PF_new.shape[0], P, PZ_new, PF_new, 20))
    Snew = np.row_stack((newI, PF_new))

    # PZ = Snew[Snew[:, 0] == 1]
    # PF = Snew[Snew[:, 0] == 0]
    # newI = filter1(PZ.shape[0], PF.shape[0], PZ, PF)
    # Snew = np.row_stack((newI, PZ))
    #
    # print('扩增后的训练集：' + str(Snew.shape))
    #
    Ftrain = np.delete(Snew, 0, 1)  # 训练集去标签
    # print(Flabel.shape)
    F1 = np.dot(Ftrain, llist)  # 矩阵分解相乘
    # print(F1.shape)
    F1_test = np.delete(test, 0, 1)  # 测试集去标签
    label = np.array(Snew[:, 0])  # 扩增后的训练集标签

    cl.classif(F1, label, F1_test, test, trainnew, Label)  # 训练+预测+指标打印
    # label = np.reshape(label, (label.shape[0], 1))
    # print(label.shape)
    # print(F1.shape)
    # F2 = np.concatenate([label, F1], axis=1)
    # F2_Z = F2[F2[:, 0] == 0]
    # F2_F = F2[F2[:, 0] == 1]
    # print(F2_F)
    # print(F2_Z)
    # F2_Z = np.delete(F2_Z, 0, 1)  # 测试集去标签
    # F2_F = np.delete(F2_F, 0, 1)
    #
    # print(F2_Z.shape)
    # print(F2_F.shape)
    # showDataSet1(1,F2_F,F2_Z)


    # X_tsne1 = TSNE(n_components=2, random_state=100).fit_transform(np.mat(trainS))
    # showDataSet1(X_tsne1, label)



    # p = []
    # q = []
    # X_tsne = TSNE(n_components=2, n_iter=300).fit_transform(np.mat(trainnew))
    # for j in range(get_num_leaf(at.root) - 1):
    #     p.append(X_tsne[p_means[j], :])
    #     q.append(X_tsne[q_means[j], :])
    # q1 = np.array(q)
    # p1 = np.array(p)
    #
    # for h in range(get_num_leaf(at.root) - 1):
    #     plt.figure(dpi=800)
    #
    #     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=2, c='k', zorder=1)
    #     # plt.scatter(np.transpose(pq_means)[0], np.transpose(pq_means)[1], c='r')
    #     #plt.scatter(np.transpose(trainnew)[0], np.transpose(trainnew)[1], c='k', zorder=1)
    #     for k in range(h + 1):
    #         plt.title(h)
    #         # plt.scatter(np.transpose(q[k])[0], np.transpose(q[k])[1], c='r', zorder=3)
    #         # plt.scatter(np.transpose(p[k])[0], np.transpose(p[k])[1], c='r', zorder=3)
    #         plt.scatter(p1[k][0], p1[k][1], s=2, c='r', zorder=3)
    #         plt.scatter(q1[k][0], q1[k][1], s=2, c='r', zorder=3)
    #         plt.savefig('./imbalance/D002446/D002446_mannoy/' + str(h) + '.png')

    # X_tsne1 = TSNE(n_components=2, random_state=33).fit_transform(np.mat(F1))
    # showDataSet1(X_tsne1, label)


    # kpca = KernelPCA(kernel='rbf', gamma=10, n_components=2)
    # newMat = kpca.fit_transform(np.mat(F1))
    # showDataSet(newMat, label)
    # F2 = np.concatenate([Flabel, F1], axis=1) #数据集与标签合并
    # # print(F2.shape)
    # #
    # # 存储为表格
    # excel = save(Snew, r'C:\Users\lenovo\Desktop\test\annoy\imbalance\D008881\D8881.xls') #扩增后的数据集


