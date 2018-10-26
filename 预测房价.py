
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import seaborn as sns
#设定绘图背景样式
sns.set_style('darkgrid')  
#设定图标颜色版
sns.set_palette('muted')


# In[6]:


df = pd.read_csv('kc_house_data.csv',encoding='utf-8')
df.head()


# 查看数据集概况

# In[7]:


df.info()


# 21613条记录,21个数值型字段，日期2014年5月到2015年5月，没有缺失值.
# 数据解释：date: “销售日期”：2014年5月到2015年5月房屋出售时的日期
# price: “销售价格”：房屋交易价格，单位为美元，是目标预测值
# bed_num: “卧室数”：房屋中的卧室数目
# bath_num: “浴室数”：房屋中的浴室数目
# sqft_living: “房屋面积”：房屋里的生活面积
# sqft_log:总占地面积
# floors:楼层数
# waterfront：是否可以看到海滨（0-1分类）
# view  ：被浏览次数
# condition  ：  总体状况如何
# grade  ： 根据King County的分级制度，对住房单元进行整体评分
# sqft_above  ：  除了地下室，房子的面积
# sqft_basement   地下室的面积
# yr_built        建造年份
# yr_renovated     房子翻修的年代
# zipcode          21613 non-null int64
# lat              21613 non-null float64
# long             21613 non-null float64
# sqft_living15    2015年的客厅区域(暗示——一些翻新)这可能会影响也可能不会影响大面积区域
# sqft_lot15      2015年总占地面积

# 数据预处理：删除不必要字段  id，date，lat，long，zipcode

# In[8]:


df.drop(['id','date','lat','long','zipcode'],axis=1,inplace=True)
df.head()


# In[9]:


df.info()


# In[10]:


#数值型变量的描述性统计：
df.describe()


# 15个自变量分为4类：
# 离散变量：数值：bedrooms，bathrooms，floors，view，condition，grade
# 0-1分类：waterfront	
# 连续变量：面积：sqft_living， sqft_lot，sqft_above，sqft_basement，sqft_living15，sqft_lot15
# 时间变量：日期：yr_built，yr_renovated	

# In[11]:


df.price.hist(bins =50)


# In[12]:


sns.boxplot(y = 'price',data=df)


# price呈现典型的右偏分布，大部分房屋的价格都在70万元以下，符合房价的一般规律。数据的离群值基本为100万以上的数据，与上面的右偏分布相吻合。

# 自变量与因变量的相关性分析
# 
# 绘制相关性矩阵热力图，比较各个变量之间的相关性：

# In[15]:


internal_chars =['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15']
corrmat = df[internal_chars].corr()  #计算相关系数
f,ax = plt.subplots(figsize=(10,6))   #设置图标尺寸大小
plt.xticks(rotation='0')
sns.heatmap(corrmat,square=False,linewidth=.5,annot=True)    #设置热力图参数



# 可以看到与price相关性较大的有：bathrooms,sqft_living,grade,sqft_above,sqft_living15

# In[16]:


sns.boxplot(x='bathrooms',y='price',data=df)   #绘制分组箱线图


# 根据箱线图，浴室数目越多，价格越高

# In[17]:


2
sns.jointplot('sqft_living','price',data=df,kind='reg')   #绘制散点图


# 房屋面积和房价呈现一定的线性关系，且房屋面积近似服从正态分布。

# In[18]:


sns.jointplot('grade','price',data =df,kind='reg')


# 因为得分为离散型变量，绘制出的散点图很像分类点线图了，两者也存在线性关系。

# In[19]:


sns.jointplot('sqft_above','price',data =df,kind='reg')


# 建筑面积和房价的关系类似上述的房屋价格，不过相关系数稍低一点

# In[20]:


sns.jointplot('sqft_living15','price',data =df,kind='reg')


# 2015年建筑面积和房价的关系类似上述的房屋价格，不过相关系数更低

# 特征筛选：
# 
# 虽然之前在探索性分析中我们已经筛选出了4个自变量，但这只是人工地进行筛选，并不科学，所以我们要用“机器学习”中的特征选择方法继续筛选。通常，可以从两个方面来选择特征：
# 
# 特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。
# 特征与因变量的相关性：这点比较显见，与因变量相关性高的特征，应当优选选择。
# 特征选择的方法有很多，例如：
# 
# Filter 法：方差选择，相关系数选择，卡方检验，互信息法
# 
# Wrapper 法：递归特征消除法
# 
# Embedded法：基于惩罚项的选择，基于树模型的选择
# 

# 1）使用 Wrapper 法筛选：
# 
# ----递归消除特征法是使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。使用feature_selection库的RFE类来选择特征。

# In[22]:


x =df
x=np.array(x)  #建立自变量二维数组
x


# In[24]:


y = df[['price']]
y = np.array(y)  #建立因变量二维数组
y


# In[34]:


from sklearn.linear_model import LinearRegression     # 导入基模型
from sklearn.feature_selection import RFE             # 导入RFE模块
model1 = LinearRegression()                           # 建立一个线性模型
rfe = RFE(model1,5)                                   # 进行多轮训练，设置筛选特征数目为5个
rfe = rfe.fit(x,y)                                    # 模型的拟合训练
print(rfe.support_)                                   # 输出特征的选择结果
print(rfe.ranking_)                                   # 特征的选择排名


# 输出结果中'True'为选择的特征变量，排名也是第1位，可以得出用Wrapper法筛选的特征为：bedrooms,sqft_living,waterfront,view,grade这5个。

# 利用“交叉检验”的方法看如果这四个变量进入模型，模型的性能怎么样。
# 
# ---交叉检验的概念是将数据分成训练集和测试集，取一部分训练集数据得到回归方程，并在测试集中进行检验，观察正确度，以此来评判模型的好坏。

# In[35]:


from sklearn.model_selection import cross_val_score               #导入交叉检验的模块
x_test1 = df[['bedrooms','sqft_living','waterfront','view','grade']]
x_test1 = np.array(x_test1)                                       # 建立自变量的二维数组
y_test1 = df[['price']]
y_test1 = np.array(y_test1)                                       # 建立因变量的二维数组
model2 = LinearRegression()                                       # 建立线性模型
model2.fit(x_test1,y_test1)                                        # 模型的拟合训练
scores = -cross_val_score(model2, x_test1, y_test1, cv=5, scoring= 'neg_mean_absolute_error')
print(np.mean(scores))                                            # 将数据集分为5份，分别进行5次回归，返回得分


# 输出：155262.9475883078，得分越高，说明模型的误差越大，所以输出的结果越小越好。

# 下面用Filter法进行特征选择：

# In[39]:


from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)                 # 用f_classif方法，设定数目为4
a=selector.fit(x,y)
print(np.array(a.scores_),'\n',a.get_support())        #  输出得分及选择的结果


# 得出用Filter法筛选的特征为：['bedrooms','sqft_lot','sqft_above','sqft_basement','sqft_living15']5个，发现这个结果与我们用人工筛选的结果一样，我们再看看交叉检验结果如何：

# In[40]:


x_test2 = df[['bedrooms','sqft_lot','sqft_above','sqft_basement','sqft_living15']]
x_test2 = np.array(x_test2)
y_test2 = df[['price']]
y_test2 = np.array(y_test2)
model3 = LinearRegression()
model3.fit(x_test2,y_test2)
scores = -cross_val_score(model3, x_test2, y_test2, cv=5, scoring= 'neg_mean_absolute_error')
print(np.mean(scores))


# 输出：169191.95245875785，比用Filter法的结果小，说明用Wrapper法筛选的效果更好。
# 选择bedrooms，sqft_lot，sqft_above,sqft_basement,sqft_living15这5个变量放入模型

# 建立多元回归模型

# In[43]:


from sklearn.linear_model import LinearRegression
x = df[['bedrooms','sqft_lot','sqft_above','sqft_basement','sqft_living15']]
x = np.array(x)
y = df[['price']]
y = np.array(y)
model = LinearRegression()
model.fit(x,y)
a = model.intercept_  # a为回归方程的截距项
b = model.coef_       # b为回归方程的回归系数
print('y = {} + {} * X'.format(a,b))


# 设'bedrooms','sqft_lot','sqft_above','sqft_basement','sqft_living15'分别为x1,x2,x3,x4,x5
# 所得到的方程为：y = -56879*x1 - 0.37*x2 +269*x3 + 310*x4 +28776

# In[56]:


df['price_pre'] = df.apply(lambda x:x.bedrooms*-56879 - x.sqft_lot*0.37 + x.sqft_above*269+ x.sqft_basement*310 +28776,axis=1)
df_select = df[['price','price_pre']]
df_select.head()


# 前三条误差可以接受，后两条误差较大

# 导入测试集数据，开始预测

# In[46]:


test = pd.read_csv('kc_house_data _test.csv',encoding='utf-8')
test['price_predice'] = test.apply(lambda x:x.bedrooms*-56879 - x.sqft_lot*0.37 + x.sqft_above*269+ x.sqft_basement*310 +28776,axis=1)
test.head()

