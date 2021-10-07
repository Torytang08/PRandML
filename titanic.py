import boost as boost
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import AdaBoostClassifier

# 读取数据集
train_data = pd.read_csv('C:/Users/Tory/Documents/学习超快乐/模式识别/titanic raw_data/train.csv')
test_data = pd.read_csv('C:/Users/Tory/Documents/学习超快乐/模式识别/titanic raw_data/test.csv')

# 选择用于训练的特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x_train = train_data[features]
x_test = test_data[features]

y_train = train_data['Survived']
# 检查缺失值
print('训练数据信息')
x_train.info()
print('-'*30)
print('测试数据信息：')
x_test.info()

# 使用登录最多的港口来填充登录港口的nan值
# print('\n\n\n登录港口信息：')
# print(x_train['Embarked'].value_counts())
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

# 使用平均年龄来填充年龄中的nan值
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

# 使用票价的均值填充票价中的nan值
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
# train_data = train_data.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# # 去除缺失值
# train_data = train_data.dropna()
#
# # 属性转化成数值型
# train_data_dummy = pd.get_dummies(train_data[['Sex', 'Embarked']])
#
# # y_train = train_data['Survived']
# # 编码后和数据拼接
# train_data_conti = pd.DataFrame(train_data, columns=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
#                                 index=train_data.index)
# train_data = train_data_conti.join(train_data_dummy)
#
# x_train = train_data.iloc[:, 1:]
# y_train = train_data.iloc[:, 0]
#
#
# # 处理test文件
# test_data = test_data.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# test_data = test_data.fillna(test_data.mean()['Age':'Fare']) #填补缺失值
# test_data_dummy = pd.get_dummies(test_data[['Sex', 'Embarked']])
# test_data_conti = pd.DataFrame(test_data, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
#                                 index=train_data.index)
# test_data = test_data_conti.join(test_data_dummy)
# x_test = test_data


# Standardize
# std = StandardScaler()
# x_train_conti_std = std.fit_transform(x_train[['Age', 'SibSp', 'Parch', 'Fare']])
# x_test_conti_std = std.fit_transform(x_test[['Age', 'SibSp', 'Parch', 'Fare']])
# # 将nd array转为traindata frame
# x_train_conti_std = pd.DataFrame(data=x_train_conti_std, columns=['Age', 'SibSp', 'Parch', 'Fare'], index=x_train.index)
# x_test_conti_std = pd.DataFrame(data=x_test_conti_std, columns=['Age', 'SibSp', 'Parch', 'Fare'], index=x_test.index)
#
# # 有序分类变量Pclass
# x_train_cat = x_train[['Pclass']]
# x_test_cat = x_test[['Pclass']]
# # 无序已编码的分类变量
# x_train_dummy = x_train[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
# x_test_dummy = x_test[['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
# # 拼接为traindataframe
# x_train_set = [x_train_cat, x_train_conti_std, x_train_dummy]
# x_test_set = [x_test_cat, x_test_conti_std, x_test_dummy]
# x_train = pd.concat(x_train_set, axis=1)
# x_test = pd.concat(x_test_set, axis=1)

# 将特征值转换成特征向量
vec = DictVectorizer(sparse=False)

x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

# 打印特征向量格式
print('\n\n\n特征向量格式')
print(vec.feature_names_)

# bayes
nb = MultinomialNB()
# Adaboost
# boost1 = AdaBoostClassifier()

print('\n\n\n模型验证:')
print('NaiveBayes acc is', np.mean(cross_val_score(nb, x_train, y_train, cv=10)))
# print('Adaboost acc is', np.mean(cross_val_score(boost1, x_train, y_train, cv=10)))

# clf_mul = MultinomialNB()
# train
nb.fit(x_train, y_train)
# boost1.fit(x_train, y_train)
# prediction
y_predict = nb.predict(x_test)
# y_predict = boost1.predict(x_test)
# save
# test_data['Survived'] = y_predict.astype(int)
# test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
# result = {'PassengerId': test_data['PassengerId'], 'survived': y_predict},
# result = pd.DataFrame(result)
# result.to_csv("submission.csv", index=False, sep=',')

test_data['Survived'] = y_predict.astype(int)
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', sep=',', index=False)
