
# coding: utf-8

# # Лабораторная работа по анализу данных
# ## Работу выполнил студент группы 14 ПМИ 
# ### Огурцов Антон
# Данные телекоммуникационной компании, содержащей информацию об оттоке клиентов. Набор данных содержит 7044 пользователей. В качестве атрибутов принимаются:
# - Id клиента.
# - Сервисы внутри телефонной компании, к которой подключился пользователь:  phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
# - Информация о пользователе, как о клиенте сервиса: how long they have been a customer, contract, payment method, paperless billing, monthly charges, and total charges.
# - Личная (демографическая) информация о пользователе:
#  gender, age range, and if they have partners and dependents. 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
get_ipython().magic('matplotlib inline')


# In[4]:


data = pd.read_csv('Telco-Customer-Churn.csv', sep = ',', engine = 'python')


# In[5]:


data


# Так как была возможность предложить свой датасет, мне стало интересно попробовать применить анализ данных для оттока клиентов.
# 

# In[6]:


type(data)


# In[7]:


data.index


# In[8]:


data.head()


# In[9]:


data.describe()


# In[10]:


from pandas.tools.plotting import scatter_matrix
scatter_matrix(data, alpha = .05, figsize = (10, 10))
pass


# In[11]:


data.corr()


# In[12]:


plt.plot(data['tenure'], data['MonthlyCharges'], 'o', alpha = 0.01)


# In[13]:


data['Churn'] == 'Yes'


# In[14]:


plt.figure(figsize = (10, 6))

plt.scatter(data[data['Churn'] == 'Yes']['tenure'],
            data[data['Churn'] == 'Yes']['MonthlyCharges'],
            alpha = 0.15,
            label = 'Yes',
            color = 'b')

plt.scatter(data[data['Churn'] == 'No']['tenure'],
            data[data['Churn'] == 'No']['MonthlyCharges'],
            alpha = 0.05,
            label = 'No',
            color = 'r')

plt.xlabel('tenure')
plt.xticks(range(17))
plt.xlim(0, 17)
plt.ylabel('MonthlyCharges')
plt.legend()
plt.grid()


# In[15]:


np.random.seed(0)
tenure_rnd = data['tenure'] + np.random.rand(data.shape[0]) - .5

plt.figure(figsize = (10, 8))

plt.scatter(tenure_rnd[data['Churn'] == 'Yes'],
            data[data['Churn'] == 'Yes']['MonthlyCharges'],
            alpha = 0.15,
            label = 'Yes',
            color = 'b')

plt.scatter(tenure_rnd[data['Churn'] == 'No'],
            data[data['Churn'] == 'No']['MonthlyCharges'],
            alpha = 0.15,
            label = 'No',
            color = 'r')

plt.xlabel('tenure')
plt.xticks(range(17))
plt.xlim(0, 17)
plt.ylabel('MonthlyCharges')
plt.title('Churn Analysis')
plt.grid()


# In[16]:


plt.figure(figsize = (10, 8))

plt.scatter(data[data['Churn'] == 'No']['tenure'],
            data[data['Churn'] == 'No']['MonthlyCharges'],
            alpha = 0.15,
            label = 'No',
            color = 'r')

plt.scatter(data[data['Churn'] == 'Yes']['tenure'],
            data[data['Churn'] == 'Yes']['MonthlyCharges'],
            alpha = 0.15,
            label = 'Yes',
            color = 'b')

plt.xlabel('tenure')
plt.ylabel('MonthlyCharges')
plt.grid()


# In[17]:


data.describe(include = ['object'])


# In[18]:


data.describe(include = 'all')


# In[19]:


data['PhoneService'].unique()


# In[20]:


data['PaymentMethod'].unique()


# In[21]:


data = data.drop('TotalCharges', axis = 1)           .drop('customerID', axis = 1)


# Убираем TotalCharges столбец, потому что у него сильная корреляция с MonthlyCharges.

# ## Обработка пропущенных значений.
# Заполняем пропущенные значения средними значениями по столбцу.

# In[22]:


#categorical_columns = []
#numerical_columns = []
#for c in data.columns:
#    if data[c].dtype.name == 'object':
#        categorical_columns.append(c)
#    else:
#        numerical_columns.append(c)


# In[23]:


categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print(categorical_columns)
print(numerical_columns)


# In[24]:


for c in categorical_columns:
    print(c, data[c].unique())


# In[25]:


data = data.fillna(data.median(axis = 0), axis = 0)


# In[26]:


data[numerical_columns].count(axis = 0)


# In[27]:


data[categorical_columns].count(axis = 0)


# In[28]:


data_describe = data.describe(include = [object])
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])


# ### Векторизация

# In[29]:


binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(binary_columns, nonbinary_columns)


# In[30]:


data['gender'].unique()


# In[31]:


data.at[data['gender'] == 'Male', 'gender'] = 1
data.at[data['gender'] == 'Female', 'gender'] = 0
data['gender'].describe()


# In[32]:


data['Dependents'].unique()


# In[33]:


data.at[data['Partner'] == 'Yes', 'Partner'] = 1
data.at[data['Partner'] == 'No', 'Partner'] = 0
data.at[data['Dependents'] == 'Yes', 'Dependents'] = 1
data.at[data['Dependents'] == 'No', 'Dependents'] = 0
data.at[data['PhoneService'] == 'Yes', 'PhoneService'] = 1
data.at[data['PhoneService'] == 'No', 'PhoneService'] = 0
data.at[data['PaperlessBilling'] == 'Yes', 'PaperlessBilling'] = 1
data.at[data['PaperlessBilling'] == 'No', 'PaperlessBilling'] = 0
data['Partner'].describe()


# In[34]:


data['Partner'].unique()[0]


# In[35]:


data['MultipleLines'].unique()


# In[36]:


data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print(data_nonbinary.columns)


# In[37]:


data_numerical = data[numerical_columns]
data_numerical.describe()


# - `Выполняем нормализацию`

# In[38]:


data_numerical = (data_numerical - data_numerical.mean(axis = 0))/data_numerical.std(axis = 0)


# In[39]:


data = pd.concat((data_numerical, data_nonbinary, data[binary_columns]), axis = 1)
print(data.shape)
#print data.columns


# In[40]:


data.describe()


# In[41]:


X = data.drop(('Churn'), axis = 1) # выбрасываем столбец 'Churn'
y = data['Churn']
feature_names = X.columns
#print feature_names


# In[42]:


type(feature_names)


# In[43]:


print(X.shape)
print(y.shape)
N, d = X.shape


# In[44]:


type(y)


# In[45]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 
print (N_train, N_test)


# Разбили выборку в пропорции 30 на 70, где 30% тестовая выборка.

# # $k$NN

# In[46]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
#knn.set_params(n_neighbors=5)
knn.fit(X_train, y_train)


# Запускаем алгоритм k ближайших значений со стоковыми параметрами для обучающей выборки.

# In[47]:


knn


# In[48]:


y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test  = np.mean(y_test  != y_test_predict)

print(err_train, err_test)


# In[49]:


err_test = 1 - knn.score(X_test, y_test)


# #### Confusion matrix 

# In[50]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_test_predict))


# ### Подбор параметров

# In[51]:


from sklearn.grid_search import GridSearchCV
n_neighbors_array = [1, 3, 5, 7, 10, 15]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
grid.fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print (best_cv_err, best_n_neighbors)


# В результате наименьшая ошибка получается при k=10 соседей.

# In[52]:


knn = KNeighborsClassifier(n_neighbors = best_n_neighbors).fit(X_train, y_train)

err_train = np.mean(y_train != knn.predict(X_train))
err_test  = np.mean(y_test  != knn.predict(X_test))

print(err_train, err_test)


# В результате ошибка на тестовой выборке уменьшилась на 0.02.

# In[53]:


k = range(1, 22, 3)
list_err_train = []
list_err_test = []
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    list_err_train.append(np.mean(y_train != knn.predict(X_train)))
    list_err_test.append(np.mean(y_test  != knn.predict(X_test)))
    print(i)


# In[54]:


plt.plot(k, list_err_test, label='test')
plt.plot(k, list_err_train, label='train')
plt.xlabel('k')
plt.ylabel('error_value')
plt.legend(loc='lower right')
plt.grid()


# ## Нейронные сети

# ### Обучение

# In[55]:


from sklearn.neural_network import MLPClassifier


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[69]:


mlp_model = MLPClassifier(hidden_layer_sizes = (100,), solver = 'lbfgs', 
                          activation = 'logistic', random_state = 10)
mlp_model.fit(X_train, y_train)


# ### Качество классификации

# In[70]:


y_train_pred = mlp_model.predict(X_train)plt.semilogx(alpha_arr, train_err, 'b-o', label = 'test')
plt.semilogx(alpha_arr, test_err, 'r-o', label = 'train')
plt.xlim([np.max(alpha_arr), np.min(alpha_arr)])
plt.title('Error vs. alpha')
plt.xlabel('alpha')
plt.ylabel('error')
plt.legend()
pass
y_test_pred = mlp_model.predict(X_test)


# In[71]:


print(mlp_model.score(X_train, y_train), mlp_model.score(X_test, y_test))


# In[72]:


print(1 - mlp_model.score(X_test, y_test))


# ## Подбираем параметры

# In[73]:


alpha_arr = np.logspace(-3, 2, 21)
test_err = []
train_err = []
for alpha in alpha_arr:
    mlp_model = MLPClassifier(alpha = alpha, hidden_layer_sizes = (100,), 
                              solver = 'lbfgs', activation = 'logistic', random_state = 10)
    mlp_model.fit(X_train, y_train)

    y_train_pred = mlp_model.predict(X_train)
    y_test_pred = mlp_model.predict(X_test)
    train_err.append(np.mean(y_train != y_train_pred))
    test_err.append(np.mean(y_test != y_test_pred))


# In[74]:


plt.semilogx(alpha_arr, train_err, 'b-o', label = 'test')
plt.semilogx(alpha_arr, test_err, 'r-o', label = 'train')
plt.xlim([np.max(alpha_arr), np.min(alpha_arr)])
plt.title('Error vs. alpha')
plt.xlabel('alpha')
plt.ylabel('error')
plt.legend()
pass


# С уменьшением коэффициента регуляризатора ошибка на тестовой выб  уменьшается. При большом значении альфа модель сглаживается и недообучается.

# ##### Minimal error value
# 

# In[75]:


print(np.min(train_err), np.min(test_err))


# In[76]:


alpha_opt = alpha_arr[test_err == np.min(test_err)]
print(alpha_opt)


# ### Как ведет себя классификатор при оптимальном значении альфа:

# In[77]:


mlp_model = MLPClassifier(alpha = alpha_opt, hidden_layer_sizes = (100,),
                          solver = 'lbfgs', activation = 'logistic', random_state = 10)
mlp_model.fit(X_train, y_train)

print(np.min(train_err), np.min(test_err))


# Подбираем чисто нейронов:

# In[79]:


hidden_layer_sizes_arr = np.arange(50,300,50)
hidden_layer_sizes_arr
test_err_hidden_layer = []
train_err_hidden_layer = []
for hidden_layer_sizes in hidden_layer_sizes_arr:
    mlp_model = MLPClassifier(alpha = alpha_opt, hidden_layer_sizes = hidden_layer_sizes, 
                              solver = 'lbfgs', activation = 'logistic', random_state = 10)
    mlp_model.fit(X_train, y_train)

    y_train_pred = mlp_model.predict(X_train)
    y_test_pred = mlp_model.predict(X_test)
    train_err_hidden_layer.append(np.mean(y_train != y_train_pred))
    test_err_hidden_layer.append(np.mean(y_test != y_test_pred))


# In[80]:


plt.semilogx(hidden_layer_sizes_arr, train_err_hidden_layer, 'b-o', label = 'test')
plt.semilogx(hidden_layer_sizes_arr, test_err_hidden_layer, 'r-o', label = 'train')
plt.xlim([np.max(hidden_layer_sizes_arr), np.min(hidden_layer_sizes_arr)])
plt.title('Error vs. hidden_layer_sizes')
plt.xlabel('hidden_layer_sizes')
plt.ylabel('error')
plt.legend()
pass


# Ошибка при оптимальном колличестве нейронов:

# In[81]:


print(np.min(train_err_hidden_layer), np.min(test_err_hidden_layer))


# In[82]:


hidden_layer = hidden_layer_sizes_arr[test_err_hidden_layer == np.min(test_err_hidden_layer)]
print(hidden_layer)


# ### Матрица рассогласования

# In[83]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test_pred, y_test))


# In[85]:


M = confusion_matrix(y_test_pred, y_test)
M = np.sqrt(M)
plt.imshow(M, interpolation = 'nearest')
plt.set_cmap('binary')
plt.xticks(range(2))
plt.yticks(range(2))
plt.xlabel("true label")
plt.ylabel("predicted label")
plt.colorbar()
pass


# Другие метрики качества:

# In[86]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))


# Так же для оценки качества можно построить рок-кривую, которая наиболее наглядно отразит качество классификации.
