import element_feature as ef
import combination_generator as cg
import preprocessing as pre
import feature_selection as fs
import test_ratio_selection as trs
import raw_plot as rp
import EDA as e
import gbr as g

data_directory = 'D:\\PycharmProjects\\HOIP_bandgap_prediction\\data\\'
model_directory = 'D:\\PycharmProjects\\HOIP_bandgap_prediction\\model\\'

#初始数据集去重
#ef.raw_drop_duplicates(data_directory, silent=0)

#分离ABX离子特征
#ef.element_feature(data_directory, silent=0)

#生成特征组合的全集和补集
#cg.combination_generator(data_directory, silent=0)
#cg.unknown_combination_seperator(data_directory, silent=0)

#初始数据集和归一化后初始数据集的特征统计描述
#e.raw_data_describe(data_directory)
#e.pre_processing_data_describe(data_directory)

#初始数据集特征对标签的分布
#rp.raw_feature_plot(data_directory, 'T_f')

#考察测试集比例对模型精度的影响
trs.ratio_test(data_directory)

#不同方法做每次舍弃最差的特征筛选
#feature_selection_gbr_reg_list = fs.feature_selector_ensemble('gbr', data_directory, model_directory)
#feature_selection_rfr_reg_list = fs.feature_selector_ensemble('rfr', data_directory, model_directory)
#feature_selection_abr_reg_list = fs.feature_selector_ensemble('abr', data_directory, model_directory)
#feature_selection_etr_reg_list = fs.feature_selector_ensemble('etr', data_directory, model_directory)

#考察PCA降维的影响
#feature_selection_mlp_reg_list = fs.feature_selector_mlp(X_train, X_test, y_train, y_test, features, model_directory)


#Gaussian Boosting R
#g.gbr(cwdm, features)

#Kernel Ridge R

#Support Vector R

#Gaussian Process R

#Decision Tree R

#MLP R
