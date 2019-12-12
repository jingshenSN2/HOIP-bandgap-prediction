import element_feature as ef
import combination_generator as cg
import preprocessing as pre
import feature_selection as fs
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

#e.raw_data_describe(data_directory)
#e.pre_processing_data_describe(data_directory)

#X_train, X_test, y_train, y_test, predict_X, features = pre.preprocessing(data_directory)

feature_selection_gbr_reg_list = fs.feature_selector_ensemble('gbr', data_directory, model_directory)
feature_selection_rfr_reg_list = fs.feature_selector_ensemble('rfr', data_directory, model_directory)
feature_selection_abr_reg_list = fs.feature_selector_ensemble('abr', data_directory, model_directory)
feature_selection_etr_reg_list = fs.feature_selector_ensemble('etr', data_directory, model_directory)
#feature_selection_mlp_reg_list = fs.feature_selector_mlp(X_train, X_test, y_train, y_test, features, model_directory)
#X_train, X_test, y_train, y_test, predict_X, features = pre.preprocessing(data_directory)
#feature_selection_pca_reg_list = fs.feature_selector_pca(X_train, X_test, y_train, y_test, features, model_directory)
#g.gbr(X_train, X_test, y_train, y_test, features)
