import element_feature as ef
import combination_generator as cg
import preprocessing as pre
import feature_selection as fs
import EDA as e
import gbr as g

data_directory = 'D:\\PycharmProjects\\HOIP_bandgap_prediction\\data\\'
model_directory = 'D:\\PycharmProjects\\HOIP_bandgap_prediction\\model\\'
#ef.element_feature(data_directory)
#cg.combination_generator(data_directory)
#cg.unknown_combination_seperator(data_directory)

e.raw_data_describe(data_directory)
e.pre_processing_data_describe(data_directory)

#X_train, X_test, y_train, y_test, predict_X, features = pre.preprocessing(data_directory)

#feature_selection_gbr_reg_list = fs.feature_selector_gbr(X_train, X_test, y_train, y_test, features, model_directory)
#feature_selection_mlp_reg_list = fs.feature_selector_mlp(X_train, X_test, y_train, y_test, features, model_directory)
#X_train, X_test, y_train, y_test, predict_X, features = pre.preprocessing(data_directory)
#feature_selection_pca_reg_list = fs.feature_selector_pca(X_train, X_test, y_train, y_test, features, model_directory)
#g.gbr(X_train, X_test, y_train, y_test, features)
