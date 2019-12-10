import element_feature as ef
import combination_generator as cg
import preprocessing as pre
import gbr as g

cwd = 'D:\\PycharmProjects\\HOIP_bandgap_prediction\\data\\'
#ef.element_feature(cwd)
#cg.combination_generator(cwd)
#cg.unknown_combination_seperator(cwd)
X_train, X_test, y_train, y_test, predict_X = pre.preprocessing(cwd)
print(X_train)
g.gbr(X_train, X_test, y_train, y_test)