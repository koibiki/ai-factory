import os
train_x_file = 'output/train_X_delete_date.csv'
test_x_file = 'output/test_X_delete_date.csv'
print(os.path.exists(train_x_file) and os.path.exists(test_x_file))