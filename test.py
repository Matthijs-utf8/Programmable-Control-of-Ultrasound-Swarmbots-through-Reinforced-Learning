import matplotlib.pyplot as plt
import numpy as np

# test = np.zeros(shape=(100, 100))
#
# def return_as_exp(index_1, index_2, exp_max_1=test.shape[0]/2, exp_max_2=test.shape[1]/2, exponent=2):
#     return exponent ** ( (1 / (1 + np.abs(exp_max_1 - index_1) )  +
#                          (1 / (1 + np.abs(exp_max_2 - index_2) ) ) ) / 2 )  / exponent

func = lambda x: (1 / (1 + np.abs(50 - x) ) ) ** (1/2)
test_1 =
test_2 =
print(test_1.shape, test_2.shape)
print(.shape)


# plt.imshow(test)
# plt.show()