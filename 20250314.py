import pandas as pd

# 用批量梯度下降(Batch Gradient Descent，BGD)做线性回归，因变量是FF三因子的里的市场收益，自变量是滞后一期的市场收益
# 读取数据
factors_ff3_monthly = pd.read_csv("data/ff3.csv", parse_dates=["Date"])

factors_ff3_monthly = factors_ff3_monthly.iloc[:1182,:]

# 计算滞后一期的市场收益
factors_ff3_monthly["Mkt-RF_lag1"] = factors_ff3_monthly["Mkt-RF"].shift(1)

factors_ff3_monthly = factors_ff3_monthly.dropna()
# 把factors_ff3_monthly转为浮点数
factors_ff3_monthly["Mkt-RF"] = factors_ff3_monthly["Mkt-RF"].astype(float)
factors_ff3_monthly["Mkt-RF_lag1"] = factors_ff3_monthly["Mkt-RF_lag1"].astype(float)


# 初始化参数
beta0 = 0
beta1 = 0
beta2 = 0
alpha = 0
# 学习率
learning_rate = 0.01
# 迭代次数
n_iterations = 1000
# 计算BGD
for i in range(n_iterations):
    # 计算残差
    factors_ff3_monthly["residual"] = factors_ff3_monthly["Mkt-RF"] - alpha - beta1 * factors_ff3_monthly["Mkt-RF_lag1"]
    # 计算梯度
    gradient_alpha = -2 * factors_ff3_monthly["residual"].mean()
    gradient_beta1 = -2 * (factors_ff3_monthly["residual"] * factors_ff3_monthly["Mkt-RF_lag1"]).mean()
    # 更新参数
    alpha = alpha - learning_rate * gradient_alpha
    beta1 = beta1 - learning_rate * gradient_beta1
# 输出结果

print(f"alpha: {alpha}")
print(f"beta1: {beta1}")

# 把原始的数据和拟合的线性回归方程画在一起
import matplotlib.pyplot as plt
plt.plot(factors_ff3_monthly["Mkt-RF"], label="Mkt-RF")
plt.plot(alpha + beta1 * factors_ff3_monthly["Mkt-RF_lag1"], label="Fitted")
plt.legend()
plt.show()





