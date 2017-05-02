import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Train/test split
SPLIT = 0.7

# Number of sample predictions to show
SAMPLE = 20

# ------------------------------------------------------------------------------
# Get data
# ------------------------------------------------------------------------------

# Get data field
df = pd.read_csv("happiness.csv")

# Add bias and shuffle
data = df.as_matrix()
n = len(data)
data = np.c_[np.ones(n), data]
np.random.shuffle(data)

# Get train test split
train = data[:int(n * SPLIT)]
test = data[int(n * SPLIT):]
x_train = train[:, :-1]
y_train = train[:, [-1]]
x_test = test[:, :-1]
y_test = test[:, [-1]]

# ------------------------------------------------------------------------------
# Run algorithm
# ------------------------------------------------------------------------------

# Train
theta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
h = lambda x: x.dot(theta)

# Get prediction
y_pred = h(x_test)

# Get average error
avg_error = (sum((y_pred - y_test) ** 2) / (2.0 * len(y_pred)))[0]

# ------------------------------------------------------------------------------
# Show results
# ------------------------------------------------------------------------------

# Print data field
print df
print "\n"

# Print sample predictions against test outputs
print np.c_[y_test, y_pred][:SAMPLE]
print "\n"

# Print average error
print avg_error

# Plot test outputs in red and predictions in blue against first input feature
plt.plot(x_test[:, 1], y_test[:, 0], 'ro')
plt.plot(x_test[:, 1], y_pred[:, 0], 'bo')
plt.show()
