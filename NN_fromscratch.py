#Designate the number of digits
digits = 10

#Load the data again
mnist = tf.keras.datasets.mnist

#Split into train and test
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Regularize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

#Reshape the data
x_trains = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_tests = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

#Hotcode the y variable so it can go into the model
y_trains = np.array(pd.get_dummies(y_train).iloc[:,0])
y_tests = np.array(pd.get_dummies(y_test).iloc[:,0])

m = 60000
m_test = 10000

# Finally get to the reshaped train and test data
X_train, X_test = x_trains.T, x_tests.T
y_train, y_test = y_trains.reshape(1,60000), y_tests.reshape(1,10000)


#Create a loss function
def compute_loss (Y, Y_pred):
    L_sum = np.sum(np.multiply(Y, np.log(Y_pred)))
    m = Y.shape[1]
    L = -(1/m) * L_sum
    return L

#Define ReLU function
def ReLU(x):
    return x * (x > 0)

#Define sigmoid function
def sig(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

#Designate X and Y
X = X_train
Y = y_train


#Define parameters, including dimensions and learning rate
dimx = X_train.shape[0]
#Define a small learning rate to prevent from large steps and erratic behavior in the model
learning_rate = .05
#Hidden layer definition
nl = 100


#Initialize the weights
weight1 = np.random.randn(nl, dimx)
weight2 = np.random.randn(digits, nl)


#Define the biases
bias1 = np.zeros((nl, 1))
bias2 = np.zeros((digits, 1))



#Implement backpropagation algorithm
#Specify the intermediate values
for i in range(1000):
    L1 = (weight1 @ X) + bias1
    A1 = sig(L1)
    L2 = (weight2 @ A1) + bias2
    A2 = np.exp(L2) / np.sum(np.exp(L2), axis=0)
    
#Compute the loss function
    cost = compute_loss(Y, A2)
    
    derL2 = A2-Y
    derW2 = (1./m) * (derL2 @ A1.T)
    derb2 = (1./m) * np.sum(derL2, axis=1, keepdims=True)
    derA1 = (weight2.T @ derL2)
    
#Use sigmoid function
    derL1 = derA1 * sig(L1) * (1 - sig(L1))
    derW1 = (1./m) * (derL1 @ X.T)
    derb1 = (1./m) * np.sum(derL1, axis=1, keepdims=True)
    weight1 = weight1 - derW1 * learning_rate 
    weight2 = weight2 - derW2 * learning_rate 
    bias2 = bias2 - derb2 * learning_rate
    bias1 = bias1 - derb1 * learning_rate 
    
#Run and print out the number of epochs and final cost
    if (i % 100 == 0):
        print("epoch", i, "cost: ", cost)

print("final cost:", cost)


