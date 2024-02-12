import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load in the "fisheriris" dataset.
iris_data = pd.read_excel('Proj1DataSet.xlsx')
iris_data.columns = ['SepL', 'SepW', 'PetL', 'PetW', 'Class']

# Map class names to numerical value
class_mapping = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
iris_data['Class'] = iris_data['Class'].map(class_mapping)

# Load variables with each individual feature data
SepL = iris_data['SepL']
SepW = iris_data['SepW']
PetL = iris_data['PetL']
PetW = iris_data['PetW']
Class = iris_data['Class']

# Create a list of features
features = [SepL, SepW, PetL, PetW]

# Create variables for data separated by class
setosa = iris_data.query('Class == 1')
versicolor = iris_data.query('Class == 2')
virginica = iris_data.query('Class == 3')

# Same for all classes, used for within-class and between-class variance.
prior_probability = 0.33

### Data Analytics Code Block

# Find the minimums
for feature in features:
    min = np.min(feature)
    print(f'Minimum for {feature.name} is: {min}')
print()

# Find the maximums
for feature in features:
    max = np.max(feature)
    print(f'Maximum for {feature.name} is: {max}')
print()

# Find the means
for feature in features:
    mean = np.mean(feature)
    print(f'Mean for {feature.name} is: {mean:.2f}')
print()

# Find the variances
for feature in features:
    var = np.var(feature)
    print(f'Variance for {feature.name} is: {var:.2f}')
print()
    
sw_SepL = np.var(setosa['SepL']) * prior_probability + np.var(versicolor['SepL']) * prior_probability + np.var(virginica['SepL']) * prior_probability
sw_SepW = np.var(setosa['SepW']) * prior_probability + np.var(versicolor['SepW']) * prior_probability + np.var(virginica['SepW']) * prior_probability
sw_PetL = np.var(setosa['PetL']) * prior_probability + np.var(versicolor['PetL']) * prior_probability + np.var(virginica['PetL']) * prior_probability
sw_PetW = np.var(setosa['PetW']) * prior_probability + np.var(versicolor['PetW']) * prior_probability + np.var(virginica['PetW']) * prior_probability

sw_set = [sw_SepL, sw_SepW, sw_PetL, sw_PetW]

# Find the within class variance (sw)
print(f'Within-class variance for SepL is: {sw_SepL:.3f}')
print(f'Within-class variance for SepW is: {sw_SepW:.3f}')
print(f'Within-class variance for PetL is: {sw_PetL:.3f}')
print(f'Within-class variance for PetW is: {sw_PetW:.3f}')
print()

sb_SepL = (1/3) * (setosa['SepL'].mean() - SepL.mean())**2 + (1/3) * (versicolor['SepL'].mean() - SepL.mean())**2 + (1/3) * (virginica['SepL'].mean() - SepL.mean())**2
sb_SepW = (1/3) * (setosa['SepW'].mean() - SepW.mean())**2 + (1/3) * (versicolor['SepW'].mean() - SepW.mean())**2 + (1/3) * (virginica['SepW'].mean() - SepW.mean())**2
sb_PetL = (1/3) * (setosa['PetL'].mean() - PetL.mean())**2 + (1/3) * (versicolor['PetL'].mean() - PetL.mean())**2 + (1/3) * (virginica['PetL'].mean() - PetL.mean())**2
sb_PetW = (1/3) * (setosa['PetW'].mean() - PetW.mean())**2 + (1/3) * (versicolor['PetW'].mean() - PetW.mean())**2 + (1/3) * (virginica['PetW'].mean() - PetW.mean())**2

sb_set = [sb_SepL, sb_SepW, sb_PetL, sb_PetW]

# Find the between class variance (sb)
print(f'Between-class variance for SepL is: {sb_SepL:.3f}')
print(f'Between-class variance for SepW is: {sb_SepW:.3f}')
print(f'Between-class variance for PetL is: {sb_PetL:.3f}')
print(f'Between-class variance for PetW is: {sb_PetW:.3f}')
print()
### End

### Plotting SepL/SepW and PetL/PetW Code Block

# Plot sepal length and width of the three classes
plt.figure(figsize=(8,6))
plt.scatter(setosa['SepL'], setosa['SepW'], c='red', label='Setosa')
plt.scatter(versicolor['SepL'], versicolor['SepW'], c='green', label='Versicolor')
plt.scatter(virginica['SepL'], virginica['SepW'], c='blue', label='Virginica')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Width')
plt.legend()

# Plot petal length and width of the three classes
plt.figure(figsize=(8,6))
plt.scatter(setosa['PetL'], setosa['PetW'], c='red', label='Setosa')
plt.scatter(versicolor['PetL'], versicolor['PetW'], c='green', label='Versicolor')
plt.scatter(virginica['PetL'], virginica['PetW'], c='blue', label='Virginica')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Width')
plt.legend()
## End

### Correlation Matrix Heatmap Code Block

# Initialize data for correlation matrix
iris_data = iris_data[['SepL', 'SepW', 'PetL', 'PetW', 'Class']]

# Apply np.corrcoef on data
correlation_matrix = np.corrcoef(iris_data, rowvar=False)

# Plot the correlation matrix
plt.figure(figsize=(8,6))
plt.imshow(correlation_matrix, cmap='jet')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Matrix Heatmap')
plt.xticks(range(len(iris_data.columns)), iris_data.columns)
plt.yticks(range(len(iris_data.columns)), iris_data.columns)
### End

### Plotting all features against class.

# Plots Sepl vs Class
plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.scatter(SepL, Class, marker='x', c='red', linewidths=.5)
plt.title('SepL vs Class')
plt.xlim(0,8)
plt.ylim(1,3)
plt.yticks([1,1.5,2,2.5,3])

# Plots SepW vs Class
plt.subplot(2,2,2)
plt.scatter(SepW, Class, marker='x', c='red', linewidths=.5)
plt.title('SepW vs Class')
plt.xlim(0,8)
plt.ylim(1,3)
plt.yticks([1,1.5,2,2.5,3])

# Plots PetL vs Class
plt.subplot(2,2,3)
plt.scatter(PetL, Class, marker='x', c='red', linewidths=.5)
plt.title('PetL vs Class')
plt.xlim(0,8)
plt.ylim(1,3)
plt.yticks([1,1.5,2,2.5,3])

# Plots PetW vs Class
plt.subplot(2,2,4)
plt.scatter(PetW, Class, marker='x', c='red', linewidths=.5)
plt.title('PetW vs Class')
plt.xlim(0,8)
plt.ylim(1,3)
plt.yticks([1,1.5,2,2.5,3])
### End

# Set the seed!
np.random.seed(11585932)

# Batch Perceptron Algorithm for four features

def batch_perceptron_four(X, rho=.001, max_epochs=1000):
    # Initialize weights
    w = np.random.rand(5,1)

    for epoch in range(max_epochs):
        
        # Reset counters for each epoch
        misclassified = 0
        misclass_sum = np.zeros((5,1))

        # Iterate through each data point
        for k in range(len(X)):
            # Check if misclassified
            if (w.transpose() @ X[k]) <= 0:
                # Update misclassified counter
                misclassified += 1
                # Update sum
                misclass_sum += X[k].reshape(5,1)   
        
        # Update weight vector with sum of misclassified data points
        w += rho * misclass_sum.reshape(5,1)

        # Check if we converged, no misclassifications.    
        if misclassified == 0:
            # Print out number of epochs and weight vector
            print(f"Batch Perceptron Misclassifications: {misclassified}")
            print(f"Converged in {epoch + 1} epochs")
            print(f"BP Weight vector is: {w.transpose()[0]}")
            break
        
    if misclassified > 0:
        print(f"Batch Perceptron Misclassifications: {misclassified}")
        print(f"Did Not Converge After {epoch + 1} epochs")  
        print(f"BP Weight vector is: {w.transpose()[0]}")  
        return w
### End
        
# Batch Perceptron Algorithm for two features

def batch_perceptron_two(X, rho=.001, max_epochs=1000):
    # Initialize weights
    w = np.random.rand(3,1)
    
    for epoch in range(max_epochs):
        
        # Reset counters for each epoch
        misclassified = 0
        misclass_sum = np.zeros((3,1))

        # Iterate through each data point
        for k in range(len(X)):
            # Check if misclassified
            if (w.transpose() @ X[k].reshape(3,1)) <= 0:
                # Update misclassified counter
                misclassified += 1
                # Update sum
                misclass_sum += X[k].reshape(3,1)  

        # Update weight vector with sum of misclassified data points
        w += rho * misclass_sum.reshape(3,1)

        # Check if we converged, no misclassifications.    
        if misclassified == 0:
            # Print out number of epochs and weight vector
            print(f"Batch Perceptron Misclassifications: {misclassified}")
            print(f"BP Converged in {epoch + 1} epochs")
            print(f"BP Weight vector is: {w.transpose()[0]}")
            return w
        
    if misclassified > 0:
        print(f"Batch Perceptron Misclassifications: {misclassified}")
        print(f"Did Not Converge After {epoch + 1} epochs.")
        print(f"BP Weight vector is: {w.transpose()[0]}")    
        return w
    
### End
        

# Least Squares Technique Algorithm

def least_squares(X, t):
    # Calculate weights using least squares
    w = np.linalg.pinv(X) @ t
    print(f"LS Weight vector is: {w.transpose()[0]}")

    misclassified = 0
    for k in range(len(X)):
        if np.sign(w.transpose() @ X[k]) != t[k]:
            misclassified += 1
    print(f"Least Squares Misclassifications: {misclassified}\n")
    return w
### End

# Setosa vs Versi+Virgi - All Features ###########################################################
    
# Initialize X
Setosa_VV_All = iris_data[['SepL', 'SepW', 'PetL', 'PetW', 'Class']].to_numpy()
# Setosa if pos, neg if not.
Setosa_VV_labels = ((iris_data['Class'] == 1).astype(int) * 2 - 1).to_numpy().reshape(150,1)
# Identify indices of non-Setosa instances
not_Setosa_indices = np.where(iris_data['Class'] != 1)[0]
# Remove the last column, Class, from X
Setosa_VV_All = Setosa_VV_All[:, :-1]
# Add 1's for bias term
Setosa_VV_All = np.hstack((Setosa_VV_All, np.ones((Setosa_VV_All.shape[0], 1))))
# Multiply features of non-Setosa instances by -1
Setosa_VV_All_Neg = np.copy(Setosa_VV_All)
Setosa_VV_All_Neg[not_Setosa_indices] *= -1

print("Setosa vs All: ")
w_BP_Setosa_All = batch_perceptron_four(Setosa_VV_All_Neg)
w_LS_Setosa_All = least_squares(Setosa_VV_All, Setosa_VV_labels)
### End

# Setosa vs Versi+Virgi - Features 3 & 4 ############################################################
    
# Initialize X
Setosa_VV_34 = iris_data[['PetL', 'PetW', 'Class']].to_numpy()
# Setosa if pos, neg if not.
Setosa_VV_labels = ((iris_data['Class'] == 1).astype(int) * 2 - 1).to_numpy().reshape(150,1)
# Identify indices of non-Setosa instances
not_Setosa_indices = np.where(iris_data['Class'] != 1)[0]
# Remove the last column, Class, from X
Setosa_VV_34 = Setosa_VV_34[:, :-1]
# Add 1's for bias term
Setosa_VV_34 = np.hstack((Setosa_VV_34, np.ones((Setosa_VV_34.shape[0], 1))))
# Multiply features of non-Setosa instances by -1
Setosa_VV_34_Neg = np.copy(Setosa_VV_34)
Setosa_VV_34_Neg[not_Setosa_indices] *= -1

print("Setosa vs 3 & 4: ")
w_BP_Setosa_34 = batch_perceptron_two(Setosa_VV_34_Neg)
w_LS_Setosa_34 = least_squares(Setosa_VV_34, Setosa_VV_labels)

# Create versi+virgi 
versi_virgi = iris_data.query('Class == 2 or Class == 3')

# Set up 
x = np.linspace(0, 7, 100)
y_batch_perceptron = -(w_BP_Setosa_34[0] * x + w_BP_Setosa_34[2]) / w_BP_Setosa_34[1]
y_least_squares = -(w_LS_Setosa_34[0] * x + w_LS_Setosa_34[2]) / w_LS_Setosa_34[1]

plt.figure(figsize=(8,6))
plt.scatter(setosa['PetL'], setosa['PetW'], marker='o', c='red', label='Setosa')
plt.scatter(versi_virgi['PetL'], versi_virgi['PetW'], marker='x', linewidths=.5, c='blue', label='Versi+Virgi')
plt.plot(x, y_batch_perceptron, c='green', label='Batch Perceptron Decision Boundary')
plt.plot(x, y_least_squares, c='orange', label='Least Squares Decision Boundary')
plt.xlim(0.5, 7)
plt.ylim(-0.5, 3)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Setosa vs Versi+Virgi, Features 3 & 4')
plt.legend()

# Virgi vs Versi+Setosa - All Features ###########################################################
    
# Initialize X
Virgi_VS_All = iris_data[['SepL', 'SepW', 'PetL', 'PetW', 'Class']].to_numpy()
# Virginica if pos, neg if not.
Virgi_VS_labels = ((iris_data['Class'] == 3).astype(int) * 2 - 1).to_numpy().reshape(150,1)
# Identify indices of non-Virginica instances
not_Virgi_indices = np.where(iris_data['Class'] != 3)[0]
# Remove the last column, Class, from X
Virgi_VS_All = Virgi_VS_All[:, :-1]
# Add 1's for bias term
Virgi_VS_All = np.hstack((Virgi_VS_All, np.ones((Virgi_VS_All.shape[0], 1))))
# Multiply features of non-Virginica instances by -1
Virgi_VS_All_Neg = np.copy(Virgi_VS_All)
Virgi_VS_All_Neg[not_Setosa_indices] *= -1

print("Virginica vs All: ")
w_BP_Virgi_All = batch_perceptron_four(Virgi_VS_All_Neg)
w_LS_Virgi_All = least_squares(Virgi_VS_All, Virgi_VS_labels)
### End

# Virgi vs Versi+Setosa - Features 3 & 4 ###########################################################

# Initialize X
Virgi_VS_34 = iris_data[['PetL', 'PetW', 'Class']].to_numpy()
# Virginica if pos, neg if not.
Virgi_VS_labels = ((iris_data['Class'] == 3).astype(int) * 2 - 1).to_numpy().reshape(150,1)
# Identify indices of non-Virginica instances
not_Virgi_indices = np.where(iris_data['Class'] != 3)[0]
# Remove the last column, Class, from X
Virgi_VS_34 = Virgi_VS_34[:, :-1]
# Add 1's for bias term
Virgi_VS_34 = np.hstack((Virgi_VS_34, np.ones((Virgi_VS_34.shape[0], 1))))
# Multiply features of non-Virginica instances by -1
Virgi_VS_34_Neg = np.copy(Virgi_VS_34)
Virgi_VS_34_Neg[not_Virgi_indices] *= -1

print("Virgi vs 3 & 4: ")
w_BP_Virgi_34 = batch_perceptron_two(Virgi_VS_34_Neg)
w_LS_Virgi_34 = least_squares(Virgi_VS_34, Virgi_VS_labels)

# Create versi+virgi 
setosa_versi = iris_data.query('Class == 1 or Class == 2')

# Set up 
x = np.linspace(0, 7, 100)
y_batch_perceptron = -(w_BP_Virgi_34[0] * x + w_BP_Virgi_34[2]) / w_BP_Virgi_34[1]
y_least_squares = -(w_LS_Virgi_34[0] * x + w_LS_Virgi_34[2]) / w_LS_Virgi_34[1]

plt.figure(figsize=(8,6))
plt.scatter(virginica['PetL'], virginica['PetW'], marker='o', c='red', label='Virginica')
plt.scatter(setosa_versi['PetL'], setosa_versi['PetW'], marker='x', linewidths=.5, c='blue', label='Versi+Setosa')
plt.plot(x, y_batch_perceptron, c='green', label='Batch Perceptron Decision Boundary')
plt.plot(x, y_least_squares, c='orange', label='Least Squares Decision Boundary')
plt.xlim(0.5, 7)
plt.ylim(-0.5, 3)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Virgi vs Versi+Setosa, Features 3 & 4')
plt.legend()
### End

# Setosa vs Versi vs Virgi - Features 3 & 4 ###########################################################

# Initialize X
X = iris_data[['PetL', 'PetW']].to_numpy()
X = np.hstack((X, np.ones((X.shape[0], 1))))
t = np.zeros((150,3))
t[:50, 0] = 1
t[50:100, 1] = 1
t[100:150, 2] = 1

# Initialize misclass
misclassified = 0

# Apply least squares technique
w = np.linalg.pinv(X) @ t

# Look for misclassifications
for k in range(0,50):
    if w.transpose()[0] @ X[k] < w.transpose()[1] @ X[k] or w.transpose()[0] @ X[k] < w.transpose()[2] @ X[k]:
        misclassified += 1

for k in range(50,100):
    if w.transpose()[1] @ X[k] < w.transpose()[0] @ X[k] or w.transpose()[1] @ X[k] < w.transpose()[2] @ X[k]:
        misclassified += 1

for k in range(100,150):
    if w.transpose()[2] @ X[k] < w.transpose()[1] @ X[k] or w.transpose()[2] @ X[k] < w.transpose()[0] @ X[k]:
        misclassified += 1

print("Multiclass LS:")
print('Multiclass Misclassifications: ', misclassified)
print(f"Multiclass LS Weight vector is: {w.transpose()[0]}")

# Set up 
x = np.linspace(0, 7, 150)
y_setosa_versicolor = -((w[0][0]-w[0][1]) * x + (w[2][0]-w[2][1])) / (w[1][0]-w[1][1])
y_setosa_virginica = -((w[0][0]-w[0][2]) * x + (w[2][0]-w[2][2])) / (w[1][0]-w[1][2])
y_versicolor_virginica = -((w[0][1]-w[0][2]) * x + (w[2][1]-w[2][2])) / (w[1][1]-w[1][2])

plt.figure(figsize=(8,6))

plt.scatter(setosa['PetL'], setosa['PetW'], c='red', label='Setosa')
plt.scatter(versicolor['PetL'], versicolor['PetW'], c='green', label='Versicolor')
plt.scatter(virginica['PetL'], virginica['PetW'], c='blue', label='Virginica')

plt.plot(x, y_setosa_versicolor, c='purple', label='Setosa Vs Versicolor Decision Boundary')
plt.plot(x, y_setosa_virginica, c='pink', label='Setosa Vs Virginica Decision Boundary')
plt.plot(x, y_versicolor_virginica, c='black', label='Versicolor Vs Virginica Decision Boundary')
plt.xlabel('Feature 3 (Petal Length)')
plt.ylabel('Feature 4 (Petal Width)')
plt.title('Multiclass Classification using Least Squares Technique')
plt.legend()
plt.xlim(0.5, 7)
plt.ylim(-0.5, 3)

plt.show()