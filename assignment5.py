import numpy as np # 1.16.4
import pandas as pd # 0.24.2
import matplotlib
import matplotlib.pyplot # 2.2.4
#Python 2.7
def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))
def sigmoid_derivative(x):
    return 0.005 * x * (1 - x )
def read_and_divide_into_train_and_test(csv_file):
    satir = csv_file.shape[0]
    corr = csv_file.head((satir*8)//10).drop("Code_number",axis=1).corr()
    training_inputs = csv_file.head((satir*8)//10).drop("Class", axis=1).drop("Code_number",axis=1).to_numpy()
    training_labels = csv_file.head((satir*8)//10)["Class"].to_numpy()
    training_labels = training_labels.reshape(training_labels.shape[0],-1)
    test_inputs = csv_file.tail((satir)-((satir*8)//10)).drop("Class", axis=1).drop("Code_number",axis=1).to_numpy()
    test_labels = csv_file.tail((satir)-((satir*8)//10))["Class"].to_numpy()    
    fig, ax = matplotlib.pyplot.subplots()
    a = ["Clump_Thickness","Uniformity_of_Cell_Size","Uniformity_of_Cell_Shape","Marginal_Adhesion","Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses"]
    ax.set_xticks(np.arange(len(a)))
    ax.set_yticks(np.arange(len(a)))    
    ax.set_xticklabels(a)
    ax.set_yticklabels(a)
    ax.imshow(corr)
    matplotlib.pyplot.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
    fig.tight_layout()
    x = ax.imshow(corr)
    fig.colorbar(x,ax=ax)
    matplotlib.pyplot.show()
    return training_inputs, training_labels, test_inputs, test_labels
def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    test_output = sigmoid(np.dot(test_inputs.astype(int),weights))
    test_predictions = (np.round(test_output))
    test_outputs = np.round(test_labels)
    for predicted_val, label in zip(test_predictions, test_outputs):
        if predicted_val == label:
            tp += 1
    accuracy = (tp / float(test_inputs.shape[0]))
    return accuracy
def plot_loss_accuracy(accuracy_array, loss_array):
    a = matplotlib.pyplot.plot(accuracy_array)
    matplotlib.pyplot.title("Accuracy")
    matplotlib.pyplot.show(a)
    b = matplotlib.pyplot.plot(loss_array)
    matplotlib.pyplot.title("Loss")
    matplotlib.pyplot.show(b)    
    return
def main():
    csv_file = pd.read_csv('./breast-cancer-wisconsin.csv')
    csv_file = csv_file.replace(to_replace="?", value=np.nan)
    csv_file.dropna(inplace=True)
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)
    for iteration in range(iteration_count):
        input = training_inputs
        training_outputs = sigmoid(np.dot(input.astype(int),weights))
        outputs = training_labels.astype(float)
        loss = outputs - training_outputs
        tuning = loss*sigmoid_derivative(training_outputs)
        weights = weights+np.dot(np.transpose(input.astype(int)),tuning)
        loss_array.append(np.mean(loss))
        accuracy_array.append(run_on_test_set(test_inputs,test_labels,weights))    
   
    plot_loss_accuracy(accuracy_array, loss_array)
if __name__ == '__main__':
    main()