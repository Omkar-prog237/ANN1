#Generate ANDNOT function using McCulloch-Pitts neural net.

def mcp_neuron(x1, x2):

    # weights
    w1 = 1      # for X1
    w2 = -1     # inhibitory input for X2

    # threshold
    theta = 1

    # net input
    net = x1 * w1 + x2 * w2

    # activation function (step)
    if net >= theta:
        return 1
    else:
        return 0


# Input combinations
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("X1  X2  Output")

for x1, x2 in inputs:
    y = mcp_neuron(x1, x2)
    print(x1, " ", x2, "   ", y)
