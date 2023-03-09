const input = [[1, 0, 0, 1]];

// hidden  layer will have 5 neurons ( 4 x 5 matrix)

// bias for hidden layer length 5

// output layer will have 3 neurons ( 5 x 3 matrix)

// bias length 3

function initializingLayers(inputSize, hiddenLayerSize, outputSize) {
    const hiddenLayerWeights = Array.from({ length: inputSize }, () => {
      return Array.from({ length: hiddenLayerSize }, () => Math.random());
    });
    const hiddenLayerBiases = Array.from({ length: hiddenLayerSize }, () => Math.random());
    const outputLayerWeights = Array.from({ length: hiddenLayerSize }, () => {
        return Array.from({ length: outputSize }, () => Math.random());
    });
    const outputLayerBiases = Array.from({ length: outputSize }, () => Math.random());

    return [
        hiddenLayerWeights,
        hiddenLayerBiases,
        outputLayerWeights,
        outputLayerBiases
    ]
}

function forwardPropagation(input, hiddenWeights, hiddenBias, outputWeight, outputBias) {
    const multiply = matrixMultiply(input, hiddenWeights);
    const addHiddenBias = addBias(multiply, hiddenBias);
    const hiddenLayerValues = activate(addHiddenBias, sigmoid)

    const multiply2 = matrixMultiply(hiddenLayerValues, outputWeight);
    const addOutputBias = addBias(multiply2, outputBias);
    const outputLayerValues = activate(addOutputBias, sigmoid);

    return {
        hiddenLayerValues,
        outputLayerValues
    }
}


// where x = [a, b, c] 
// x2 = [d, e, f]
// product = [a*d + b*e + c*f]

function dotProduct(x, x2) {
    let sum = 0;

    for (let i = 0; i < x.length; i++) {
        sum = (x[i] * x2[i]) + sum
    }
    return sum;
};

function transpose(matrix) {
    const transposedMatrix = Array.from({ length: matrix[0].length }, () => new Array(matrix.length));
    matrix.forEach((row, i) => row.forEach((cell, j) => transposedMatrix[j][i] = cell));
    return transposedMatrix;
}

function matrixMultiply(A, B) {
    // m x n * n x z
    let columnsA = A[0].length;
    let rowsA = A.length;
    let columnsB = B[0].length;
    let rowsB = B.length;
    const result = []

    let B1 = transpose(B);

    if (columnsA !== rowsB ) {
        throw new Error('incompatible matrix operation')
    } else {
        for (let i = 0; i < rowsA; i++) {
            let row = [];
            for (let j = 0; j< columnsB; j++) {
                let element = dotProduct(A[i], B1[j]);
                row.push(element)
            }
            result.push(row);
        }
        return result;
    }

}

function addBias(matrix, bias) {
    if (matrix[0].length !== bias.length) {
        throw new Error('cannot add bias')
    } else {
        let result = []
        for (let i = 0; i < matrix.length; i++) {
            let row = [];
            for (let j = 0; j < matrix[0].length; j++) {
                let sum = matrix[i][j] + bias[j]
                row.push(sum);
            }
            result.push(row)
        }
        return result;
    }
}

function sigmoid(x) {
    // return (x * x)
    return 1 / (1 + Math.exp(-x));
}

function activate(biasedMatrix, activationFunction) {
    let result = [];
    for (let i = 0; i < biasedMatrix.length; i++) {
        let row = [];
        for (let j = 0; j < biasedMatrix[0].length; j++) {
            let element = biasedMatrix[i][j];
            let activatedElement = activationFunction(element);
            row.push(activatedElement)
        }
        result.push(row)
    }
    return result;
}


const initialValues = initializingLayers(4, 5, 3);
let result = forwardPropagation(input, ...initialValues);
console.log(result)