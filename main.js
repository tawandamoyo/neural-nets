const input = [[1], [0], [0], [1]];

// hidden  layer will have 5 neurons ( 4 x 5 matrix)

// bias for hidden layer length 5

// output layer will have 3 neurons ( 5 x 3 matrix)

// bias length 3

const initialValues = initializingLayers([4, 5, 5, 7, 5, 5, 6, 9, 5, 7, 9, 4, 8, 2]);
let result = forwardPropagation(input, initialValues);
console.log('untrained result: ', result[result.length - 1])

console.log('initial weights', initialValues[0].layerWeights );

for (let i = 0; i < 100000; i++ ) {
    const trainingInput = numberToMatrix(Math.floor(Math.random() * 16 ));
    backPropagation(trainingInput, initialValues);    
} 

let finalResult = forwardPropagation(input, initialValues);
console.log('trained result: ', finalResult[finalResult.length - 1]);
console.log('final weights', initialValues[0].layerWeights );
console.log('true result: ', getTrue(input));

function initializingLayers(arr) {
    // arr = [inputSize, 5, 3, 4]:

    // inputSize, hiddenLayerSize, outputSize
    let weightsAndBiases = [];

    for (let i = 0; i < arr.length - 1; i++ ) {
        const layerWeights = Array.from({ length: arr[i + 1]}, () => {
            return Array.from({ length: arr[i]}, () => Math.random() * 2 - 1)
        });
        const layerBiases = Array.from({ length: arr[i + 1]}, () => {
            return Array.from({ length: 1}, () => Math.random() * 2 -1)
        })

        weightsAndBiases.push({layerWeights, layerBiases})
    }

    return weightsAndBiases;

    // const hiddenLayerWeights = Array.from({ length: hiddenLayerSize }, () => {
    //   return Array.from({ length: inputSize }, () => Math.random());
    // });
    // const hiddenLayerBiases = Array.from({ length: hiddenLayerSize }, () => {
    //     return Array.from({ length: 1 }, () => Math.random());
    //   });
    // const outputLayerWeights = Array.from({ length: outputSize }, () => {
    //     return Array.from({ length: hiddenLayerSize }, () => Math.random());
    // });
    // const outputLayerBiases = Array.from({ length: outputSize }, () => {
    //     return Array.from({ length: 1 }, () => Math.random());
    // });

    // return [
        
    // ]
    // return [
    //     hiddenLayerWeights,
    //     hiddenLayerBiases,
    //     outputLayerWeights,
    //     outputLayerBiases
    // ]
}

function forwardPropagation(input, initArr) {

    // input, hiddenWeights, hiddenBias, outputWeight, outputBias

    let layerActivations = [];
    let previousActivation = input;
    for (let i = 0; i < initArr.length; i++ ) {
        const product = matrixMultiply(initArr[i].layerWeights, previousActivation);
        const biased = addBias(product, initArr[i].layerBiases);
        const layerActivation = activate(biased, sigmoid);
        layerActivations.push(layerActivation)
        previousActivation = layerActivation;
    }

    return layerActivations;
}

function backPropagation(input, initArr) {
    // input, hiddenWeights, hiddenBias, outputWeight, outputBias
    const learningRate = 0.1;
    const outputs = forwardPropagation(input, initArr)

    const trueValues = getTrue(input);
    let error = matrixSubtract(trueValues, outputs[outputs.length - 1]);

    for (let i = initArr.length -1; i >= 0; i--) {
        let gradient = pointwise(outputs[i], sigmoidDerivative);
        gradient = hadamard(gradient, error);
        gradient = scale(gradient, learningRate);

        const prevLayerTranspose = transpose(i === 0 ? input : outputs[i - 1]);
        const deltas = matrixMultiply(gradient, prevLayerTranspose);

        matrixAdd(initArr[i].layerWeights, deltas);

        // update biases
        matrixAdd(initArr[i].layerBiases, gradient);

        error = matrixMultiply(transpose(initArr[i].layerWeights), error);
    }

    // const {hiddenLayerValues, outputLayerValues} = forwardPropagation(input, hiddenWeights, hiddenBias, outputWeight, outputBias);


    // let gradient = pointwise(outputLayerValues, sigmoidDerivative);
    // gradient = hadamard(gradient, outputError);
    // gradient = scale(gradient, learningRate);

    // const hiddenLayerTranspose = transpose(hiddenLayerValues);
    // const deltas = matrixMultiply(gradient, hiddenLayerTranspose);

    // // update new weights
    // matrixAdd(outputWeight, deltas);

    // // update biases of output layer
    // matrixAdd(outputBias, gradient)

    // const hiddenError = matrixMultiply(transpose(outputWeight), outputError);

    // let gradientHidden = pointwise(hiddenLayerValues, sigmoidDerivative);
    // gradientHidden = hadamard(gradientHidden, hiddenError);
    // gradientHidden = scale(gradientHidden, learningRate)

    // const inputTranspose = transpose(input);
    // const deltasHiddenLayer = matrixMultiply(gradientHidden, inputTranspose);

    // // update new hidden layer weights
    // matrixAdd(hiddenWeights, deltasHiddenLayer);

    // // update biases of hidden layer
    // matrixAdd(hiddenBias, gradientHidden);

}

function getTrue(inputMatrix) {
    let num = 0;
    for (let i = 3; i >= 0; i--) {
        num |= inputMatrix[i][0] << (3-i);
    }
    if (num > 5 && num < 10) {
        return [[1], [0]]
    } else {
        return [[0], [1]]
    }
     
}

// ========== MATRIX OPERATIONS ======== //

function numberToMatrix(num) {
    const matrix = [];
    for (let i = 1; i < 16; i = i << 1) {
      matrix.unshift(i & num ? [1] : [0]);
    }
    return matrix;
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

function matrixAdd (matrix, matrixTwo) {
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j ++) {
            matrix[i][j] += matrixTwo[i][j];
        }
    }
}

function matrixSubtract (matrix, matrixTwo) {
    let result = [];
    for (let i = 0; i < matrix.length; i++) {
        let row = [];
        for (let j = 0; j < matrix[0].length; j ++) {
            let element = matrix[i][j] - matrixTwo[i][j]
            row.push(element);
        }
        result.push(row)
    }

    return result;
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
    if (matrix.length !== bias.length) {
        throw new Error('cannot add bias')
    } else {
        let result = []
        for (let i = 0; i < matrix.length; i++) {
            let row = [];
            for (let j = 0; j < matrix[0].length; j++) {
                let sum = matrix[i][j] + bias[j][0]
                row.push(sum);
            }
            result.push(row)
        }
        return result;
    }
}

function hadamard (matrix, matrixTwo) {
    let result = [];
    for (let i = 0; i < matrix.length; i++) {
        let row = [];
        for (let j = 0; j < matrix[0].length; j ++) {
            let element = matrix[i][j] * matrixTwo[i][j]
            row.push(element);
        }
        result.push(row)
    }

    return result;
}

function scale(matrix, scalar) {
    const multiply = (element) => {
        return element * scalar;
    }

    const result = pointwise (matrix, multiply);
    return result;
}

function pointwise(matrix, operation) {
    let result = [];
    for (let i = 0; i < matrix.length; i++) {
        let row = [];
        for (let j = 0; j < matrix[0].length; j++) {
            let element = matrix[i][j];
            let activatedElement = operation(element);
            row.push(activatedElement)
        }
        result.push(row)
    }
    return result;
}

// =========== ACTIVATION OPERATIONS ========== //

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

function outputError (predictedOutput, trueOutput) {
    return predictedOutput - trueOutput;
}

function sigmoidDerivative (x) {
    return ( x * (1 - x) )
}

// ======== //


