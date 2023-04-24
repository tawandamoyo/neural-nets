const { json } = require('express');
const fs = require('fs');
const mnist = require('mnist');

const set = mnist.set(8000, 3000);

const trainingSet = set.training;
const testSet = set.test;



let trainingObject = trainingSet[0];
// console.log(inputObj)

// hidden  layer will have 5 neurons ( 4 x 5 matrix)

// bias for hidden layer length 5

// output layer will have 3 neurons ( 5 x 3 matrix)

// bias length 3


const initialValues = initializingLayers([784, 500, 80, 10]);
let result = forwardPropagation(trainingObject, initialValues);
console.log('untrained result: ', result[result.length - 1])
let trueAnswer = trainingObject.output;
console.log('trueAnswers', trueAnswer)

// console.log('initial weights', initialValues[0].layerWeights );
let trainingRounds = 3000;
console.log(`training for ${trainingRounds} times: `)
for (let i = 0; i < trainingRounds; i++ ) {
    let trainingInput = trainingSet[i];
    backPropagation(trainingInput, initialValues);    
} 

console.log('training complete');

// for (let i = 0; i < 100000; i++ ) {
//     const trainingInput = numberToMatrix(Math.floor(Math.random() * 16 ));
//     backPropagation(trainingInput, initialValues);    
// } 

// let finalResult = forwardPropagation(trainingObject, initialValues);
// console.log('trained result: ', finalResult[finalResult.length - 1]);
// console.log('final weights', initialValues[0].layerWeights );
// console.log('true result: ', getTrue(input));

// TEST on Test Data

console.log('running tests on test data: ')

let testSetSize = testSet.length;

let testResults = [];

for (let i = 0; i < testSetSize; i++) {
    let trueOutput = testSet[i].output;
    let trueOutputIndex = trueOutput.findIndex( x => x === 1);

    let result = forwardPropagation(testSet[i], initialValues);
    let [resultArray] = transpose(result[result.length - 1]);
    let trainedOutputIndex = resultArray.indexOf(Math.max(...resultArray));

    if (trueOutputIndex === trainedOutputIndex) {
        testResults.push(trueOutputIndex);
    } 
}
console.log(testResults);

let percentageAccuracy = (testResults.length / testSetSize ) * 100;

console.log(
    `Your model results: \n
     With ${trainingRounds} trainings, your model accuracy is ${percentageAccuracy} % on test data of ${testSetSize} items
    `
)

fs.writeFileSync('./weightsAndBiases.json', JSON.stringify(initialValues))

// Function definitions:

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

function forwardPropagation(inputObj, initArr) {

    // input, hiddenWeights, hiddenBias, outputWeight, outputBias

    // get input vals
    input = inputObj.input;
    input = [input];
    // console.log('input: ', input)
    input = transpose(input)

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

function backPropagation(inputObj, initArr) {
    // input, hiddenWeights, hiddenBias, outputWeight, outputBias
    const learningRate = 0.1;

    // get input
    input = inputObj.input;
    input = [input];
    input = transpose(input)


    const outputs = forwardPropagation(inputObj, initArr)

    // retrieve true values

    // const trueValues = getTrue(input);
    let trueValues = inputObj.output;
    trueValues = [trueValues];
    trueValues = transpose(trueValues);
    // console.log('calc vals', outputs[outputs.length -1]);
    // console.log('true values', trueValues)
    
    
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
function relu(x) {
    return Math.max(0, x);
};
function reluDerivative(x) {
    if (x >= 0) {
        return 1
    } else {
        return 0
    }
};

// leaky relu
function leakyRelu(x) {
    if (x < 0) {
        return 0.01 * x
    } else {
        return x;
    }
};

function leakyReluDerivative(x) {
    if (x >= 0) {
        return 1
    } else {
        return 0.01
    }
};

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

 


