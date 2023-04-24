import mnist from 'mnist';

export function trainNetwork() {
    const set = mnist.set(8000, 3000);

    const trainingSet = set.training;
    const testSet = set.test;

    let trainingObject = trainingSet[0];

    const initialValues = initializingLayers([784, 500, 80, 10]);
    let result = forwardPropagation(trainingObject, initialValues);
    console.log('untrained result: ', result[result.length - 1])
    let trueAnswer = trainingObject.output;
    console.log('trueAnswers', trueAnswer)

    let trainingRounds = 3000;
    console.log(`training for ${trainingRounds} times: `)
    for (let i = 0; i < trainingRounds; i++ ) {
        let trainingInput = trainingSet[i];
        backPropagation(trainingInput, initialValues);    
    } 

    console.log('training complete');


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
    );

    return {
        weightsAndBiases: initialValues,
        precision: percentageAccuracy
    };
}

// Function definitions:

function initializingLayers(arr) {
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

}

export function forwardPropagation(inputObj, initArr) {
    let input = inputObj.input;
    input = [input];
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
    const learningRate = 0.1;

    // get input
    let input = inputObj.input;
    input = [input];
    input = transpose(input)


    const outputs = forwardPropagation(inputObj, initArr)

    let trueValues = inputObj.output;
    trueValues = [trueValues];
    trueValues = transpose(trueValues);
        
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
}


// ========== MATRIX OPERATIONS ======== //

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

function sigmoidDerivative (x) {
    return ( x * (1 - x) )
}



