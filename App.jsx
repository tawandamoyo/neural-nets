import React, {useEffect, useState} from "react";
import {forwardPropagation, trainNetwork} from './neuralnet';
import weightsAndBiases from './weightsAndBiases.json'

function App () {
    const [isDrawing, setIsDrawing] = useState(false);
    const [x, setX] = useState(0);
    const [y, setY] = useState(0);
    const [clearCanvas, setClearCanvas] = useState(false);
    const [imageData, setImageData] = useState([]);
    // const [weightsAndBiases, setWeightsAndBiases] = useState(null);
    const [precision, setPrecision] = useState(85.09883426254436);
    const [prediction, setPrediction] = useState(null);

    useEffect(() => {
        // if (weightsAndBiases) {

            const imageDataPixels = imageData.filter((_, i) => i % 4 === 3);
            let converted = scaleDown(imageDataPixels).map((element) => {
                return element > 125 ? 1 : 0
            });
            let predictedDigit = forwardPropagation({input: converted}, weightsAndBiases);
            predictedDigit = predictedDigit[predictedDigit.length - 1].map((activ) => {
                return activ[0]
            });
            console.log(predictedDigit);
            setPrediction(predictedDigit.indexOf(Math.max(...predictedDigit)))
       // }
        
    }, [imageData]);

    // useEffect(() => {
    //     let {weightsAndBiases, precision} = trainNetwork();
    //     setWeightsAndBiases(weightsAndBiases);
    //     setPrecision(precision);
    // }, [])


    const draw = (event) => {
        const canvas = document.querySelector('#canvas');
        const context = canvas.getContext('2d');
        context.strokeStyle = '#000000';
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.lineWidth = 30;
        

        if (isDrawing) {
            context.beginPath();
            context.moveTo(x,y);
            context.lineTo(event.nativeEvent.offsetX, event.nativeEvent.offsetY);
            context.stroke();
            setX(event.nativeEvent.offsetX);
            setY(event.nativeEvent.offsetY);
            let imageDetails = context.getImageData(0,0, canvas.width, canvas.height).data

            setImageData(imageDetails)

        }

        
    };
    const clearBoard = () => {
        setClearCanvas(!clearCanvas);
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height)
    }

 


    function scaleDown(inputArr) {
        const convertedImageData = [];

        for (let i = 0; i < 784; i++) {
            const pixelValues = [];

            let row = 10 * Math.floor(i/10);
            let column = 10 * (i%10);

            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    const pixelRow = row + i;
                    const pixelCol = column + j;

                    const pixelValue = inputArr[(pixelRow * 100) + pixelCol];

                    pixelValues.push(pixelValue)
                }
            }
            const averagePixelValue = pixelValues.reduce((sum, number) => number + sum, 0) / 100;
            convertedImageData.push(averagePixelValue);
        }
        return convertedImageData;

    }

    return (
        <>
            <h1>Precision: {precision}</h1>
            <canvas
                id="canvas"
                width={'280px'}
                height={'280px'}
                onMouseMove={draw}
                onMouseDown={(event)=> {
                    setIsDrawing(true);
                    setX(event.nativeEvent.offsetX);
                    setY(event.nativeEvent.offsetY);
                }}
                onMouseUp={()=> {
                    setIsDrawing(false)
                }}
                onMouseOut = {() => {
                    setIsDrawing(false)
                }}
                style={
                    {backgroundColor: clearCanvas ? 'red' : "green" }
                }
            />
            <button onClick={clearBoard}>Clear</button>
            <canvas
                id="canvas"
                width={'280px'}
                height={'280px'}
                onMouseMove={draw}
                onMouseDown={(event)=> {
                    setIsDrawing(true);
                    setX(event.nativeEvent.offsetX);
                    setY(event.nativeEvent.offsetY);
                }}
                onMouseUp={()=> {
                    setIsDrawing(false)
                }}
                onMouseOut = {() => {
                    setIsDrawing(false)
                }}
                style={
                    {backgroundColor: clearCanvas ? 'red' : "green" }
                }
            />
            <button onClick={clearBoard}>Draw</button>
            <div>
                <h3>Results</h3>
                <p>The prediction is {prediction ? prediction : null } , with precision of {precision}</p>
            </div>
        </>
        
    )
}

export default App;

