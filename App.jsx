import React, {useEffect, useState} from "react";


function App () {
    const [isDrawing, setIsDrawing] = useState(false);
    const [x, setX] = useState(0);
    const [y, setY] = useState(0);

    const draw = (event) => {
        const canvas = document.querySelector('#canvas');
        const context = canvas.getContext('2d');
        context.strokeStyle = '#000000';
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.lineWidth = 20;

        if (isDrawing) {
            context.beginPath();
            context.moveTo(x,y);
            context.lineTo(event.nativeEvent.offsetX, event.nativeEvent.offsetY);
            context.stroke();
            setX(event.nativeEvent.offsetX);
            setY(event.nativeEvent.offsetY);
            console.log(context.getImageData(0, 0, 280, 280))
        }
        
    }

    return (
        <>
            <h1>Hello</h1>
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
                    {
                        border: '3px green',
                        backgroundColor: 'red'

                    }
                }
            />
        </>
        
    )
}

export default App;