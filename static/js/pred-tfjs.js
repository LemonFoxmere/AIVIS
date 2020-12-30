// network parameter
let layers = 3
let neurons = [3,7,2]

// training parameters
let lr = 0.001
let epoch = 10
let b_size = 1

// data parameters
let trainSplit = 0.9
let valSplit = 0.1
let dataStream = '../../TrainingData.json'
let fullData
let X = null; let y = null; let tX = null; let ty = null 
let isTrainEmpty = true; let isLabelEmpty = true; let isEpochEmpty = true; let isBsizeEmpty = true

// create and initialize model structure
const model = tf.sequential({
    layers:[
        tf.layers.dense({units: 10, activation: 'relu', inputShape: [3]}),
        tf.layers.dense({units: 2, activation: 'softmax'})
    ]
});

// create simple test data
const testPred = tf.tensor2d([
    [22,56,79]
])

// training model for EPOCH amount of times
async function train(model) {
    for (let i = 0; i < epoch; i++){
        const response = await model.fit(X, y, {
            batchSize: b_size,
            epoch: 1,
            shuffle: true,
            validationSplit: valSplit
        })
        console.log(response.history.loss[0])
    }
}

// read in text file to be processed
function readTextFile(file, callback) {
    var rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function() {
        if (rawFile.readyState === 4 && rawFile.status == "200") {
            callback(rawFile.responseText);
        }
    }
    rawFile.send(null);
}

// test model
document.querySelector('#testBtn').addEventListener('click', () => {
    let ys = model.predict(testPred)
    ys.print();
})

// train model
let trainBtn = document.querySelector('#trainBtn')
trainBtn.addEventListener('click', () => {
    // check if required input exists
    readTextFile(dataStream, function(text){
        // test if trainable
        if(isTrainEmpty || isLabelEmpty || isEpochEmpty || isBsizeEmpty){
            return
        }

        epoch = Number(document.querySelector("#epochs").value)
        b_size = Number(document.querySelector("#bsize").value)

        // create optimizer
        const opt = tf.train.adam(learningRate = lr, beta2 = lr/epoch)

        // update configuration
        const config = {
            optimizer: opt,
            // loss: "categoricalCrossentropy"
            loss: "binaryCrossentropy"
            // loss: "meanSquaredError"
        }
        console.log(JSON.stringify(model.outputs[0].shape)); // verifies output shape
        // compile model
        model.compile(config)

        console.log(model)
        train(model).then(() => console.log("Training Done"))
    })
})

// error information
trainBtn.addEventListener('mouseover', ()=>{
    if(isTrainEmpty){
        trainBtn.innerHTML = "Missing Input"
        return
    } if (isLabelEmpty){
        trainBtn.innerHTML = "Missing Label"
        return
    } if (isEpochEmpty){
        trainBtn.innerHTML = "Invalid Epochs"
        return
    } if (isBsizeEmpty){
        trainBtn.innerHTML = "Invalid B-Size"
        return
    }
})
trainBtn.addEventListener('mouseout', ()=>{
    trainBtn.innerHTML = "Train"
})

// convert the CSV to JSON data
function CSVToJSON(csvData) {
    let jsonFormat = "{\n"
    let i = 0
    csvData.split('\n').forEach(s => {
        jsonFormat += "\t\"" + i + "\":[" + s + "]" + (csvData.split('\n').length-1 != i ? ",\n" : "\n")
        i += 1
    });
    jsonFormat += "}"
    return jsonFormat 
}

// save the uploaded file to accesible variables within the current instance
function saveFileState() {
    // read in from local storage
    let jsonTrain = JSON.parse(localStorage.getItem('trainSet'))
    let jsonLabel = JSON.parse(localStorage.getItem('labelSet'))

    if(jsonTrain != null){
        isTrainEmpty = false
        document.querySelector('#uploadTrain').style.backgroundColor = "#28a524"
        document.querySelector('#uploadTrain').innerHTML = "Change Input"
    } else {
        isTrainEmpty = true
        document.querySelector('#uploadTrain').style.backgroundColor = "#e23a31"
        document.querySelector('#uploadTrain').innerHTML = "Upload Input"
    } 
    
    if (jsonLabel != null){
        isLabelEmpty = false
        document.querySelector('#uploadLabel').style.backgroundColor = "#28a524"
        document.querySelector('#uploadLabel').innerHTML = "Change Labels"
    } else {
        isLabelEmpty = true
        document.querySelector('#uploadLabel').style.backgroundColor = "#e23a31"
        document.querySelector('#uploadLabel').innerHTML = "Upload Labels"
    }
    
    if (document.querySelector("#epochs").value != "" &&
            document.querySelector("#epochs").value >= 1 && document.querySelector("#epochs").value <= 200 &&
            Math.floor(document.querySelector("#epochs").value) == document.querySelector("#epochs").value){
        isEpochEmpty = false
    } else {
        isEpochEmpty = true
    } 
    
    if (document.querySelector("#bsize").value != "" &&
        document.querySelector("#bsize").value >= 1 && document.querySelector("#bsize").value <=500 &&
        Math.floor(document.querySelector("#bsize").value) == document.querySelector("#bsize").value){
        isBsizeEmpty = false
    } else {
        isBsizeEmpty = true
    }
    
    if(isLabelEmpty || isTrainEmpty || isBsizeEmpty || isEpochEmpty){
        document.querySelector('#trainBtn').style.backgroundColor = "#e23a31"
        return
    }

    document.querySelector('#trainBtn').style.backgroundColor = "#28a524"
    
    splitPoint = Math.floor(Object.keys(jsonTrain).length * trainSplit)
    
    let kX = []; let ky = []; let ktX = []; let kty = [] 
    
    for (let i = 0; i < Object.keys(jsonTrain).length; i++){
        if (i < splitPoint){
            kX.push(jsonTrain[i])
            ky.push(jsonLabel[i])
        } else {
            ktX.push(jsonTrain[i])
            kty.push(jsonLabel[i])
        }
    }
    
    // reclass and store data
    X = tf.tensor2d(kX)
    y = tf.tensor2d(ky)
    tX = tf.tensor2d(ktX)
    ty = tf.tensor2d(kty)
}

// upload and update local storage
document.querySelector('#training-input').onchange = () => {
    // read csv file from the upload button
    let file = document.querySelector('#training-input').files[0], read = new FileReader()
    read.readAsBinaryString(file); // convert to String
    read.onloadend = function(){
        localStorage.setItem('trainSet', CSVToJSON(read.result)) // save to local storage
        saveFileState()
    }
}
document.querySelector('#label-input').onchange = () => {
    let file = document.querySelector('#label-input').files[0], read = new FileReader()
    read.readAsBinaryString(file);
    read.onloadend = function(){
        localStorage.setItem('labelSet', CSVToJSON(read.result))
        saveFileState()
    }
}
// -------------------------------------------------------BUILD NEURAL NETWORK-------------------------------------------------------
document.querySelector('#build').addEventListener('click', () => {
    // create lines connecting neurons
})

// --------------------------------------------------PROGRESS BAR STATUS MANAGEMENT----------------------------------------------------------
let lastDisplayProgress = false
let displayProgress = lastDisplayProgress;
function updateProgressWidth(){
    if(displayProgress != lastDisplayProgress){
        lastDisplayProgress = displayProgress
        let line = anime.timeline({
            targets:".progressCase",
            easing:'easeOutExpo',
            duration: 100
        })
        if (displayProgress){
            line.add({
                width:"90%"
            })
        } else {
            line.add({
                width:"10%"
            })
        }
        line.play()
    }
}

setInterval(()=>{
    updateProgressWidth()
}, 100)

// ------------------------------------------------------HELP BUTTON FUNCTION----------------------------------------------------------
let helpWanted = false
let helpQuote = "Hover Over Area of Interst"
let currentQuote = helpQuote
let animating = false
// when the help button is clicked
var last_clicked = 0;
document.querySelector('#help').addEventListener('click', ()=>{
    if (Date.now() - last_clicked < 1000) return;
    last_clicked = Date.now();
    helpWanted = !helpWanted;
    lastHover = null
    if (helpWanted){ // when help is wanted, start progress bar transition and text change
        displayProgress = true
        anime({
            easing:'easeOutExpo',
            targets:'#trainProgressCase',
            backgroundColor:"#c4c4c4",
            duration: 100,
        })
        anime({
            easing:'easeOutExpo',
            targets:'#help',
            backgroundColor:"#363636",
            duration:200,
            delay:100
        })
        anime({
            easing:'easeOutExpo',
            targets:'#trainProgressBar',
            width:"67%",
            duration: 1000,
            delay:200
        })
        anime({
            easing:'easeOutExpo',
            targets:'#progress',
            height: '10%',
            duration:700
        })
        changeStatus("Hover Over Area of Interst")
    } else {
        displayProgress = false
        anime({
            easing:'easeOutExpo',
            targets:'#trainProgressBar',
            width:"0%",
            duration: 700,
        })
        anime({
            easing:'easeOutExpo',
            targets:'#help',
            backgroundColor:"#5c6d7e",
            duration: 200,
        })
        anime({
            easing:'easeOutQuart',
            targets:'#trainProgressCase',
            backgroundColor:"#29ac25",
            duration: 300,
            delay:400
        })
        anime({
            easing:'easeOutExpo',
            targets:'#progress',
            height: '5%',
            duration:700
        })
    changeStatus("Ready")
    }
})
// hover animation
function changeStatus(status){
    if(status != document.querySelector("#status").innerHTML){
        animating = true
        let line = anime.timeline({
            targets:"#status",
            easing:'easeOutQuart',
            duration: 125
        })
        line.add({
            translateY:15,
            opacity:0
        })
        line.add({
            update: () => {
                document.querySelector("#status").innerHTML = status
            }
        })
        line.add({
            translateY:0,
            opacity:1
        })
        line.add({
            update:(anim) => {
                if(Math.round(anim.progress) == 100){
                    animating = false
                }
            }
        })
        line.play()
    }
}
// hover detection
// progress bar
let lastHover = null
document.querySelector("#trainProgressCase").addEventListener('mouseover', ()=>{
    lastHover = "#trainProgressCase"
})
// train button
document.querySelector("#trainBtn").addEventListener('mouseover', ()=>{
    lastHover = "#trainBtn"
})
// test button
document.querySelector("#testBtn").addEventListener('mouseover', ()=>{
    lastHover = "#testBtn"
})
// input button
document.querySelector("#uploadTrain").addEventListener('mouseover', ()=>{
    lastHover = "#uploadTrain"
})
// test button
document.querySelector("#uploadLabel").addEventListener('mouseover', ()=>{
    lastHover = "#uploadLabel"
})
// epoch input
document.querySelector("#epochs").addEventListener('mouseover', ()=>{
    lastHover = "#epochs"
})
// batch size input
document.querySelector("#bsize").addEventListener('mouseover', ()=>{
    lastHover = "#bsize"
})
// tutorial button
document.querySelector("#tutorial").addEventListener('mouseover', ()=>{
    lastHover = "#tutorial"
})
// help button
document.querySelector("#help").addEventListener('mouseover', ()=>{
    lastHover = "#help"
})
// logo
document.querySelector(".logo").addEventListener('mouseover', ()=>{
    lastHover = ".logo"
})
// build
document.querySelector("#build").addEventListener('mouseover', ()=>{
    lastHover = "#build"
})
// layer
document.querySelectorAll(".layer").forEach((elem) => {
    elem.addEventListener('mouseover', ()=>{
        lastHover = ".layer"
    })
})

setInterval(() => {
    if(helpWanted && !animating){
        switch (lastHover){
            case "#trainProgressCase":
                changeStatus("<b>Progress Bar</b> <br> Displays the progress of training")
                break;
            case "#trainBtn":
                changeStatus("<b>Train Button</b> <br>Trains your neural network")
                break;
            case "#testBtn":
                changeStatus("<b>Test Button</b> <br>Test your neural network")
                break;
            case "#uploadTrain":
                changeStatus("<b>Upload Inputs</b> <br>Upload training and testing input data")
                break;
            case "#uploadLabel":
                changeStatus("<b>Upload Labels</b> <br>Upload training and testing label data")
                break;
            case "#epochs":
                changeStatus("<b>Epochs</b> <br>The number of times the neural network will be trained")
                break;
            case "#bsize":
                changeStatus("<b>Batch Size</b> <br>The amount of data that will be used per train cycle")
                break;
            case "#tutorial":
                changeStatus("<b>Tutorial</b> <br>A more detailed instruction and explanation of a neural network")
                break;
            case "#help":
                changeStatus("<b>Help</b> <br>Press again to return to editing mode")
                break;
            case ".logo":
                changeStatus("<b>AIVIS</b> <br>Return to home screen")
                break;
            case "#build":
                changeStatus("<b>Build</b> <br>Compile your neural network")
                break;
            case ".layer":
                changeStatus("<b>Layers</b> <br>Part of the neural network architechture. Click on <i>Tutorial</i> to learn more.")
                break;
            default:
                changeStatus(helpQuote)
        }
    }
}, 10);


// initial startup script
document.querySelector('#uploadTrain').style.backgroundColor = "#e23a31"
document.querySelector('#uploadLabel').style.backgroundColor = "#e23a31"
document.querySelector('#trainBtn').style.backgroundColor = "#e23a31"
setInterval(() => {
    saveFileState()
}, 50);