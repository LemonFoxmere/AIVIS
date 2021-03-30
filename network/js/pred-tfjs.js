// network parameter
let addable = true
let remable = true
let layers = 3
let training = false
let currentStatusOK = 0 // 0 = OK, 1 = WARNING, 2 = ERROR
let hasOutputAct = true
let animatingProgress = false
let trainIter = 0;

// training parameters
let lr = 0.001
let epoch = 10
let b_size = 1
let model = null;

// data parameters
let trainSplit = 0.9
let valSplit = 0.1
let fullData
let X = null; let y = null; let tX = null; let ty = null 
let tXarr = null;
let isTrainEmpty = true; let isLabelEmpty = true; let isEpochEmpty = true; let isBsizeEmpty = true

// create simple test data
const testPred = null;
let help_clicked = false;

const sleep = (delay) => new Promise((resolve) => setTimeout(resolve, delay))

// training model for EPOCH amount of times
async function train(model) {
    for (let i = 0; i < epoch; i++){
        const response = await model.fit(X, y, {
            batchSize: b_size,
            epoch: 1,
            shuffle: true,
            validationSplit: valSplit
        })
        percFinished = String(Math.floor(((i+1)/epoch)*100))
        document.querySelector("#status").innerHTML = "Network Loss<br>" + Math.round(Number(response.history.loss[0])*10000000)/10000000
        document.querySelector("#trainProgressBar").style.width = percFinished +"%"
        
        await sleep(0.1)
        console.log(response.history.loss[0])
    }
    document.querySelector("#trainBtn").disabled = false;
    training = false
    trainIter++;
    anime({
        easing:'easeOutExpo',
        targets:'#testBtn',
        backgroundColor:"#28a524",
        duration: 100,
    })
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
    if(model == null || trainIter == 0) return

    helpWanted = false

    // select a random input
    let length = tX.shape[0]
    let index = Math.floor(Math.random() * (length-1))

    let dataPoint = tf.tensor2d([tXarr[index]])

    let ys = model.predict(dataPoint)
    ys.print();

    let pred =  ys.arraySync()
    pred[0].forEach((e) => {
         Math.round(e*1000)/1000
        
    })
    
    for(let i = 0; i < pred[0].length; i++){
        pred[0][i] = Math.round(pred[0][i]*1000)/1000
    }

    changeStatus("Input Data: <b>" + tXarr[index] + "</b><br>Prediction: <b>" + pred + "</b>")
})

document.querySelector('#testBtn').addEventListener('mouseover', () => {
    if(model == null || trainIter == 0){
        document.querySelector('#testBtn').innerHTML = "No Trained Model"
    }
})

document.querySelector('#testBtn').addEventListener('mouseout', () => {
    document.querySelector('#testBtn').innerHTML = "Test"
})

// train model
let trainBtn = document.querySelector('#trainBtn')
trainBtn.addEventListener('click', () => {
    if(isTrainEmpty || isLabelEmpty || isEpochEmpty || isBsizeEmpty){ //fail safe check
        return
    }
    // pull data
    let inputDataSize = X.shape[1]
    let outputDataSize = y.shape[1]
    if(inputDataSize != document.querySelectorAll('.input').length || outputDataSize != document.querySelectorAll('.output').length){
        return
    }
    
    // expand and initialize progress bar
    displayProgress = true;
    anime({
        easing:'easeOutExpo',
        targets:'#trainProgressCase',
        backgroundColor:"#c4c4c4",
        duration: 100,
    })
    training = true
    anime({
        easing:'easeOutExpo',
        targets:'#trainBtn',
        backgroundColor:"#3d3c42",
        duration: 100,
    })
    document.querySelector("#trainBtn").disabled = true;
    anime({
        easing:'easeOutExpo',
        targets:'#trainProgressBar',
        width:"0%",
        duration: 1000,
        delay:200
    })
    changeStatus("Network Loss<br>N/a");
    setTimeout(()=>{
        //start initializing network parameters
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
        model = tf.sequential()
        
        // generate layer data
        let layerNum = document.querySelectorAll(".layer").length
        let neurons = []
        for (let i = 1; i < layerNum-1; i++){ // loop thru every layer
            let neuronCount = document.querySelector(String("#l" + i)).childElementCount
            neurons.push(neuronCount)
        }
        
        // create and initialize model structure
        model.add(tf.layers.dense({units: inputDataSize, inputShape: [inputDataSize]}));
        for (let i = 0; i < neurons.length; i++){
            model.add(tf.layers.dense({units: neurons[i], activation: 'relu'}));
        }
        if(hasOutputAct){
            model.add(tf.layers.dense({units: outputDataSize, activation: 'softmax'}));
        } else {
            model.add(tf.layers.dense({units: outputDataSize}));
        }
        console.log(model)
        
        model.compile(config) //compile model
        train(model).then(() => console.log("Training Done")) // start async function of training network
    }, 1000)
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
    } if (X.shape[1] != document.querySelectorAll('.input').length){
        trainBtn.innerHTML = "Invalid Input Size"
        return
    } if (y.shape[1] != document.querySelectorAll('.output').length){
        trainBtn.innerHTML = "Invalid Output Size"
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
    if(training) return
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

    if(X == null || y == null || tX == null || ty == null){      
        X = y = tX = ty = null;  
        splitPoint = Math.floor(Object.keys(jsonTrain).length * trainSplit)
        
        let kX = []; let ky = []; let ktX = []; let kty = [] 
        
        for (let i = 0; i < Object.keys(jsonTrain).length-1; i++){
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
        tXarr = tX.arraySync()
    }
    
    if (document.querySelector("#bsize").value != "" &&
        document.querySelector("#bsize").value >= 1 && document.querySelector("#bsize").value <=500 &&
        Math.floor(document.querySelector("#bsize").value) == document.querySelector("#bsize").value){
        isBsizeEmpty = false
    } else {
        isBsizeEmpty = true
    }

    if(isLabelEmpty || isTrainEmpty || isBsizeEmpty || isEpochEmpty || X.shape[1] != document.querySelectorAll('.input').length || y.shape[1] != document.querySelectorAll('.output').length){
        document.querySelector('#trainBtn').style.backgroundColor = "#e23a31"
        return
    }

    document.querySelector('#trainBtn').style.backgroundColor = "#28a524"
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

let blink=true

// slowly flash the lines
// setInterval(()=>{
//     if(training){
//         if(blink){
//             lineOpacity=1
//             // red=255
//         } else if(!blink){
//             lineOpacity=0.5
//             // red=0
//         }
//         blink = !blink
//     } else {
//         lineOpacity=0.5
//     }
// },700)

let lineWidth = 2
// let boxCenterXOffset = document.querySelector('.l1').getBoundingClientRect().width/2
// let boxCenterYOffset = document.querySelector('.l1').getBoundingClientRect().height/2
let boxCenterXOffset = 0
let boxCenterYOffset = 0

setInterval(()=>{
    while(document.querySelector('.asdfghjkl')){
        const ele = document.querySelector('.asdfghjkl')
        ele.remove()
    }
    for (let i = 0; i < layers-1; i++){
        if(document.querySelectorAll('.l' + (i+1)).length + document.querySelectorAll('.l' + (i+2)).length > 9){
            buildVisNetHidden(document.querySelector('#l' + (i+1)), document.querySelector('#l' + (i+2))) 
            continue
        }
        for (let j = 0; j < document.querySelectorAll('.l' + (i+1)).length; j++){
            for (let k = 0; k < document.querySelectorAll('.l' + (i+2)).length; k++){
                buildVisNet(i+1, i+2, j, k)
            }
        }
    }
},10);

let lineOpacity=1
let currentLineOpacity=1
let red=0
let cr=0

function buildVisNetHidden(layer1, layer2){

    let shortHand = document.createElement('div')
    shortHand.classList.add('asdfghjkl')
    shortHand.style.backgroundColor='rgb(0,0,0)'
    shortHand.style.borderRadius='1000px'
    shortHand.style.opacity = currentLineOpacity
    shortHand.style.position='fixed'
    shortHand.style.width = '10px'
    shortHand.style.height = '10px'

    shortHand1 = shortHand.cloneNode(true)
    shortHand2 = shortHand.cloneNode(true)

    const centerX = (layer1.offsetLeft + layer1.offsetWidth + layer2.offsetLeft)/2
    const centerY = (layer1.offsetTop) + layer1.offsetHeight / 2

    shortHand.style.transform = 'translate(' + (centerX-5) + 'px,' + (centerY-25) + 'px)'
    shortHand1.style.transform = 'translate(' + (centerX-5) + 'px,' + (centerY-5) + 'px)'
    shortHand2.style.transform = 'translate(' + (centerX-5) + 'px,' + (centerY+15) + 'px)'

    document.querySelector('#main').appendChild(shortHand)
    document.querySelector('#main').appendChild(shortHand1)
    document.querySelector('#main').appendChild(shortHand2)
}

function buildVisNet(layer1, layer2, neuron1, neuron2){
    var x1 = document.querySelectorAll('.l'+layer1)[neuron1].offsetLeft + boxCenterXOffset;
    var x2 = document.querySelectorAll('.l'+layer2)[neuron2].offsetLeft + boxCenterXOffset;
    var y1 = document.querySelectorAll('.l'+layer1)[neuron1].offsetTop + boxCenterYOffset;
    var y2 = document.querySelectorAll('.l'+layer2)[neuron2].offsetTop + boxCenterYOffset;
    var hypotenuse = Math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    var angle = Math.atan2((y1-y2), (x1-x2)) *  (180/Math.PI);

    if(angle >= 90 && angle < 180){
        y1 = y1 - (y1-y2);
    }
    if(angle > 0 && angle < 90){
        x1 = x1 - (x1-x2);
        y1 = y1 - (y1-y2);
    }
    if(angle <= 0 && angle > -90){
        x1 = x1 - (x1-x2);
    }

    var x1 = document.querySelectorAll('.l'+layer1)[neuron1].offsetLeft + boxCenterXOffset;
    var x2 = document.querySelectorAll('.l'+layer2)[neuron2].offsetLeft + boxCenterXOffset;
    var y1 = document.querySelectorAll('.l'+layer1)[neuron1].offsetTop + boxCenterYOffset;
    var y2 = document.querySelectorAll('.l'+layer2)[neuron2].offsetTop + boxCenterYOffset;
    
    let r = hypotenuse/2

    let circRad = document.querySelector('.l1').getBoundingClientRect().height/2

    let offsetY = (Math.sin(toRadians(angle)))*r - circRad + lineWidth/4
    // let offsetY = 0
    let offsetX = (1+Math.cos(toRadians(angle))) * r - circRad + lineWidth/4
    // let offsetX = 0

    let line = document.createElement('div')
    line.style.width = hypotenuse + 'px'
    line.style.height = lineWidth + 'px'
    if(Math.floor(cr) != Math.floor(r)){
        if(cr > red){
            cr-=0.1
        }else{
            cr+=0.1
        }

        if(cr > 255){
            cr = 255
        } else if(cr < 0){
            cr = 0
        }
    }
    line.style.backgroundColor='rgb('+cr+',0,0)'
    line.style.borderRadius='1000px'
    if(currentLineOpacity != lineOpacity){
        if(currentLineOpacity > lineOpacity){
            currentLineOpacity-=0.0003
        }else{
            currentLineOpacity+=0.0003
        }

        if(currentLineOpacity > 1){
            currentLineOpacity = 1
        } else if(currentLineOpacity < 0){
            currentLineOpacity = 0
        }
    }
    line.style.opacity = currentLineOpacity
    line.style.position='fixed'

    line.style.transform = 'translate(' + (x1) + 'px,' + (y1) + 'px)' + 'translate(' + -offsetX + 'px,' + -offsetY + 'px) ' + 'rotate(' + angle + 'deg) '

    line.className = "asdfghjkl"
    document.querySelector('#main').appendChild(line)
}

function toRadians (angle) {
    return angle * (Math.PI / 180);
}

// ------------------------------------------------------- NEURON MANIPULATION -------------------------------------------------------

document.querySelector("#toggleOutputActivation").addEventListener("click", () => {
    hasOutputAct = !hasOutputAct;
    if(!hasOutputAct){
        document.querySelector("#toggleOutputActivation").innerHTML = "Off"
        anime({
            targets:document.querySelector("#toggleOutputActivation"),
            backgroundColor:"#3d3c42",
            duration:30,
            easing:"easeOutCubic"
        })
        return;
    }
    document.querySelector("#toggleOutputActivation").innerHTML = "Softmax"
    anime({
        targets:document.querySelector("#toggleOutputActivation"),
        backgroundColor:"#28a524",
        duration:30,
        easing:"easeOutCubic"
    })
})

document.querySelector("#toggleOutputActivation").addEventListener("mouseover", () => {
    if(hasOutputAct){
        document.querySelector("#toggleOutputActivation").innerHTML = "Softmax"
        return
    }
    document.querySelector("#toggleOutputActivation").innerHTML = "Off"
})

document.querySelector("#toggleOutputActivation").addEventListener("mouseout", () => {
    document.querySelector("#toggleOutputActivation").innerHTML = "Output Activation"
})

// defaults
document.querySelector('#l1.layer').addEventListener('wheel', (e)=>{
    if (checkScrollDirectionIsUp(e)) {
        addNeuron(1,'input')
    } else {
        delNeuron(1)
    }
})
document.querySelector('#l2.layer').addEventListener('wheel', (e)=>{
    if (checkScrollDirectionIsUp(e)) {
        addNeuron(2)
    } else {
        delNeuron(2)
    }
})
document.querySelector('#l3.layer').addEventListener('wheel', (e)=>{
    if (checkScrollDirectionIsUp(e)) {
        addNeuron(3,'output')
    } else {
        delNeuron(3)
    }
})

function addNeuron(layerNum,special=null){
    let numNeurons = document.querySelector('#l'+String(layerNum)+'.layer').childElementCount+1
    if(numNeurons > 9){
        return;
    }
    // create neuron div
    let neuron = document.createElement('div')
    neuron.classList.add('neuron')
    if(special != null){
        neuron.classList.add(special)
    }
    let layerTag = String('l' + String(layerNum))
    neuron.classList.add(layerTag)
    neuron.id = 'n' + String(numNeurons)
    document.querySelector('#l'+String(layerNum)+'.layer').appendChild(neuron)
}

function delNeuron(layerNum){
    let numNeurons = document.querySelector('#l'+String(layerNum)+'.layer').childElementCount-1
    if(numNeurons < 1){
        return;
    }
    document.querySelector('#l'+String(layerNum)+'.layer').removeChild(document.querySelector('#n'+String(numNeurons+1) + '.l' + layerNum))
}

function checkScrollDirectionIsUp(event) {
    if (event.wheelDelta) {
        return event.wheelDelta > 0;
    }
    return event.deltaY < 0;
}

// ------------------------------------------------------- LAYER MANIPULATION -------------------------------------------------------

document.querySelector('#addLayer').addEventListener('click', ()=>{
    if(layers+1 > 9){
        addable = false
        return
    }
    if(layers+1 > 8){
        document.querySelector('#addLayer').style.backgroundColor = '#e23a31'
    }
    
    remable = true
    document.querySelector('#delLayer').style.backgroundColor = '#28a524'

    div = document.createElement('div')

    document.querySelectorAll('.l'+layers).forEach((ele)=>{
        ele.classList.remove('output')
    })
    
    layers += 1;
    div.id=String('l'+layers)
    div.classList.add('layer')
    div.addEventListener('mouseover', ()=>{
        lastHover = ".layer"
    })
    
    // create neurons: <div class="neuron l2" id="n1"></div>
    defNeuron = document.createElement('div')
    defNeuron.id = 'n1'
    defNeuron.classList.add('neuron')
    defNeuron.classList.add('output')
    defNeuron.classList.add(String('l'+layers))
    
    div.appendChild(defNeuron)

    document.querySelector('#editor').appendChild(div)

    const lay = layers
    let prevLayer = document.querySelector('#l' + (lay-1)); // remove last layer's shit
    prevLayer.replaceWith(prevLayer.cloneNode(true));
    
    document.querySelector('#l' + lay).addEventListener('wheel', (e)=>{ // add new event listener
        if (checkScrollDirectionIsUp(e)) {
            addNeuron(lay,'output')
        } else {
            delNeuron(lay)
        }
    })
    document.querySelector('#l' + (lay-1)).addEventListener('wheel', (e)=>{ // change old shit
        if (checkScrollDirectionIsUp(e)) {
            addNeuron(lay-1)
        } else {
            delNeuron(lay-1)
        }
    })
    document.querySelector('#l' + (lay-1)).addEventListener('mouseover', (e)=>{ // add new event listener
        lastHover = ".layer"
    })
})

document.querySelector('#delLayer').addEventListener('click', ()=>{
    if(layers-1 < 2){
        document.querySelector('#delLayer').style.backgroundColor = '#e23a31'
        remable = false
        return
    }
    if(layers-1 < 3){
        document.querySelector('#delLayer').style.backgroundColor = '#e23a31'
    }

    addable = true
    document.querySelector('#addLayer').style.backgroundColor = '#28a524'
    
    id = String('l' + layers)
    layers-=1
    
    document.querySelectorAll('.l'+layers).forEach((ele)=>{
        ele.classList.add('output')
    })

    document.querySelector('#editor').removeChild(document.getElementById(id))

    const lay = layers

    let prevLayer = document.querySelector('#l' + lay); // remove this layer's shit
    prevLayer.replaceWith(prevLayer.cloneNode(true));
    document.querySelector('#l' + lay).addEventListener('wheel', (e)=>{ // add new event listener
        if (checkScrollDirectionIsUp(e)) {
            addNeuron(lay,'output')
        } else {
            delNeuron(lay)
        }
    })
})

setInterval(()=>{
    if(displayProgress) return
    if(!helpWanted){
        if(document.querySelectorAll('.neuron').length > 17){
            changeStatus("<b>Warning:</b> Too Much Layers or/and Neurons can Lead to Performance Loss")
            currentStatusOK = 1
            return
        }
        if(document.querySelectorAll('.neuron').length < 18){
            if(!help_clicked){
                changeStatus("Need Help Getting Started? Click the Help Button!")
            } else {
                changeStatus("Ready")
            }
            currentStatusOK = 0
        }
    }
},200)

// --------------------------------------------------PROGRESS BAR STATUS MANAGEMENT----------------------------------------------------------
let lastDisplayProgress = false
let displayProgress = lastDisplayProgress
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

function updateProgressColor(){
    if(helpWanted || displayProgress){
        return
    }
    if(currentStatusOK == 0){
        document.querySelector(".progressCase").style.backgroundColor = "#29ac25"
        return
    }
    if(currentStatusOK == 1){
        document.querySelector(".progressCase").style.backgroundColor = "#ffb400"
        return
    }
    if(currentStatusOK == 2){
        document.querySelector(".progressCase").style.backgroundColor = "#e20001"
        return
    }
}

setInterval(()=>{
    updateProgressWidth()
    updateProgressColor()
}, 200)

// ------------------------------------------------------HELP BUTTON FUNCTION----------------------------------------------------------
let helpWanted = false
let helpQuote = "Hover Over Area of Interest"
let currentQuote = helpQuote
let animating = false
// when the help button is clicked
var last_clicked = 0;
document.querySelector('#help').addEventListener('click', ()=>{
    if(training) return
    if (Date.now() - last_clicked < 1000) return;
    last_clicked = Date.now();
    helpWanted = !helpWanted;
    lastHover = null
    help_clicked = true
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
        changeStatus("Hover Over Area of Interest")
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
            height: '10%',
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
// help button
document.querySelector("#help").addEventListener('mouseover', ()=>{
    lastHover = "#help"
})
// logo
document.querySelector(".logo").addEventListener('mouseover', ()=>{
    lastHover = ".logo"
})
// addLayer
document.querySelector("#addLayer").addEventListener('mouseover', ()=>{
    lastHover = "#addLayer"
})
// delLayer
document.querySelector("#delLayer").addEventListener('mouseover', ()=>{
    lastHover = "#delLayer"
})
//toggleOutputActivation
document.querySelector("#toggleOutputActivation").addEventListener('mouseover', ()=>{
    lastHover = "#toggleOutputActivation"
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
                changeStatus("<b>Layers</b> <br>Hover and scroll on a layer to modify its neuron content.")
                break;
            case "#addLayer":
                changeStatus("<b>Add Layer</b> <br>Add a layer to the current network architecture.")
                break;
            case "#delLayer":
                changeStatus("<b>Remove Layer</b> <br>Remove a layer from the current network architecture.")
                break;
            case "#toggleOutputActivation":
                changeStatus("<b>Change Output Activation</b> <br>Enable or disable output activation in your network.")
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

for (let i = 0; i < layers-1; i++){
    for (let j = 0; j < document.querySelectorAll('.l' + (i+1)).length; j++){
        for (let k = 0; k < document.querySelectorAll('.l' + (i+2)).length; k++){
            buildVisNet(i+1, i+2, j, k)
        }
    }
}