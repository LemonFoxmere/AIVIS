let lr = 0.001
let epoch = 10
let b_size = 1

let trainSplit = 0.9
let valSplit = 0.1
let dataStream = '../../TrainingData.json'
let fullData
let X = []; let y = []; let tX = []; let ty = [] 

const model = tf.sequential({
    layers:[
        tf.layers.dense({units: 10, activation: 'relu', inputShape: [3]}),
        tf.layers.dense({units: 2, activation: 'softmax'})
    ]
});

const testPred = tf.tensor2d([
    [244,164,151]
])

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
    let ys = model.predict(testPred)
    ys.print();
}


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

/*readTextFile(dataStream, function(text){
    fullData = JSON.parse(text);
    console.log(fullData)
    splitPoint = Math.floor(fullData.length * trainSplit)

    let kX = []; let ky = []; let ktX = []; let kty = [] 

    // prepare train data
    for (let i = 0; i < splitPoint; i++){
        let temp = []
        for(let j = 0; j < fullData[i].length-1; j++){
            temp.push(fullData[i][j])
        }
        kX.push(temp)
        let label = fullData[i][3]
        if (label == 0){
            ky.push([1,0])
        } else {
            ky.push([0,1])
        }
    }

    // prepare test data
    for (let i = splitPoint; i < fullData.length; i++){
        let temp = []
        for(let j = 0; j < fullData[i].length-1; j++){
            temp.push(fullData[i][j])
        }
        ktX.push(temp)
        let label = fullData[i][3]
        if (label == 0){
            kty.push([1,0])
        } else {
            kty.push([0,1])
        }
    }

    X = tf.tensor2d(kX)
    y = tf.tensor2d(ky)
    tX = tf.tensor2d(ktX)
    ty = tf.tensor2d(kty)

    const opt = tf.train.adam(learningRate = lr, beta2 = lr/epoch)

    const config = {
        optimizer: opt,
        loss: "categoricalCrossentropy"
    }

    console.log(JSON.stringify(model.outputs[0].shape));

    model.compile(config)

    console.log(model)
    train(model).then(() => console.log("Training Done"))
})*/

function run() {
    const opt = tf.train.adam(learningRate = lr, beta2 = lr/epoch)

    const model = tf.sequential({
        layers:[
            tf.layers.dense({units: 100, activation: 'relu', inputShape: [3]}),
            tf.layers.dense({units: 300, activation: 'selu'}),
            tf.layers.dense({units: 50, activation: 'selu'}),
            tf.layers.dense({units: 20, activation: 'selu'}),
            tf.layers.dense({units: 2, activation: 'softmax'})
        ]
    });

    const config = {
        optimizer: opt,
        loss: "categoricalCrossentropy"
    }

    console.log(JSON.stringify(model.outputs[0].shape));

    model.compile(config)

    train().then(() => console.log("Training Done"))
}