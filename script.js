let data = [];
let labels = [];
let isDrawing = false;
let model;

const gridContainer = document.getElementById('grid-container');
for (let i = 0; i < 100; i++) {
    const cell = document.createElement('div');
    cell.classList.add('cell');
    cell.addEventListener('mousedown', () => {
        isDrawing = true;
        activateCell(cell);
    });
    cell.addEventListener('mouseenter', () => {
        if (isDrawing) activateCell(cell);
    });
    gridContainer.appendChild(cell);
}
document.body.addEventListener('mouseup', () => (isDrawing = false));

function activateCell(cell) {
    cell.classList.add('active');
}

function clearGrid() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => cell.classList.remove('active'));
}

document.getElementById('clear-grid').addEventListener('click', clearGrid);

function getGridData() {
    const cells = document.querySelectorAll('.cell');
    return Array.from(cells).map(cell => cell.classList.contains('active') ? 1 : 0);
}

document.getElementById('save-digit').addEventListener('click', async () => {
    const gridData = getGridData();
    const label = parseInt(document.getElementById('label-input').value);
    if (isNaN(label) || label < 0 || label > 9) {
        alert("Voer een geldig cijfer in (0-9)");
        return;
    }
    data.push(gridData);
    labels.push(label);
    clearGrid();
    await trainModel();
});

async function trainModel() {
    if (data.length === 0) {
        alert('Sla eerst wat cijfers op');
        return;
    }

    const xs = tf.tensor2d(data);
    const ys = tf.tensor1d(labels, 'int32');
    const ysOneHot = tf.oneHot(ys, 10);

    if (!model) {
        model = tf.sequential();
        model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [100] }));
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    await model.fit(xs, ysOneHot, {
        epochs: 10,
        batchSize: 16,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}`);
            }
        }
    });

    xs.dispose();
    ys.dispose();
    ysOneHot.dispose();
}

document.getElementById('predict-digit').addEventListener('click', async () => {
    if (!model) {
        alert('Train het model eerst');
        return;
    }
    const input = tf.tensor2d([getGridData()]);
    const prediction = model.predict(input);
    const predictedIndex = prediction.argMax(1).dataSync()[0];
    document.getElementById('output').innerText = `Voorspelling: ${predictedIndex}`;
});
