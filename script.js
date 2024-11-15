let data = [];
let labels = [];
let isDrawing = false;
let model;

// Haalt de grid-container element op
const gridContainer = document.getElementById('grid-container');

// Maakt een grid van 100 cellen
for (let i = 0; i < 100; i++) {
    const cell = document.createElement('div');
    cell.classList.add('cell');
    // Voeg event listener toe voor mousedown
    cell.addEventListener('mousedown', () => {
        isDrawing = true;
        activateCell(cell);
    });
    // Voeg event listener toe voor mouseenter
    cell.addEventListener('mouseenter', () => {
        if (isDrawing) activateCell(cell);
    });
    gridContainer.appendChild(cell);
}

// Stopt met tekenen wanneer de muisknop wordt losgelaten
document.body.addEventListener('mouseup', () => (isDrawing = false));

// Activeert een cel
function activateCell(cell) {
    cell.classList.add('active');
}

// Maakt het grid leeg
function clearGrid() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => cell.classList.remove('active'));
}

// Voegt event listener toe aan de knop om het grid te wissen
document.getElementById('clear-grid').addEventListener('click', clearGrid);

// Haalt de data van het grid op (1 voor actieve cellen, 0 voor inactieve cellen)
function getGridData() {
    const cells = document.querySelectorAll('.cell');
    return Array.from(cells).map(cell => cell.classList.contains('active') ? 1 : 0);
}

// Voegt event listener toe aan de knop om een cijfer op te slaan
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

// Traint het model met de opgeslagen data
async function trainModel() {
    if (data.length === 0) {
        alert('Sla eerst wat cijfers op');
        return;
    }

    const xs = tf.tensor2d(data);
    const ys = tf.tensor1d(labels, 'int32');
    const ysOneHot = tf.oneHot(ys, 10);

    // Maakt een nieuw model als het nog niet bestaat
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

    // Traint het model
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

// Voegt event listener toe aan de knop om een voorspelling te doen
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