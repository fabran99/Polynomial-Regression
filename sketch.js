let x_vals = [];
let y_vals = [];

let a, b, c, d;
let dragging = false;

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

// Inicializo las variables
function setup() {
  createCanvas(400, 400);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
}

// La funcion de loss es mean_square_error
function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^3 + bx^2 + cx + d
  const ys = xs
    .pow(tf.scalar(3))
    .mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d);
  return ys;
}

function mousePressed() {
  dragging = true;
}

function mouseReleased() {
  dragging = false;
}

function draw() {
  // Mientras arrastro agrego los puntos que esten dentro del canvas
  if (dragging) {
    if (mouseX > 0 && mouseX < 400 && mouseY > 0 && mouseY < 400) {
      let x = map(mouseX, 0, width, -1, 1);
      let y = map(mouseY, 0, height, 1, -1);
      x_vals.push(x);
      y_vals.push(y);
    }
  } else {
    // Si no estoy arrastrando entreno con los puntos que esten guardados
    tf.tidy(() => {
      if (x_vals.length > 0) {
        console.log(
          a.dataSync()[0],
          b.dataSync()[0],
          c.dataSync()[0],
          d.dataSync()[0]
        );
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
      }
    });
  }

  background(0);

  // Muestro los puntos
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

  // Predigo los puntos para cada x
  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  // Muestro la forma de la curva
  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();
}

// Boton de reseteo
var button = document.getElementById("reset");
const resetBackground = () => {
  x_vals = [];
  y_vals = [];
};

button.onclick = resetBackground;
