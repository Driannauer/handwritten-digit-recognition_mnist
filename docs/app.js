import { MnistCnn, getTopPrediction } from "./model.js";

const drawCanvas = document.querySelector("#draw-canvas");
const drawContext = drawCanvas.getContext("2d", { willReadFrequently: true });
const previewCanvas = document.querySelector("#preview-canvas");
const previewContext = previewCanvas.getContext("2d");
const statusText = document.querySelector("#model-status");
const predictButton = document.querySelector("#predict-button");
const clearButton = document.querySelector("#clear-button");
const uploadInput = document.querySelector("#upload-input");
const brushSize = document.querySelector("#brush-size");
const resultDigit = document.querySelector("#result-digit");
const resultConfidence = document.querySelector("#result-confidence");
const probabilityList = document.querySelector("#probabilities");
const sampleGrid = document.querySelector("#sample-grid");
const sourceLabel = document.querySelector("#source-label");

let model = null;
let isDrawing = false;
let lastPoint = null;
let lastInput = null;

const canvasSize = drawCanvas.width;

function setStatus(message, tone = "neutral") {
  statusText.textContent = message;
  statusText.dataset.tone = tone;
}

function resetDrawingCanvas() {
  drawContext.fillStyle = "#050806";
  drawContext.fillRect(0, 0, canvasSize, canvasSize);
  drawContext.lineCap = "round";
  drawContext.lineJoin = "round";
  drawContext.strokeStyle = "#ffffff";
  drawContext.shadowColor = "rgba(255,255,255,0.22)";
  drawContext.shadowBlur = 3;
  sourceLabel.textContent = "画板输入";
  lastInput = null;
  renderPreview(new Float32Array(28 * 28));
}

function getCanvasPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  const pointer = event.touches ? event.touches[0] : event;
  return {
    x: ((pointer.clientX - rect.left) / rect.width) * canvasSize,
    y: ((pointer.clientY - rect.top) / rect.height) * canvasSize,
  };
}

function drawLine(from, to) {
  drawContext.lineWidth = Number(brushSize.value);
  drawContext.beginPath();
  drawContext.moveTo(from.x, from.y);
  drawContext.lineTo(to.x, to.y);
  drawContext.stroke();
}

function startDrawing(event) {
  event.preventDefault();
  isDrawing = true;
  lastPoint = getCanvasPoint(event);
  drawLine(lastPoint, lastPoint);
}

function continueDrawing(event) {
  if (!isDrawing) {
    return;
  }
  event.preventDefault();
  const point = getCanvasPoint(event);
  drawLine(lastPoint, point);
  lastPoint = point;
}

function stopDrawing() {
  isDrawing = false;
  lastPoint = null;
}

function sourceToInk(sourceCanvas) {
  const context = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const { width, height } = sourceCanvas;
  const imageData = context.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  const borderValues = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const value = (0.2126 * imageData[offset] + 0.7152 * imageData[offset + 1] + 0.0722 * imageData[offset + 2]) / 255;
      gray[y * width + x] = value;
      if (x < 4 || y < 4 || x >= width - 4 || y >= height - 4) {
        borderValues.push(value);
      }
    }
  }

  borderValues.sort((a, b) => a - b);
  const borderMedian = borderValues[Math.floor(borderValues.length / 2)] ?? 0;
  const darkInk = borderMedian > 0.55;
  const ink = new Float32Array(width * height);
  let sum = 0;
  let sumSquares = 0;

  for (let index = 0; index < gray.length; index += 1) {
    const value = darkInk ? 1 - gray[index] : gray[index];
    ink[index] = Math.max(0, Math.min(1, value));
    sum += ink[index];
    sumSquares += ink[index] * ink[index];
  }

  const mean = sum / ink.length;
  const variance = Math.max(0, sumSquares / ink.length - mean * mean);
  const threshold = Math.max(0.08, Math.min(0.55, mean + Math.sqrt(variance) * 0.55));
  return { ink, width, height, threshold };
}

function bilinear(source, width, height, x, y) {
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const x1 = Math.max(0, Math.min(width - 1, x0 + 1));
  const y1 = Math.max(0, Math.min(height - 1, y0 + 1));
  const dx = x - x0;
  const dy = y - y0;
  const a = source[y0 * width + x0] * (1 - dx) + source[y0 * width + x1] * dx;
  const b = source[y1 * width + x0] * (1 - dx) + source[y1 * width + x1] * dx;
  return a * (1 - dy) + b * dy;
}

function shiftInput(input, shiftX, shiftY) {
  const output = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      output[y * 28 + x] = bilinear(input, 28, 28, x - shiftX, y - shiftY);
    }
  }
  return output;
}

function preprocessCanvas(sourceCanvas) {
  const { ink, width, height, threshold } = sourceToInk(sourceCanvas);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = ink[y * width + x];
      if (value > threshold) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return new Float32Array(28 * 28);
  }

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const scale = Math.min(20 / cropWidth, 20 / cropHeight);
  const targetWidth = Math.max(1, Math.round(cropWidth * scale));
  const targetHeight = Math.max(1, Math.round(cropHeight * scale));
  const left = Math.floor((28 - targetWidth) / 2);
  const top = Math.floor((28 - targetHeight) / 2);
  const normalized = new Float32Array(28 * 28);
  let maxValue = 0;

  for (let ty = 0; ty < targetHeight; ty += 1) {
    for (let tx = 0; tx < targetWidth; tx += 1) {
      const sourceX = minX + (tx + 0.5) / scale - 0.5;
      const sourceY = minY + (ty + 0.5) / scale - 0.5;
      const value = bilinear(ink, width, height, sourceX, sourceY);
      normalized[(top + ty) * 28 + left + tx] = value;
      maxValue = Math.max(maxValue, value);
    }
  }

  if (maxValue > 0) {
    for (let index = 0; index < normalized.length; index += 1) {
      normalized[index] = Math.min(1, normalized[index] / maxValue);
    }
  }

  let mass = 0;
  let cx = 0;
  let cy = 0;
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      const value = normalized[y * 28 + x];
      mass += value;
      cx += x * value;
      cy += y * value;
    }
  }

  if (mass > 0) {
    return shiftInput(normalized, 13.5 - cx / mass, 13.5 - cy / mass);
  }
  return normalized;
}

function renderPreview(input) {
  const image = previewContext.createImageData(28, 28);
  for (let index = 0; index < input.length; index += 1) {
    const value = Math.round(Math.max(0, Math.min(1, input[index])) * 255);
    image.data[index * 4] = value;
    image.data[index * 4 + 1] = value;
    image.data[index * 4 + 2] = value;
    image.data[index * 4 + 3] = 255;
  }

  const scratch = document.createElement("canvas");
  scratch.width = 28;
  scratch.height = 28;
  scratch.getContext("2d").putImageData(image, 0, 0);
  previewContext.imageSmoothingEnabled = false;
  previewContext.fillStyle = "#050806";
  previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewContext.drawImage(scratch, 0, 0, previewCanvas.width, previewCanvas.height);
}

function renderProbabilities(probabilities) {
  probabilityList.replaceChildren();
  for (let digit = 0; digit < 10; digit += 1) {
    const value = probabilities[digit] ?? 0;
    const item = document.createElement("div");
    item.className = "probability";
    item.innerHTML = `
      <span class="probability__digit">${digit}</span>
      <span class="probability__track"><span style="width: ${(value * 100).toFixed(2)}%"></span></span>
      <span class="probability__value">${(value * 100).toFixed(1)}%</span>
    `;
    probabilityList.append(item);
  }
}

function predictCurrent() {
  if (!model) {
    setStatus("模型还在加载", "warn");
    return;
  }

  const input = lastInput ?? preprocessCanvas(drawCanvas);
  lastInput = input;
  renderPreview(input);

  const probabilities = model.predict(input);
  const top = getTopPrediction(probabilities);
  resultDigit.textContent = String(top.digit);
  resultConfidence.textContent = `${(top.confidence * 100).toFixed(2)}%`;
  renderProbabilities(probabilities);
  setStatus("浏览器端推理完成", "ready");
}

async function loadImageToCanvas(source) {
  const image = new Image();
  image.decoding = "async";
  image.src = source;
  await image.decode();

  drawContext.fillStyle = "#050806";
  drawContext.fillRect(0, 0, canvasSize, canvasSize);
  drawContext.shadowBlur = 0;

  const scale = Math.min(canvasSize / image.width, canvasSize / image.height);
  const width = image.width * scale;
  const height = image.height * scale;
  const left = (canvasSize - width) / 2;
  const top = (canvasSize - height) / 2;
  drawContext.drawImage(image, left, top, width, height);
  drawContext.shadowBlur = 3;
  lastInput = preprocessCanvas(drawCanvas);
  renderPreview(lastInput);
}

async function handleUpload(event) {
  const [file] = event.target.files;
  if (!file) {
    return;
  }
  const url = URL.createObjectURL(file);
  try {
    await loadImageToCanvas(url);
    sourceLabel.textContent = file.name;
    predictCurrent();
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function loadSamples() {
  const response = await fetch("assets/samples/samples.json");
  if (!response.ok) {
    return;
  }
  const { samples } = await response.json();
  sampleGrid.replaceChildren();

  for (const sample of samples) {
    const button = document.createElement("button");
    button.className = "sample";
    button.type = "button";
    button.innerHTML = `
      <img alt="样例 ${sample.label}" src="${sample.image}">
      <span>${sample.label}</span>
    `;
    button.addEventListener("click", async () => {
      await loadImageToCanvas(sample.image);
      sourceLabel.textContent = `样例 ${sample.name}`;
      predictCurrent();
    });
    sampleGrid.append(button);
  }
}

async function boot() {
  resetDrawingCanvas();
  renderProbabilities(new Float32Array(10));

  try {
    setStatus("正在加载模型权重", "neutral");
    [model] = await Promise.all([MnistCnn.load(), loadSamples()]);
    setStatus("模型已就绪", "ready");
  } catch (error) {
    console.error(error);
    setStatus("模型加载失败，请通过本地服务或 Pages 打开", "warn");
  }
}

drawCanvas.addEventListener("pointerdown", startDrawing);
drawCanvas.addEventListener("pointermove", continueDrawing);
window.addEventListener("pointerup", stopDrawing);
drawCanvas.addEventListener("pointerleave", stopDrawing);
predictButton.addEventListener("click", predictCurrent);
clearButton.addEventListener("click", () => {
  resetDrawingCanvas();
  resultDigit.textContent = "-";
  resultConfidence.textContent = "0.00%";
  renderProbabilities(new Float32Array(10));
  setStatus(model ? "模型已就绪" : "正在加载模型权重", model ? "ready" : "neutral");
});
uploadInput.addEventListener("change", handleUpload);

boot();
