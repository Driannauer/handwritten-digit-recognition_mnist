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
let lastCandidate = null;

const canvasSize = drawCanvas.width;
const candidateRotations = [0, 90, 270, 180];
const localPreprocessSigma = 18;
const localPreprocessThreshold = 30;

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
  lastCandidate = null;
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
  lastCandidate = null;
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

function sourceToBrightness(sourceCanvas) {
  const context = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const { width, height } = sourceCanvas;
  const imageData = context.getImageData(0, 0, width, height).data;
  const bright = new Float32Array(width * height);
  const borderValues = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const value = Math.max(imageData[offset], imageData[offset + 1], imageData[offset + 2]);
      bright[y * width + x] = value;
      if (x < 4 || y < 4 || x >= width - 4 || y >= height - 4) {
        borderValues.push(value / 255);
      }
    }
  }

  borderValues.sort((a, b) => a - b);
  const borderMedian = borderValues[Math.floor(borderValues.length / 2)] ?? 0;
  if (borderMedian < 0.45) {
    for (let index = 0; index < bright.length; index += 1) {
      bright[index] = 255 - bright[index];
    }
  }
  return { bright, width, height };
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

function recenterInput(input) {
  let mass = 0;
  let cx = 0;
  let cy = 0;
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      const value = input[y * 28 + x];
      mass += value;
      cx += x * value;
      cy += y * value;
    }
  }

  if (mass <= 0) {
    return input;
  }

  const shifted = shiftInput(input, 13.5 - cx / mass, 13.5 - cy / mass);
  for (let index = 0; index < shifted.length; index += 1) {
    shifted[index] = Math.max(0, Math.min(1, shifted[index]));
  }
  return shifted;
}

function reflectIndex(index, length) {
  if (length <= 1) {
    return 0;
  }
  let reflected = index;
  while (reflected < 0 || reflected >= length) {
    reflected = reflected < 0 ? -reflected - 1 : 2 * length - reflected - 1;
  }
  return reflected;
}

function gaussianKernel(sigma) {
  const radius = Math.ceil(sigma * 3);
  const kernel = new Float32Array(radius * 2 + 1);
  let total = 0;
  for (let offset = -radius; offset <= radius; offset += 1) {
    const value = Math.exp(-(offset * offset) / (2 * sigma * sigma));
    kernel[offset + radius] = value;
    total += value;
  }
  for (let index = 0; index < kernel.length; index += 1) {
    kernel[index] /= total;
  }
  return { kernel, radius };
}

function gaussianBlur(source, width, height, sigma) {
  const { kernel, radius } = gaussianKernel(sigma);
  const temp = new Float32Array(source.length);
  const output = new Float32Array(source.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let value = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        value += source[y * width + reflectIndex(x + offset, width)] * kernel[offset + radius];
      }
      temp[y * width + x] = value;
    }
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let value = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        value += temp[reflectIndex(y + offset, height) * width + x] * kernel[offset + radius];
      }
      output[y * width + x] = value;
    }
  }

  return output;
}

function rotateInk(ink, width, height, rotationDegrees) {
  const normalizedRotation = ((rotationDegrees % 360) + 360) % 360;
  if (normalizedRotation === 0) {
    return { ink, width, height };
  }

  const targetWidth = normalizedRotation === 180 ? width : height;
  const targetHeight = normalizedRotation === 180 ? height : width;
  const output = new Float32Array(targetWidth * targetHeight);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let targetX = x;
      let targetY = y;
      if (normalizedRotation === 90) {
        targetX = height - 1 - y;
        targetY = x;
      } else if (normalizedRotation === 180) {
        targetX = width - 1 - x;
        targetY = height - 1 - y;
      } else if (normalizedRotation === 270) {
        targetX = y;
        targetY = width - 1 - x;
      }
      output[targetY * targetWidth + targetX] = ink[y * width + x];
    }
  }

  return { ink: output, width: targetWidth, height: targetHeight };
}

function emptyCandidate(rotationDegrees) {
  return {
    input: new Float32Array(28 * 28),
    rotationDegrees,
    orientationScore: 0,
    qualityScore: 0,
  };
}

function scoreProcessedDigitQuality(input) {
  let minX = 28;
  let minY = 28;
  let maxX = -1;
  let maxY = -1;
  let pixelCount = 0;

  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      if (input[y * 28 + x] > 0.05) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        pixelCount += 1;
      }
    }
  }

  if (pixelCount < 8 || maxX < minX || maxY < minY) {
    return 0;
  }

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  if (height < 4 || width < 2) {
    return 0;
  }

  const bboxArea = Math.max(1, width * height);
  const fillRatio = pixelCount / bboxArea;
  const sizeScore = Math.min(1, Math.max(0.2, bboxArea / 160));
  let fillScore = 1;
  if (fillRatio < 0.10) {
    fillScore = Math.max(0.25, fillRatio / 0.10);
  } else if (fillRatio > 0.62) {
    fillScore = Math.max(0.18, 1 - (fillRatio - 0.62) / 0.30);
  }

  const aspectRatio = Math.max(height, width) / Math.max(1, Math.min(height, width));
  let aspectScore = 0.85 + 0.15 * Math.min(1, aspectRatio / 4);
  if (height >= width * 1.8) {
    aspectScore *= 1.08;
  }

  const margin = Math.min(minY, minX, 27 - maxY, 27 - maxX);
  const borderScore = margin <= 0 ? 0.75 : 1;
  return Math.max(0, Math.min(1, sizeScore * fillScore * aspectScore * borderScore));
}

function inputSum(input) {
  let total = 0;
  for (const value of input) {
    total += value;
  }
  return total;
}

function candidateIsBlank(candidate) {
  return candidate.qualityScore <= 0 || inputSum(candidate.input) < 0.5;
}

function findBestComponent(ink, width, height, threshold) {
  const visited = new Uint8Array(ink.length);
  const stack = new Int32Array(ink.length);
  let bestComponent = null;
  let bestScore = -1;

  for (let start = 0; start < ink.length; start += 1) {
    if (visited[start] || ink[start] <= threshold) {
      continue;
    }

    let top = 0;
    stack[top] = start;
    top += 1;
    visited[start] = 1;
    let area = 0;
    let inkStrength = 0;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    while (top > 0) {
      top -= 1;
      const index = stack[top];
      const x = index % width;
      const y = (index - x) / width;
      const value = ink[index];
      area += 1;
      inkStrength += value;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      const neighbors = [
        x > 0 ? index - 1 : -1,
        x < width - 1 ? index + 1 : -1,
        y > 0 ? index - width : -1,
        y < height - 1 ? index + width : -1,
      ];
      for (const neighbor of neighbors) {
        if (neighbor >= 0 && !visited[neighbor] && ink[neighbor] > threshold) {
          visited[neighbor] = 1;
          stack[top] = neighbor;
          top += 1;
        }
      }
    }

    if (area < 30) {
      continue;
    }

    const componentHeight = maxY - minY + 1;
    const componentWidth = maxX - minX + 1;
    const aspectRatio = Math.max(componentHeight, componentWidth) / Math.max(1, Math.min(componentHeight, componentWidth));
    if (aspectRatio > 12) {
      continue;
    }
    if ((minY <= 2 || minX <= 2 || maxY >= height - 3 || maxX >= width - 3) && aspectRatio > 4) {
      continue;
    }

    const fillRatio = area / Math.max(1, componentHeight * componentWidth);
    let strokeDensityBonus = Math.min(1, fillRatio / 0.18);
    if (fillRatio > 0.58) {
      strokeDensityBonus *= Math.max(0.2, 1 - (fillRatio - 0.58) / 0.32);
    }

    const centerX = (minX + maxX + 1) / 2;
    const centerY = (minY + maxY + 1) / 2;
    const centerBonus = 1 / (
      1 +
      ((centerX - width / 2) / (width / 2)) ** 2 +
      ((centerY - height / 2) / (height / 2)) ** 2
    );
    const heightWidthRatio = componentHeight / Math.max(1, componentWidth);
    const shapeBonus = heightWidthRatio >= 1
      ? Math.min(1.45, 0.70 + 0.18 * heightWidthRatio)
      : Math.max(0.35, heightWidthRatio);
    const score = inkStrength * strokeDensityBonus * centerBonus * shapeBonus;
    if (score > bestScore) {
      bestScore = score;
      bestComponent = { minX, minY, maxX, maxY, score };
    }
  }

  return bestComponent;
}

function suppressGuideRows(crop, width, height) {
  const rowMeans = new Float32Array(height);
  const rowStds = new Float32Array(height);
  let meanTotal = 0;

  for (let y = 0; y < height; y += 1) {
    let rowTotal = 0;
    for (let x = 0; x < width; x += 1) {
      rowTotal += crop[y * width + x];
    }
    rowMeans[y] = rowTotal / width;
    meanTotal += rowMeans[y];
  }

  const meanAverage = meanTotal / height;
  let meanVariance = 0;
  for (let y = 0; y < height; y += 1) {
    meanVariance += (rowMeans[y] - meanAverage) ** 2;
    let rowVariance = 0;
    for (let x = 0; x < width; x += 1) {
      rowVariance += (crop[y * width + x] - rowMeans[y]) ** 2;
    }
    rowStds[y] = Math.sqrt(rowVariance / width);
  }

  const meanStd = Math.sqrt(meanVariance / height);
  for (let y = 0; y < height; y += 1) {
    if (rowMeans[y] > meanAverage + 0.5 * meanStd && rowStds[y] < 25) {
      for (let x = 0; x < width; x += 1) {
        crop[y * width + x] *= 0.15;
      }
    }
  }
}

function maxFilter3(source, width, height) {
  const output = new Float32Array(source.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let value = 0;
      for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy += 1) {
        for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx += 1) {
          value = Math.max(value, source[yy * width + xx]);
        }
      }
      output[y * width + x] = value;
    }
  }
  return output;
}

function resizeToInput(source, width, height) {
  const output = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      const sourceX = (x + 0.5) * (width / 28) - 0.5;
      const sourceY = (y + 0.5) * (height / 28) - 0.5;
      output[y * 28 + x] = Math.max(0, Math.min(1, bilinear(source, width, height, sourceX, sourceY) / 255));
    }
  }
  return output;
}

function digitShapeFeatures(input) {
  let minX = 28;
  let minY = 28;
  let maxX = -1;
  let maxY = -1;
  let upperPixels = 0;
  let lowerPixels = 0;
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      if (input[y * 28 + x] > 0.05) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        if (y < 14) {
          upperPixels += 1;
        } else {
          lowerPixels += 1;
        }
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return null;
  }

  let topWidth = 0;
  let bottomWidth = 0;
  for (let x = 0; x < 28; x += 1) {
    let topHasInk = false;
    let bottomHasInk = false;
    for (let y = minY; y < Math.min(28, minY + 5); y += 1) {
      topHasInk = topHasInk || input[y * 28 + x] > 0.05;
    }
    for (let y = Math.max(0, maxY - 4); y <= maxY; y += 1) {
      bottomHasInk = bottomHasInk || input[y * 28 + x] > 0.05;
    }
    if (topHasInk) {
      topWidth += 1;
    }
    if (bottomHasInk) {
      bottomWidth += 1;
    }
  }

  return {
    width: maxX - minX + 1,
    height: maxY - minY + 1,
    topRow: minY,
    topWidth,
    bottomWidth,
    upperPixels,
    lowerPixels,
  };
}

function buildTopBarBoostCandidate(candidate) {
  if (candidateIsBlank(candidate)) {
    return null;
  }
  const features = digitShapeFeatures(candidate.input);
  if (
    !features ||
    features.width < 8 ||
    features.height < 18 ||
    features.topWidth < Math.max(7, Math.ceil(features.width * 0.75)) ||
    features.topWidth < features.bottomWidth + 5 ||
    features.upperPixels < 0.8 * Math.max(1, features.lowerPixels)
  ) {
    return null;
  }

  const enhanced = Float32Array.from(candidate.input);
  for (let y = features.topRow; y < Math.min(28, features.topRow + 4); y += 1) {
    for (let x = 0; x < 28; x += 1) {
      enhanced[y * 28 + x] = Math.min(1, enhanced[y * 28 + x] * 1.8);
    }
  }
  const input = recenterInput(enhanced);
  return {
    ...candidate,
    input,
    qualityScore: scoreProcessedDigitQuality(input),
    preprocessingAdjustment: "top_bar_boost",
  };
}

function preprocessBrightness(brightData, rotationDegrees = 0) {
  const rotated = rotateInk(brightData.bright, brightData.width, brightData.height, rotationDegrees);
  const { ink: bright, width, height } = rotated;
  const background = gaussianBlur(bright, width, height, localPreprocessSigma);
  const ink = new Float32Array(bright.length);
  let maxInk = 0;

  for (let index = 0; index < bright.length; index += 1) {
    const value = Math.max(0, background[index] - bright[index]);
    ink[index] = value;
    maxInk = Math.max(maxInk, value);
  }

  if (maxInk > 0) {
    for (let index = 0; index < ink.length; index += 1) {
      ink[index] = (ink[index] / maxInk) * 255;
    }
  }

  const bestComponent = findBestComponent(ink, width, height, localPreprocessThreshold);
  if (!bestComponent) {
    return emptyCandidate(rotationDegrees);
  }

  let cropMinX = width;
  let cropMinY = height;
  let cropMaxX = -1;
  let cropMaxY = -1;
  for (let y = bestComponent.minY; y <= bestComponent.maxY; y += 1) {
    for (let x = bestComponent.minX; x <= bestComponent.maxX; x += 1) {
      if (ink[y * width + x] > localPreprocessThreshold) {
        cropMinX = Math.min(cropMinX, x);
        cropMinY = Math.min(cropMinY, y);
        cropMaxX = Math.max(cropMaxX, x);
        cropMaxY = Math.max(cropMaxY, y);
      }
    }
  }

  if (cropMaxX < cropMinX || cropMaxY < cropMinY) {
    return emptyCandidate(rotationDegrees);
  }

  const cropWidth = cropMaxX - cropMinX + 1;
  const cropHeight = cropMaxY - cropMinY + 1;
  const crop = new Float32Array(cropWidth * cropHeight);
  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      const value = ink[(cropMinY + y) * width + cropMinX + x];
      crop[y * cropWidth + x] = value > localPreprocessThreshold ? value : 0;
    }
  }

  suppressGuideRows(crop, cropWidth, cropHeight);

  const side = Math.max(cropWidth, cropHeight) + 12;
  const padded = new Float32Array(side * side);
  const top = Math.floor((side - cropHeight) / 2);
  const left = Math.floor((side - cropWidth) / 2);
  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      padded[(top + y) * side + left + x] = crop[y * cropWidth + x];
    }
  }

  const filtered = maxFilter3(padded, side, side);
  const input = recenterInput(resizeToInput(filtered, side, side));
  return {
    input,
    rotationDegrees,
    orientationScore: bestComponent.score,
    qualityScore: scoreProcessedDigitQuality(input),
  };
}

function preprocessCanvas(sourceCanvas, rotationDegrees = 0) {
  return preprocessBrightness(sourceToBrightness(sourceCanvas), rotationDegrees).input;
}

function preprocessCanvasCandidates(sourceCanvas) {
  const brightData = sourceToBrightness(sourceCanvas);
  const candidates = [];
  for (const rotationDegrees of candidateRotations) {
    const candidate = preprocessBrightness(brightData, rotationDegrees);
    candidates.push(candidate);
    const boostedCandidate = buildTopBarBoostCandidate(candidate);
    if (boostedCandidate) {
      candidates.push(boostedCandidate);
    }
  }
  return candidates;
}

function chooseCandidateForModel(modelInstance, candidates) {
  const usableCandidates = candidates.filter((candidate) => !candidateIsBlank(candidate));
  if (usableCandidates.length === 0) {
    const fallback = candidates[0] ?? emptyCandidate(0);
    return {
      ...fallback,
      probabilities: modelInstance.predict(fallback.input),
    };
  }

  const maxOrientationScore = Math.max(...usableCandidates.map((candidate) => candidate.orientationScore));
  const predictedCandidates = usableCandidates.map((candidate) => {
    const probabilities = modelInstance.predict(candidate.input);
    const top = getTopPrediction(probabilities);
    return { ...candidate, probabilities, top };
  });

  if (maxOrientationScore <= 0) {
    return predictedCandidates[0];
  }

  const originalCandidate = predictedCandidates.find((candidate) => candidate.rotationDegrees === 0) ?? predictedCandidates[0];
  const originalConfidence = originalCandidate.top.confidence;
  if (
    originalConfidence >= 0.85 &&
    originalCandidate.orientationScore >= 0.70 * maxOrientationScore &&
    originalCandidate.qualityScore >= 0.45
  ) {
    let bestAdjustedCandidate = originalCandidate;
    let bestAdjustedConfidence = originalConfidence;
    for (const candidate of predictedCandidates) {
      if (candidate.rotationDegrees !== originalCandidate.rotationDegrees || !candidate.preprocessingAdjustment) {
        continue;
      }
      if (
        candidate.qualityScore >= 0.80 * originalCandidate.qualityScore &&
        candidate.top.confidence >= bestAdjustedConfidence + 0.08
      ) {
        bestAdjustedCandidate = candidate;
        bestAdjustedConfidence = candidate.top.confidence;
      }
    }
    return bestAdjustedCandidate;
  }

  let bestCandidate = predictedCandidates[0];
  let bestScore = -1;
  let bestConfidence = 0;
  for (const candidate of predictedCandidates) {
    const orientationWeight = candidate.orientationScore / maxOrientationScore;
    const combinedScore = candidate.top.confidence * (
      0.35 + 0.30 * orientationWeight + 0.35 * candidate.qualityScore
    );
    if (combinedScore > bestScore) {
      bestScore = combinedScore;
      bestConfidence = candidate.top.confidence;
      bestCandidate = candidate;
    }
  }

  if (
    bestCandidate.rotationDegrees === 180 &&
    originalCandidate.orientationScore >= 0.95 * maxOrientationScore &&
    originalConfidence >= 0.30 &&
    bestConfidence <= 0.75
  ) {
    return originalCandidate;
  }

  return bestCandidate;
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

  const candidate = lastCandidate ?? chooseCandidateForModel(model, preprocessCanvasCandidates(drawCanvas));
  lastCandidate = candidate;
  renderPreview(candidate.input);

  const probabilities = candidate.probabilities ?? model.predict(candidate.input);
  const top = getTopPrediction(probabilities);
  resultDigit.textContent = String(top.digit);
  resultConfidence.textContent = `${(top.confidence * 100).toFixed(2)}%`;
  renderProbabilities(probabilities);
  const rotationText = candidate.rotationDegrees === 0 ? "" : `，自动旋转 ${candidate.rotationDegrees} 度`;
  setStatus(`浏览器端推理完成${rotationText}`, "ready");
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
  lastCandidate = null;
  renderPreview(preprocessCanvas(drawCanvas));
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
