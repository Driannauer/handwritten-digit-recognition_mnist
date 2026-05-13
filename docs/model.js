const RELU = true;
const LINEAR = false;

function weightSize(shape) {
  return shape.reduce((total, value) => total * value, 1);
}

function topIndex(values) {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) {
      best = index;
    }
  }
  return best;
}

export class MnistCnn {
  constructor(manifest, weightsBuffer) {
    this.manifest = manifest;
    this.allWeights = new Float32Array(weightsBuffer);
    this.weights = new Map();

    for (const [name, meta] of Object.entries(manifest.weights)) {
      const expected = weightSize(meta.shape);
      if (expected !== meta.size) {
        throw new Error(`Bad weight shape for ${name}`);
      }
      this.weights.set(name, this.allWeights.subarray(meta.offset, meta.offset + meta.size));
    }
  }

  static async load(basePath = "assets/model") {
    const [manifestResponse, weightsResponse] = await Promise.all([
      fetch(`${basePath}/model.json`),
      fetch(`${basePath}/weights.bin`),
    ]);

    if (!manifestResponse.ok) {
      throw new Error(`Cannot load model manifest (${manifestResponse.status})`);
    }
    if (!weightsResponse.ok) {
      throw new Error(`Cannot load model weights (${weightsResponse.status})`);
    }

    const manifest = await manifestResponse.json();
    const weightsBuffer = await weightsResponse.arrayBuffer();
    return new MnistCnn(manifest, weightsBuffer);
  }

  w(name) {
    const value = this.weights.get(name);
    if (!value) {
      throw new Error(`Missing model weight: ${name}`);
    }
    return value;
  }

  predict(inputPixels) {
    if (!inputPixels || inputPixels.length !== 28 * 28) {
      throw new Error("Model input must contain 784 normalized pixels.");
    }

    let tensor = {
      data: inputPixels instanceof Float32Array ? inputPixels : Float32Array.from(inputPixels),
      height: 28,
      width: 28,
      channels: 1,
    };

    tensor = this.conv2dSame(tensor, "conv2d", 32, RELU);
    tensor = this.batchNorm(tensor, "batch_normalization");
    tensor = this.conv2dSame(tensor, "conv2d_1", 32, RELU);
    tensor = this.batchNorm(tensor, "batch_normalization_1");
    tensor = this.maxPool2d(tensor);
    tensor = this.conv2dSame(tensor, "conv2d_2", 64, RELU);
    tensor = this.batchNorm(tensor, "batch_normalization_2");
    tensor = this.conv2dSame(tensor, "conv2d_3", 64, RELU);
    tensor = this.batchNorm(tensor, "batch_normalization_3");
    tensor = this.maxPool2d(tensor);
    tensor = this.conv2dSame(tensor, "conv2d_4", 128, RELU);
    tensor = this.batchNorm(tensor, "batch_normalization_4");

    let vector = tensor.data;
    vector = this.dense(vector, "dense", 128, RELU);
    vector = this.batchNormVector(vector, "batch_normalization_5");
    vector = this.dense(vector, "dense_1", 10, LINEAR);

    return this.softmax(vector);
  }

  conv2dSame(input, layerName, outChannels, useRelu) {
    const kernel = this.w(`${layerName}/kernel`);
    const bias = this.w(`${layerName}/bias`);
    const { data, height, width, channels } = input;
    const output = new Float32Array(height * width * outChannels);

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        for (let out = 0; out < outChannels; out += 1) {
          let sum = bias[out];

          for (let ky = 0; ky < 3; ky += 1) {
            const inY = y + ky - 1;
            if (inY < 0 || inY >= height) {
              continue;
            }
            for (let kx = 0; kx < 3; kx += 1) {
              const inX = x + kx - 1;
              if (inX < 0 || inX >= width) {
                continue;
              }
              const inputBase = (inY * width + inX) * channels;
              const kernelBase = ((ky * 3 + kx) * channels) * outChannels + out;
              for (let channel = 0; channel < channels; channel += 1) {
                sum += data[inputBase + channel] * kernel[kernelBase + channel * outChannels];
              }
            }
          }

          output[(y * width + x) * outChannels + out] = useRelu && sum < 0 ? 0 : sum;
        }
      }
    }

    return { data: output, height, width, channels: outChannels };
  }

  batchNorm(input, layerName) {
    const gamma = this.w(`${layerName}/gamma`);
    const beta = this.w(`${layerName}/beta`);
    const mean = this.w(`${layerName}/moving_mean`);
    const variance = this.w(`${layerName}/moving_variance`);
    const { data, channels } = input;
    const output = new Float32Array(data.length);

    for (let index = 0; index < data.length; index += channels) {
      for (let channel = 0; channel < channels; channel += 1) {
        output[index + channel] =
          gamma[channel] * ((data[index + channel] - mean[channel]) / Math.sqrt(variance[channel] + 0.001)) +
          beta[channel];
      }
    }

    return { ...input, data: output };
  }

  maxPool2d(input) {
    const { data, height, width, channels } = input;
    const outHeight = Math.floor(height / 2);
    const outWidth = Math.floor(width / 2);
    const output = new Float32Array(outHeight * outWidth * channels);

    for (let y = 0; y < outHeight; y += 1) {
      for (let x = 0; x < outWidth; x += 1) {
        for (let channel = 0; channel < channels; channel += 1) {
          let value = -Infinity;
          for (let py = 0; py < 2; py += 1) {
            for (let px = 0; px < 2; px += 1) {
              const source = (((y * 2 + py) * width + (x * 2 + px)) * channels) + channel;
              value = Math.max(value, data[source]);
            }
          }
          output[(y * outWidth + x) * channels + channel] = value;
        }
      }
    }

    return { data: output, height: outHeight, width: outWidth, channels };
  }

  dense(input, layerName, units, useRelu) {
    const kernel = this.w(`${layerName}/kernel`);
    const bias = this.w(`${layerName}/bias`);
    const output = new Float32Array(units);

    for (let unit = 0; unit < units; unit += 1) {
      let sum = bias[unit];
      for (let index = 0; index < input.length; index += 1) {
        sum += input[index] * kernel[index * units + unit];
      }
      output[unit] = useRelu && sum < 0 ? 0 : sum;
    }

    return output;
  }

  batchNormVector(input, layerName) {
    const gamma = this.w(`${layerName}/gamma`);
    const beta = this.w(`${layerName}/beta`);
    const mean = this.w(`${layerName}/moving_mean`);
    const variance = this.w(`${layerName}/moving_variance`);
    const output = new Float32Array(input.length);

    for (let index = 0; index < input.length; index += 1) {
      output[index] =
        gamma[index] * ((input[index] - mean[index]) / Math.sqrt(variance[index] + 0.001)) + beta[index];
    }

    return output;
  }

  softmax(logits) {
    let maxValue = -Infinity;
    for (const value of logits) {
      maxValue = Math.max(maxValue, value);
    }

    let total = 0;
    const output = new Float32Array(logits.length);
    for (let index = 0; index < logits.length; index += 1) {
      const value = Math.exp(logits[index] - maxValue);
      output[index] = value;
      total += value;
    }
    for (let index = 0; index < output.length; index += 1) {
      output[index] /= total;
    }
    return output;
  }
}

export function getTopPrediction(probabilities) {
  const index = topIndex(probabilities);
  return {
    digit: index,
    confidence: probabilities[index],
  };
}
