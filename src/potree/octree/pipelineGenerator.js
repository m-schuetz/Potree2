import { generate as generateShaders } from "./shaderGenerator.js";

function getStride(attribute) {
  return 4;
}

function getFormat(attribute) {
  let typename = attribute.type.name;
  let numElements = attribute.numElements;

  if (typename === "uint16" && numElements === 1) {
    return "uint32";
  } else if (typename === "uint16" && numElements === 3) {
    return "unorm8x4";
  } else {
    throw "unsupported type";
  }
}

export function generate(renderer, args = {}) {
  let { device } = renderer;
  let { vs, fs } = generateShaders(args);
  let { flags } = args;

  let depthWrite = true;
  let blend = {
    color: {
      srcFactor: "one",
      dstFactor: "zero",
      operation: "add",
    },
    alpha: {
      srcFactor: "one",
      dstFactor: "zero",
      operation: "add",
    },
  };

  let isAdditive = flags.includes("additive_blending");
  let format = "bgra8unorm";

  // isAdditive = true;
  if (isAdditive) {
    format = "rgba32float";
    depthWrite = false;

    blend = {
      color: {
        srcFactor: "one",
        dstFactor: "one",
        operation: "add",
      },
      alpha: {
        srcFactor: "one",
        dstFactor: "one",
        operation: "add",
      },
    };
  }

  let attributeBufferDescriptor = {
    arrayStride: getStride(args.attribute),
    stepMode: "vertex",
    attributes: [
      {
        shaderLocation: 1,
        offset: 0,
        format: getFormat(args.attribute),
      },
    ],
  };
  // Define an empty pipeline layout
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
          type: "uniform",
        },
      },
      // Add other entries if needed
    ],
  });
  const secondBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // Sampler
        visibility: GPUShaderStage.FRAGMENT, // Assuming it's used in the fragment shader
        sampler: {
          type: "filtering",
        },
      },
      {
        binding: 1, // Texture view
        visibility: GPUShaderStage.FRAGMENT, // Assuming it's used in the fragment shader
        texture: {
          sampleType: "float",
          viewDimension: "2d", // Explicitly set view dimension
          multisampled: false, // Explicitly state multisampling is not used
        },
      },
    ],
  });
  const layout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout, secondBindGroupLayout],
  });
  const pipeline = device.createRenderPipeline({
    layout,
    vertex: {
      module: device.createShaderModule({ code: vs }),
      entryPoint: "main",
      buffers: [
        {
          // point position
          arrayStride: 3 * 4,
          stepMode: "vertex",
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x3",
            },
          ],
        },
        attributeBufferDescriptor,
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: fs }),
      entryPoint: "main",
      targets: [{ format: format, blend: blend }],
    },
    primitive: {
      topology: "point-list",
      cullMode: "none",
    },
    depthStencil: {
      depthWriteEnabled: depthWrite,
      depthCompare: "greater",
      format: "depth32float",
    },
  });

  return pipeline;
}
