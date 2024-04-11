import { Matrix4 } from "potree";
import * as Timer from "../renderer/Timer.js";

const vs = `
struct Uniforms {
	worldView : mat4x4<f32>,
	proj : mat4x4<f32>,
};

struct VertexInput{
	@location(0) pos_point : vec4<f32>,
	@location(1) color : vec4<f32>,
};

struct VertexOutput{
	@builtin(position) pos : vec4<f32>,
	@location(0) color : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn renderPoints(vertex : VertexInput) -> VertexOutput {

	var viewPos : vec4<f32> = uniforms.worldView * vertex.pos_point;
	var c : vec4<f32> = vertex.color;

	var vout : VertexOutput;
	vout.pos = uniforms.proj * viewPos;
	vout.color = c;

	return vout;
}
`;

const fs = `

struct FragmentInput{
	@location(0) color : vec4<f32>,
};

@fragment
fn renderPoints(fragment : FragmentInput) -> @location(0) vec4<f32> {
	return fragment.color;
}
`;

let initialized = false;

// let uniformBufferNumElements = 1000;
// let uniformBufferCapacity = uniformBufferNumElements * 256;
// let uniformBuffer = null;
// let uniformBufferData = new ArrayBuffer(uniformBufferCapacity);
let states = new Map();
let pipeline = null;

function init(renderer) {
  if (initialized) {
    return;
  }

  let { device } = renderer;
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: vs }),
      entryPoint: "renderPoints",
      buffers: [
        {
          // position
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
        {
          // color
          arrayStride: 4,
          stepMode: "vertex",
          attributes: [
            {
              shaderLocation: 1,
              offset: 0,
              format: "unorm8x4",
            },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: fs }),
      entryPoint: "renderPoints",
      targets: [{ format: "bgra8unorm" }],
    },
    primitive: {
      topology: "point-list",
      cullMode: "back",
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "greater",
      format: "depth32float",
    },
  });

  // uniformBuffer = device.createBuffer({
  // 	size: uniformBufferCapacity,
  // 	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  // });

  // bindGroup = device.createBindGroup({
  // 	layout: pipeline.getBindGroupLayout(0),
  // 	entries: [
  // 		{
  // 			binding: 0, resource: {
  // 				buffer: uniformBuffer,
  // 				offset: 0,
  // 				size: 256,
  // 			}
  // 		},
  // 	],
  // });
}

function getState(renderer, node) {
  let { device } = renderer;

  if (!states.has(node)) {
    let uniformBuffer = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    let state = { uniformBuffer, bindGroup };

    states.set(node, state);
  }

  return states.get(node);
}

export function render(nodes, drawstate) {
  let { renderer, pass } = drawstate;
  let { passEncoder } = pass;
  let { device } = renderer;

  init(renderer);

  Timer.timestamp(passEncoder, "points-start");

  passEncoder.setPipeline(pipeline);

  let i = 0;
  for (let node of nodes) {
    let state = getState(renderer, node);
    let { bindGroup } = state;

    {
      // update uniforms
      let { uniformBuffer } = state;

      let data = new ArrayBuffer(256);
      let f32 = new Float32Array(data);

      {
        // transform
        let world = node.world;
        let view = camera.view;
        let worldView = new Matrix4().multiplyMatrices(view, world);

        f32.set(worldView.elements, 0);
        f32.set(camera.proj.elements, 16);
      }

      device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
    }

    passEncoder.setBindGroup(0, bindGroup);

    let vboPosition = renderer.getGpuBuffer(
      node.geometry.buffers.find((s) => s.name === "position").buffer
    );
    let vboColor = renderer.getGpuBuffer(
      node.geometry.buffers.find((s) => s.name === "rgba").buffer
    );

    passEncoder.setVertexBuffer(0, vboPosition);
    passEncoder.setVertexBuffer(1, vboColor);

    let numElements = node.geometry.numElements;
    passEncoder.draw(numElements, 1, 0, 0);

    i++;
  }

  Timer.timestamp(passEncoder, "points-end");
}
