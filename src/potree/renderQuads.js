import { Matrix4, Vector3 } from "../math/math.js";
import { SPECTRAL } from "../misc/Gradients.js";
import * as Timer from "../renderer/Timer.js";

const vs = `
struct Uniforms {
	@offset(0) worldView : mat4x4<f32>,
	@offset(64) proj : mat4x4<f32>,
	@offset(128) screen_width : f32,
	@offset(132) screen_height : f32,
	@offset(136) point_size : f32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@location(0) var<in> pos_point : vec4<f32>;
@location(1) var<in> pos_quad : vec4<f32>;
@location(2) var<in> color : vec4<f32>;

@builtin(position) var<out> out_pos : vec4<f32>;
@location(0) var<out> fragColor : vec4<f32>;

@vertex
fn renderQuad() -> void {

	var viewPos : vec4<f32> = uniforms.worldView * pos_point;
	out_pos = uniforms.proj * viewPos;

	var fx : f32 = out_pos.x / out_pos.w;
	fx = fx + uniforms.point_size * pos_quad.x / uniforms.screen_width;
	out_pos.x = fx * out_pos.w;

	var fy : f32 = out_pos.y / out_pos.w;
	fy = fy + uniforms.point_size * pos_quad.y / uniforms.screen_height;
	out_pos.y = fy * out_pos.w;

	fragColor = color;

	return;
}
`;

const fs = `
@location(0) var<in> fragColor : vec4<f32>;
@location(0) var<out> outColor : vec4<f32>;

@fragment
fn renderQuad() -> void {
	outColor = fragColor;
	return;
}
`;

let octreeStates = new Map();
let nodeStates = new Map();

let quad_position = new Float32Array([
  // indexed
  -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0,

  // non-indexed
  // -1.0, -1.0, 0.0,
  //  1.0, -1.0, 0.0,
  //  1.0,  1.0, 0.0,
  // -1.0, -1.0, 0.0,
  //  1.0,  1.0, 0.0,
  // -1.0,  1.0, 0.0,
]);
let quad_elements = new Uint32Array([0, 1, 2, 0, 2, 3]);
let vbo_quad = null;

function getVboQuad(renderer) {
  if (!vbo_quad) {
    let geometry = {
      buffers: [
        {
          name: "position",
          buffer: quad_position,
        },
        {
          name: "elements",
          buffer: quad_elements,
        },
      ],
    };
    let node = { geometry };

    vbo_quad = renderer.getGpuBuffers(node.geometry);
  }

  return vbo_quad;
}

function createPipeline(renderer) {
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
      // Add other entries if needed
    ],
  });
  const layout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createRenderPipeline({
    layout,
    vertexStage: {
      module: device.createShaderModule({ code: vs }),
      entryPoint: "renderQuad",
    },
    fragmentStage: {
      module: device.createShaderModule({ code: fs }),
      entryPoint: "renderQuad",
    },
    primitiveTopology: "triangle-list",
    depthStencilState: {
      depthWriteEnabled: true,
      depthCompare: "greater",
      format: "depth32float",
    },
    vertexState: {
      vertexBuffers: [
        {
          // point position
          arrayStride: 3 * 4,
          stepMode: "instance",
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x3",
            },
          ],
        },
        {
          // quad position
          arrayStride: 3 * 4,
          stepMode: "vertex",
          attributes: [
            {
              shaderLocation: 1,
              offset: 0,
              format: "float32x3",
            },
          ],
        },
        {
          // color
          arrayStride: 4,
          stepMode: "instance",
          attributes: [
            {
              shaderLocation: 2,
              offset: 0,
              format: "unorm8x4",
            },
          ],
        },
      ],
    },
    rasterizationState: {
      cullMode: "none",
    },
    colorStates: [
      {
        format: "bgra8unorm",
      },
    ],
  });

  return pipeline;
}

function getOctreeState(renderer, node) {
  let { device } = renderer;

  let state = octreeStates.get(node);

  if (!state) {
    let pipeline = createPipeline(renderer);

    const uniformBufferSize = 2 * 4 * 16 + 8 + 4;

    const uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
      ],
    });

    state = {
      pipeline: pipeline,
      uniformBuffer: uniformBuffer,
      uniformBindGroup: uniformBindGroup,
    };

    octreeStates.set(node, state);
  }

  return state;
}

function getNodeState(renderer, node) {
  let state = nodeStates.get(node);

  if (!state) {
    let vbos = renderer.getGpuBuffers(node.geometry);

    state = { vbos };
    nodeStates.set(node, state);
  }

  return state;
}

export function render(renderer, pass, octree, camera) {
  let { device } = renderer;

  let vbo_quad = getVboQuad(renderer);

  let octreeState = getOctreeState(renderer, octree);

  {
    // update uniforms
    let { uniformBuffer } = octreeState;

    {
      // transform
      let world = octree.world;
      let view = camera.view;
      let worldView = new Matrix4().multiplyMatrices(view, world);

      let tmp = new Float32Array(16);

      tmp.set(worldView.elements);
      device.queue.writeBuffer(
        uniformBuffer,
        0,
        tmp.buffer,
        tmp.byteOffset,
        tmp.byteLength
      );

      tmp.set(camera.proj.elements);
      device.queue.writeBuffer(
        uniformBuffer,
        64,
        tmp.buffer,
        tmp.byteOffset,
        tmp.byteLength
      );
    }

    {
      // screen size, point size, ...
      let size = renderer.getSize();
      let data = new Float32Array([size.width, size.height, octree.pointSize]);
      device.queue.writeBuffer(
        uniformBuffer,
        128,
        data.buffer,
        data.byteOffset,
        data.byteLength
      );
    }
  }

  let { passEncoder } = pass;
  let { pipeline, uniformBindGroup } = octreeState;

  Timer.timestamp(passEncoder, "quads-start");

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, uniformBindGroup);

  let nodes = octree.visibleNodes;

  for (let node of nodes) {
    let nodeState = getNodeState(renderer, node);

    passEncoder.setVertexBuffer(0, nodeState.vbos[0].vbo);
    passEncoder.setVertexBuffer(1, vbo_quad[0].vbo);
    passEncoder.setVertexBuffer(2, nodeState.vbos[1].vbo);
    passEncoder.setIndexBuffer(vbo_quad[1].vbo, "uint32");

    if (octree.showBoundingBox === true) {
      let position = node.boundingBox.min.clone();
      position.add(node.boundingBox.max).multiplyScalar(0.5);
      // position.applyMatrix4(octree.world);
      let size = node.boundingBox.size();
      let color = new Vector3(...SPECTRAL.get(node.level / 5));
      renderer.drawBoundingBox(position, size, color);
    }

    let numElements = node.geometry.numElements;
    passEncoder.drawIndexed(6, numElements, 0, 0);
  }

  Timer.timestamp(passEncoder, "quads-end");
}
