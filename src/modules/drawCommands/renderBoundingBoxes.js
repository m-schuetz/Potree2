import { Geometry } from "../../core/Geometry.js";
import { Matrix4 } from "../../math/math.js";
import { cube_wireframe_thick } from "../geometries/cube.js";

const vs = `
struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>,
	[[offset(64)]] proj : mat4x4<f32>,
	[[offset(128)]] screen_width : f32,
	[[offset(132)]] screen_height : f32,
};



@location(0) var<in> box_pos : vec3<f32>;
@location(1) var<in> box_scale : vec3<f32>;
@location(2) var<in> point_pos : vec3<f32>;
@location(3) var<in> box_color : vec4<f32>;
@location(4) var<in> in_dir : vec3<f32>;

@builtin(vertex_index) var<in> vertexID : i32,

@builtin(position) var<out> out_pos : vec4<f32>;
@location(0) var<out> fragColor : vec4<f32>;

@vertex
fn renderBoundingBoxes() -> void {


	var worldPos : vec3<f32> = box_pos + point_pos * box_scale;
	var worldDir : vec3<f32> = worldPos + in_dir;

	var viewPos : vec3<f32> = (uniforms.worldView * vec4<f32>(worldPos, 1.0)).xyz;
	var viewTar : vec3<f32> = (uniforms.worldView * vec4<f32>(worldDir, 1.0)).xyz;
	var viewDir : vec3<f32> = (uniforms.worldView * vec4<f32>(worldDir, 1.0)).xyz - viewPos;
	viewDir = normalize(viewDir);

	var projPos : vec4<f32> = uniforms.proj * vec4<f32>(viewPos, 1.0);
	var projTar : vec4<f32> = uniforms.proj * vec4<f32>(viewTar, 1.0);

	projPos.x = uniforms.screen_width * (1.0 + projPos.x / projPos.w) / 2.0;
	projPos.y = uniforms.screen_height * (1.0 + projPos.y / projPos.w) / 2.0;

	projTar.x = uniforms.screen_width * (1.0 + projTar.x / projTar.w) / 2.0;
	projTar.y = uniforms.screen_height * (1.0 + projTar.y / projTar.w) / 2.0;

	var projDiff : vec3<f32> = normalize((projTar - projPos).xyz);

	var screenDir : vec2<f32> = vec2<f32>(
		projTar.x - projPos.x,
		projTar.y - projPos.y
	);
	screenDir = 5.0 * normalize(screenDir);

	var lineWidth : f32 = 2.0;
	var lineWidthH : f32 = lineWidth / 2.0;

	projPos.x = projPos.x - lineWidthH * projDiff.y;
	projPos.y = projPos.y + lineWidthH * projDiff.x;

	projPos.x = ((projPos.x / uniforms.screen_width) * 2.0 - 1.0) * projPos.w;
	projPos.y = ((projPos.y / uniforms.screen_height) * 2.0 - 1.0) * projPos.w;

	out_pos = projPos;

	fragColor = box_color;

	return;
}
`;

const fs = `
@location(0) var<in> fragColor : vec4<f32>;
@location(0) var<out> outColor : vec4<f32>;

@fragment
fn renderBoundingBoxes() -> void {
	outColor = fragColor;
	return;
}
`;

let initialized = false;
let pipeline = null;
let geometry_boxes = null;
let uniformBuffer = null;
let bindGroup = null;
let capacity = 10_000;

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
  pipeline = device.createRenderPipeline({
    layout,
    vertex: {
      module: device.createShaderModule({ code: vs }),
      entryPoint: "renderBoundingBoxes",
      buffers: [
        {
          // box position
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
          // box scale
          arrayStride: 3 * 4,
          stepMode: "instance",
          attributes: [
            {
              shaderLocation: 1,
              offset: 0,
              format: "float32x3",
            },
          ],
        },
        {
          // box-vertices position
          arrayStride: 3 * 4,
          stepMode: "vertex",
          attributes: [
            {
              shaderLocation: 2,
              offset: 0,
              format: "float32x3",
            },
          ],
        },
        {
          // box color
          arrayStride: 4,
          stepMode: "instance",
          attributes: [
            {
              shaderLocation: 3,
              offset: 0,
              format: "unorm8x4",
            },
          ],
        },
        {
          // box line directions
          arrayStride: 3 * 4,
          stepMode: "vertex",
          attributes: [
            {
              shaderLocation: 4,
              offset: 0,
              format: "float32x3",
            },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: fs }),
      entryPoint: "renderBoundingBoxes",
      targets: [{ format: "bgra8unorm" }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "greater",
      format: "depth32float",
    },
  });

  return pipeline;
}

function init(renderer) {
  if (initialized) {
    return;
  }

  geometry_boxes = new Geometry({
    buffers: [
      {
        name: "position",
        buffer: new Float32Array(3 * capacity),
      },
      {
        name: "scale",
        buffer: new Float32Array(3 * capacity),
      },
      {
        name: "color",
        buffer: new Uint8Array(4 * capacity),
      },
    ],
  });

  {
    pipeline = createPipeline(renderer);

    let { device } = renderer;
    const uniformBufferSize = 256;

    uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });
  }

  initialized = true;
}

function updateUniforms(renderer) {
  let data = new ArrayBuffer(256);
  let f32 = new Float32Array(data);
  let view = new DataView(data);

  {
    // transform
    let world = new Matrix4();
    let view = camera.view;
    let worldView = new Matrix4().multiplyMatrices(view, world);

    f32.set(worldView.elements, 0);
    f32.set(camera.proj.elements, 16);
  }

  {
    // misc
    let size = renderer.getSize();

    view.setFloat32(128, size.width, true);
    view.setFloat32(132, size.height, true);
  }

  renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}

export function render(renderer, pass, boxes, camera) {
  let { device } = renderer;

  init(renderer);

  updateUniforms(renderer);

  let { passEncoder } = pass;

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);

  let position = geometry_boxes.buffers.find(
    (g) => g.name === "position"
  ).buffer;
  let scale = geometry_boxes.buffers.find((g) => g.name === "scale").buffer;
  let color = geometry_boxes.buffers.find((g) => g.name === "color").buffer;
  let vboPosition = renderer.getGpuBuffer(position);
  let vboScale = renderer.getGpuBuffer(scale);
  let vboColor = renderer.getGpuBuffer(color);
  {
    for (let i = 0; i < boxes.length; i++) {
      let box = boxes[i];
      let pos = box[0];

      position[3 * i + 0] = pos.x;
      position[3 * i + 1] = pos.y;
      position[3 * i + 2] = pos.z;

      scale[3 * i + 0] = box[1].x;
      scale[3 * i + 1] = box[1].y;
      scale[3 * i + 2] = box[1].z;

      color[4 * i + 0] = box[2].x;
      color[4 * i + 1] = box[2].y;
      color[4 * i + 2] = box[2].z;
      color[4 * i + 3] = 255;
    }

    device.queue.writeBuffer(
      vboPosition,
      0,
      position.buffer,
      0,
      position.byteLength
    );
    device.queue.writeBuffer(vboScale, 0, scale.buffer, 0, scale.byteLength);
    device.queue.writeBuffer(vboColor, 0, color.buffer, 0, color.byteLength);
  }

  {
    // wireframe
    let boxVertices = cube_wireframe_thick.buffers.find(
      (b) => b.name === "position"
    ).buffer;
    let boxLineDirections = cube_wireframe_thick.buffers.find(
      (b) => b.name === "direction"
    ).buffer;
    let vboBoxVertices = renderer.getGpuBuffer(boxVertices);
    let vboBoxLineDirections = renderer.getGpuBuffer(boxLineDirections);

    passEncoder.setVertexBuffer(0, vboPosition);
    passEncoder.setVertexBuffer(1, vboScale);
    passEncoder.setVertexBuffer(2, vboBoxVertices);
    passEncoder.setVertexBuffer(3, vboColor);
    passEncoder.setVertexBuffer(4, vboBoxLineDirections);
    // passEncoder.setIndexBuffer(vboBoxIndices, "uint32");

    let numBoxes = boxes.length;
    let numVertices = cube_wireframe_thick.numElements;
    passEncoder.draw(numVertices, numBoxes, 0, 0);
  }

  // { // solid
  // 	let boxVertices = cube.buffers.find(b => b.name === "position").buffer;
  // 	let vboBoxVertices = renderer.getGpuBuffer(boxVertices);

  // 	passEncoder.setVertexBuffer(0, vboPosition);
  // 	passEncoder.setVertexBuffer(1, vboScale);
  // 	passEncoder.setVertexBuffer(2, vboBoxVertices);
  // 	passEncoder.setVertexBuffer(3, vboColor);
  // 	// passEncoder.setIndexBuffer(vboBoxIndices, "uint32");

  // 	let numBoxes = boxes.length;
  // 	let numVertices = cube.numElements;
  // 	passEncoder.draw(numVertices, numBoxes, 0, 0);
  // }
}
