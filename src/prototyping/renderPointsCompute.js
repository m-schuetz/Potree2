import glslangModule from "glslang";
import { Matrix4, Timer } from "potree";

let glslang = undefined;

let csDepth = `

#version 450

layout(local_size_x = 32, local_size_y = 1) in;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldView;
	mat4 proj;
	uint width;
	uint height;
} uniforms;


layout(std430, set = 0, binding = 1) buffer SSBO {
	uint framebuffer[];
};

layout(std430, set = 0, binding = 2) buffer SSBO_position {
	float positions[];
};

layout(std430, set = 0, binding = 3) buffer SSBO_color {
	uint colors[];
};

// [[binding(4), set(0)]] var myTexture: texture_2d<f32>;
// layout(binding = 4) writeonly  uniform image2D destTex;
// uniform layout(set = 0, binding = 4, rgba8ui) writeonly  uimage2D uFractalTexture;

shared uint sX;
shared uint sY;
shared uint sDiverging;

void renderPointsCompute(){

	uint index = gl_GlobalInvocationID.x;

	vec4 pos_point = vec4(
		positions[3 * index + 0],
		positions[3 * index + 1],
		positions[3 * index + 2],
		1.0);

	vec4 viewPos = uniforms.worldView * pos_point;
	vec4 pos = uniforms.proj * viewPos;

	pos.xyz = pos.xyz / pos.w;

	bool isClipped = false;
	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		isClipped = true;
	}

	ivec2 imageSize = ivec2(
		int(uniforms.width),
		int(uniforms.height)
	);

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * imageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * imageSize.x;

	uint depth = floatBitsToUint(pos.w);
	uint old = framebuffer[pixelID];

	if(depth < old && !isClipped){
		atomicMin(framebuffer[pixelID], depth);
	}

	// if(gl_LocalInvocationID.x == 0){
	// 	sX = pixelCoords.x;
	// 	sY = pixelCoords.y;
	// 	sDiverging = 0;
	// }

	// barrier();

	// if(pixelCoords.x != sX || pixelCoords.y != sY){
	// 	atomicAdd(sDiverging, 1);
	// }

	// barrier();

	// bool inDifferentPixels = sDiverging > 0;
	// bool inSamePixel = !inDifferentPixels;

	// if(isClipped){

	// }else if(inDifferentPixels){
	// 	uint depth = floatBitsToUint(pos.w);
	// 	uint old = framebuffer[pixelID];

	// 	if(depth < old){
	// 		atomicMin(framebuffer[pixelID], depth);
	// 	}
	// }else if(inSamePixel && gl_LocalInvocationID.x == 0){
	// 	uint depth = floatBitsToUint(pos.w);
	// 	uint old = framebuffer[pixelID];

	// 	if(depth < old){
	// 		atomicMin(framebuffer[pixelID], depth);
	// 	}
	// }

	
}

`;

// let csDepth = `

// struct Uniforms {
// 	worldView : mat4x4<f32>;
// 	proj : mat4x4<f32>;
// 	width : u32;
// 	height: u32;
// };

// struct U32s{
// 	values : [[stride(4)]] array<u32>;
// };

// struct F32s{
// 	values : [[stride(4)]] array<f32>;
// };

// @binding(0), group(0) var<uniform> uniforms : Uniforms,
// @binding(1), group(0) var<storage> framebuffer : [[access(read_write)]] U32s,
// @binding(2), group(0) var<storage> positions : [[access(read)]] F32s,
// @binding(3), group(0) var<storage> colors : [[access(read)]] U32s,

// [[stage(compute), workgroup_size(128)]]
// fn renderPointsCompute([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

// 	var index : u32 = GlobalInvocationID.x;

// 	var pos_point : vec4<f32> = vec4<f32>(
// 		positions.values[3u * index + 0u],
// 		positions.values[3u * index + 1u],
// 		positions.values[3u * index + 2u],
// 		1.0
// 	);

// 	var viewPos : vec4<f32> = uniforms.worldView * pos_point;
// 	var pos : vec4<f32> = uniforms.proj * viewPos;

// 	pos.x = pos.x / pos.w;
// 	pos.y = pos.y / pos.w;
// 	pos.z = pos.z / pos.w;

// 	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
// 		return;
// 	}

// 	var imgPos : vec2<f32> = vec2<f32>(
// 		(pos.x * 0.5 + 0.5) * f32(uniforms.width),
// 		(pos.y * 0.5 + 0.5) * f32(uniforms.height)
// 	);

// 	var pixelCoords : vec2<u32> = vec2<u32>(imgPos);

// 	var pixelID : u32 = pixelCoords.x + pixelCoords.y * uniforms.width;

// 	var depth : u32 = bitcast<u32>(pos.w);

// 	var oldDepth : u32 = framebuffer.values[pixelID];

// 	if(depth < oldDepth){
// 		framebuffer.values[pixelID] = depth;
// 	}

// 	// framebuffer.values[pixelID] = 0x42c80000u;
// }

// `;

let csColor = `

#version 450

layout(local_size_x = 128, local_size_y = 1) in;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldView;
	mat4 proj;
	uint width;
	uint height;
} uniforms;


layout(std430, set = 0, binding = 1) buffer SSBO_COLORS {
	uint ssbo_colors[];
};

layout(std430, set = 0, binding = 2) buffer SSBO_DEPTH {
	uint ssbo_depth[];
};

layout(std430, set = 0, binding = 3) buffer SSBO_position {
	float positions[];
};

layout(std430, set = 0, binding = 4) buffer SSBO_color {
	uint colors[];
};



void renderPointsCompute(){

	uint index = gl_GlobalInvocationID.x;

	vec4 pos_point = vec4(
		positions[3 * index + 0],
		positions[3 * index + 1],
		positions[3 * index + 2],
		1.0);

	vec4 viewPos = uniforms.worldView * pos_point;
	vec4 pos = uniforms.proj * viewPos;

	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	ivec2 imageSize = ivec2(
		int(uniforms.width),
		int(uniforms.height)
	);

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * imageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * imageSize.x;

	uint color = colors[index];

	uint r = (color >> 0) & 0xFFu;
	uint g = (color >> 8) & 0xFFu;
	uint b = (color >> 16) & 0xFFu;

	float depth = pos.w;
	float bufferedDepth = uintBitsToFloat(ssbo_depth[pixelID]);

	// just sum up points with the same depth
	if(depth <= bufferedDepth){
		atomicAdd(ssbo_colors[4 * pixelID + 0], r);
		atomicAdd(ssbo_colors[4 * pixelID + 1], g);
		atomicAdd(ssbo_colors[4 * pixelID + 2], b);
		atomicAdd(ssbo_colors[4 * pixelID + 3], 1);
	}
	

	// // or within a depth range (1.01 -> 1%)
	// if(depth <= bufferedDepth * 1.001){
	// 	atomicAdd(ssbo_colors[4 * pixelID + 0], r);
	// 	atomicAdd(ssbo_colors[4 * pixelID + 1], g);
	// 	atomicAdd(ssbo_colors[4 * pixelID + 2], b);
	// 	atomicAdd(ssbo_colors[4 * pixelID + 3], 1);
	// }

	// directly write points with same depth
	// will likely cause flickering
	// if(depth == bufferedDepth){
	// 	ssbo_colors[4 * pixelID + 0] = r;
	// 	ssbo_colors[4 * pixelID + 1] = g;
	// 	ssbo_colors[4 * pixelID + 2] = b;
	// 	ssbo_colors[4 * pixelID + 3] = 1u;
	// }
}

`;

let csReset = `

#version 450

layout(local_size_x = 128, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer SSBO {
	uint framebuffer[];
};

layout(set = 0, binding = 1) uniform Uniforms {
	uint value;
} uniforms;

void renderPointsCompute(){

	uint index = gl_GlobalInvocationID.x;

	framebuffer[index] = uniforms.value;
}
`;

let vs = `

	struct Uniforms {
		@size(4) uTest : u32,
		@size(4) x : f32,
		@size(4) y : f32,
		@size(4) width : f32,
		@size(4) height : f32,
	};
	@group(0) @binding(0) var<uniform> uniforms : Uniforms;

	struct VertexInput {
		@builtin(vertex_idx) index : u32,
	};

	struct VertexOutput {
		@builtin(position) position : vec4<f32>,
	};

	@vertex
	fn renderPointsCompute(vertex : VertexInput) -> VertexOutput {

		var output : VertexOutput;

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		var vi : i32 = vertex.index;

		if(vi == 0 || vi == 3 || vi == 5){
			output.position.x = x;
		}else{
			output.position.x = x + width;
		}

		if(vi == 0 || vi == 1 || vi == 3){
			output.position.y = y;
		}else{
			output.position.y = y + height;
		}

		output.position.z = 0.01;
		output.position.w = 1.0;

		return output;
	}
`;

let fs = `

	struct Colors {
		values : [[stride(4)]] array<u32>;
	};

	struct Uniforms {
		uTest : u32;
		x : f32;
		y : f32;
		width : f32;
		height : f32;
		screenWidth : f32;
		screenHeight : f32;
	};
	@group(0) @binding(0) var<uniform> uniforms : Uniforms;

	@binding(1), set(0) var<storage_buffer> ssbo_colors : [[access(read)]]Colors,

	@location(0) var<out> outColor : vec4<f32>;

	@builtin(frag_coord) var<in> fragCoord : vec4<f32>,

	@fragment
	fn renderPointsCompute() -> void {

		var frag_x : i32 = i32(fragCoord.x);
		var frag_y : i32 = i32(uniforms.screenHeight) - i32(fragCoord.y);
		var width : i32 = i32(uniforms.screenWidth);
		var index : u32 = u32(frag_x + frag_y * width);

		var c : u32 = ssbo_colors.values[4u * index + 3u];

		if(c == 0u){
			discard;
		}else{
			var r : u32 = ssbo_colors.values[4u * index + 0u] / c;
			var g : u32 = ssbo_colors.values[4u * index + 1u] / c;
			var b : u32 = ssbo_colors.values[4u * index + 2u] / c;

			outColor.r = f32(r) / 256.0;
			outColor.g = f32(g) / 256.0;
			outColor.b = f32(b) / 256.0;
			outColor.a = 1.0;
		}

	}
`;

let depthState = null;
let colorState = null;
let resetState = null;
let screenPassState = null;

function getDepthState(renderer) {
  if (!depthState) {
    let { device } = renderer;

    let ssboSize = 2560 * 1440 * 4 * 4;
    let ssbo = renderer.createBuffer(ssboSize);

    // let csDescriptor = {
    // 	code: csDepth,
    // };
    let csDescriptor = {
      code: glslang.compileGLSL(csDepth, "compute"),
      source: csDepth,
    };
    let csModule = device.createShaderModule(csDescriptor);

    let uniformBufferSize = 2 * 64 + 8;
    let uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let pipeline = device.createComputePipeline({
      compute: {
        module: csModule,
        entryPoint: "renderPointsCompute",
      },
    });

    depthState = { pipeline, ssbo, ssboSize, uniformBuffer };
  }

  return depthState;
}

function getColorState(renderer) {
  if (!colorState) {
    let { device } = renderer;

    let ssboSize = 4 * 2560 * 1440 * 4 * 4;
    let ssbo_colors = renderer.createBuffer(ssboSize);

    let csDescriptor = {
      code: glslang.compileGLSL(csColor, "compute"),
      source: csColor,
    };
    let csModule = device.createShaderModule(csDescriptor);

    let uniformBufferSize = 2 * 64 + 8;
    let uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let pipeline = device.createComputePipeline({
      compute: {
        module: csModule,
        entryPoint: "renderPointsCompute",
      },
    });

    colorState = { pipeline, ssbo_colors, ssboSize, uniformBuffer };
  }

  return colorState;
}

function getResetState(renderer) {
  if (!resetState) {
    let { device } = renderer;

    let uniformBufferSize = 4;
    let uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let csDescriptor = {
      code: glslang.compileGLSL(csReset, "compute"),
      source: csReset,
    };
    let csModule = device.createShaderModule(csDescriptor);

    let pipeline = device.createComputePipeline({
      compute: {
        module: csModule,
        entryPoint: "renderPointsCompute",
      },
    });

    resetState = { pipeline, uniformBuffer };
  }

  return resetState;
}

function getScreenPassState(renderer) {
  if (!screenPassState) {
    let { device, swapChainFormat } = renderer;

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

    let pipeline = device.createRenderPipeline({
      layout,
      vertex: {
        module: device.createShaderModule({ code: vs }),
        entryPoint: "renderPointsCompute",
      },
      fragment: {
        module: device.createShaderModule({ code: fs }),
        entryPoint: "renderPointsCompute",
        targets: [{ format: "bgra8unorm" }],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "greater",
        format: "depth32float",
      },
    });

    let uniformBufferSize = 32;
    let uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    screenPassState = { pipeline, uniformBuffer };
  }

  return screenPassState;
}

let frame = 0;

function reset(renderer, ssbo, numUints, value) {
  let { device } = renderer;
  let { pipeline, uniformBuffer } = getResetState(renderer);

  let bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ssbo } },
      { binding: 1, resource: { buffer: uniformBuffer } },
    ],
  });

  {
    // uniform buffer
    let data = new Uint32Array([value]);
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      data.buffer,
      data.byteOffset,
      data.byteLength
    );
  }

  const commandEncoder = device.createCommandEncoder();

  let passEncoder = commandEncoder.beginComputePass();

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);

  let groups = numUints / 128;
  passEncoder.dispatch(groups, 1, 1);
  passEncoder.endPass();

  device.queue.submit([commandEncoder.finish()]);
}

function depthPass(renderer, nodes, camera) {
  let { device } = renderer;

  const commandEncoder = device.createCommandEncoder();
  let passEncoder = commandEncoder.beginComputePass();

  Timer.timestamp(passEncoder, "depth-start");

  let { pipeline, uniformBuffer, ssbo, ssboSize } = getDepthState(renderer);
  passEncoder.setPipeline(pipeline);

  for (let node of nodes) {
    {
      // update uniforms DEPTH
      let { uniformBuffer } = getDepthState(renderer);

      {
        // transform
        let world = node.world;
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
        // screen size
        let size = renderer.getSize();
        let data = new Uint32Array([size.width, size.height]);
        device.queue.writeBuffer(
          uniformBuffer,
          128,
          data.buffer,
          data.byteOffset,
          data.byteLength
        );
      }
    }

    {
      let gpuBuffers = renderer.getGpuBuffers(node.geometry);

      let bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: ssbo } },
          { binding: 2, resource: { buffer: gpuBuffers[0].vbo } },
          { binding: 3, resource: { buffer: gpuBuffers[1].vbo } },
          // {binding: 4, resource: target.colorAttachments[0].texture.createView()},
        ],
      });

      passEncoder.setBindGroup(0, bindGroup);

      // let groups = [
      // 	Math.floor(node.geometry.numElements / 128),
      // 	1, 1
      // ];
      // passEncoder.dispatch(...groups);
      passEncoder.dispatch(Math.ceil(node.geometry.numElements / 32));

      // passEncoder.dispatch(node.geometry.numElements);
    }
  }

  Timer.timestamp(passEncoder, "depth-end");

  passEncoder.endPass();

  Timer.resolve(renderer, commandEncoder);

  device.queue.submit([commandEncoder.finish()]);
}

function colorPass(renderer, nodes, camera) {
  let { device } = renderer;

  let { pipeline, uniformBuffer } = getColorState(renderer);
  let { ssbo_colors } = getColorState(renderer);
  let ssbo_depth = getDepthState(renderer).ssbo;

  const commandEncoder = device.createCommandEncoder();
  let passEncoder = commandEncoder.beginComputePass();

  Timer.timestamp(passEncoder, "color-start");

  passEncoder.setPipeline(pipeline);

  for (let node of nodes) {
    {
      // update uniforms COLOR
      let { uniformBuffer } = getColorState(renderer);

      {
        // transform
        let world = node.world;
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
        // screen size
        let size = renderer.getSize();
        let data = new Uint32Array([size.width, size.height]);
        device.queue.writeBuffer(
          uniformBuffer,
          128,
          data.buffer,
          data.byteOffset,
          data.byteLength
        );
      }
    }

    {
      let gpuBuffers = renderer.getGpuBuffers(node.geometry);

      let bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: ssbo_colors } },
          { binding: 2, resource: { buffer: ssbo_depth } },
          { binding: 3, resource: { buffer: gpuBuffers[0].vbo } },
          { binding: 4, resource: { buffer: gpuBuffers[1].vbo } },
        ],
      });

      passEncoder.setBindGroup(0, bindGroup);

      let groups = [Math.ceil(node.geometry.numElements / 128), 1, 1];
      passEncoder.dispatch(...groups);
    }
  }

  Timer.timestamp(passEncoder, "color-end");

  passEncoder.endPass();

  Timer.resolve(renderer, commandEncoder);

  device.queue.submit([commandEncoder.finish()]);
}

export function render(nodes, drawstate) {
  if (glslang === undefined) {
    glslang = null;

    glslangModule().then((result) => {
      glslang = result;
    });

    return;
  } else if (glslang === null) {
    return;
  }

  let { renderer, camera } = drawstate;
  let { device } = renderer;

  Timer.timestampSep(renderer, "compute-start");

  // init(renderer);

  {
    // RESET BUFFERS
    let size = renderer.getSize();
    let numUints = size.width * size.height;
    let { ssbo } = getDepthState(renderer);
    let { ssbo_colors } = getColorState(renderer);

    reset(renderer, ssbo, numUints, 0x7fffffff);
    reset(renderer, ssbo_colors, 4 * numUints, 0);
  }

  depthPass(renderer, nodes, camera);
  colorPass(renderer, nodes, camera);

  {
    // resolve
    let { pass } = drawstate;
    let { passEncoder } = pass;
    let { ssbo_colors } = getColorState(renderer);
    let { pipeline, uniformBuffer } = getScreenPassState(renderer);
    let size = renderer.getSize();

    let uniformBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: ssbo_colors } },
      ],
    });

    passEncoder.setPipeline(pipeline);

    {
      let source = new ArrayBuffer(32);
      let view = new DataView(source);

      let x = 0;
      let y = 0;
      let width = 1;
      let height = 1;
      let screenWidth = size.width;
      let screenHeight = size.height;

      view.setUint32(0, 5, true);
      view.setFloat32(4, x, true);
      view.setFloat32(8, y, true);
      view.setFloat32(12, width, true);
      view.setFloat32(16, height, true);
      view.setFloat32(20, screenWidth, true);
      view.setFloat32(24, screenHeight, true);

      device.queue.writeBuffer(uniformBuffer, 0, source, 0, source.byteLength);

      passEncoder.setBindGroup(0, uniformBindGroup);
    }

    passEncoder.draw(6, 1, 0, 0);
  }

  Timer.timestampSep(renderer, "compute-end");
}
