
import {Vector3, Matrix4} from "../math/math.js";
import {SPECTRAL} from "../misc/Gradients.js";
import {RenderTarget} from "../core/RenderTarget.js";

import glslangModule from "../../libs/glslang/glslang.js";


let glslang = null;
glslangModule().then( result => {
	glslang = result;
});

let csDepth = `

#version 450

layout(local_size_x = 128, local_size_y = 1) in;

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



void main(){

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

	//uint r = (color >> 0) & 0xFFu;
	//uint g = (color >> 8) & 0xFFu;
	//uint b = (color >> 16) & 0xFFu;
	//uint a = 255u;
	//uint c = (r << 24) | (g << 16) | (b << 8) | a;

	//framebuffer[pixelID] = c;

	uint depth = uint(-viewPos.z * 1000.0);

	atomicMin(framebuffer[pixelID], depth);
}

`;

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



void main(){

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

	uint depth = uint(-viewPos.z * 1000.0);
	uint bufferedDepth = ssbo_depth[pixelID];

	if(depth < 1.01 * bufferedDepth){
		atomicAdd(ssbo_colors[4 * pixelID + 0], r);
		atomicAdd(ssbo_colors[4 * pixelID + 1], g);
		atomicAdd(ssbo_colors[4 * pixelID + 2], b);
		atomicAdd(ssbo_colors[4 * pixelID + 3], 1);
	}
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

void main(){

	uint index = gl_GlobalInvocationID.x;

	framebuffer[index] = uniforms.value;
}
`;



let vs = `
	const pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.1)
	);

	const uv : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 1.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(0.0, 0.0)
	);

	[[builtin(position)]] var<out> Position : vec4<f32>;
	[[builtin(vertex_idx)]] var<in> VertexIndex : i32;

	[[block]] struct Uniforms {
		[[offset(0)]] uTest : u32;
		[[offset(4)]] x : f32;
		[[offset(8)]] y : f32;
		[[offset(12)]] width : f32;
		[[offset(16)]] height : f32;
	};
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	[[location(0)]] var<out> fragUV : vec2<f32>;

	[[stage(vertex)]]
	fn main() -> void {
		Position = vec4<f32>(pos[VertexIndex], 0.999, 1.0);
		fragUV = uv[VertexIndex];

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		if(VertexIndex == 0){
			Position.x = x;
			Position.y = y;
		}elseif(VertexIndex == 1){
			Position.x = x + width;
			Position.y = y;
		}elseif(VertexIndex == 2){
			Position.x = x + width;
			Position.y = y + height;
		}elseif(VertexIndex == 3){
			Position.x = x;
			Position.y = y;
		}elseif(VertexIndex == 4){
			Position.x = x + width;
			Position.y = y + height;
		}elseif(VertexIndex == 5){
			Position.x = x;
			Position.y = y + height;
		}

		return;
	}
`;

let fs = `

	[[block]] struct Colors {
		[[offset(0)]] values : [[stride(4)]] array<u32>;
	};

	[[block]] struct Uniforms {
		[[offset(0)]] uTest : u32;
		[[offset(4)]] x : f32;
		[[offset(8)]] y : f32;
		[[offset(12)]] width : f32;
		[[offset(16)]] height : f32;
		[[offset(20)]] screenWidth : f32;
		[[offset(24)]] screenHeight : f32;
	};
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	[[binding(1), set(0)]] var<storage_buffer> ssbo_colors : Colors;

	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[builtin(frag_coord)]] var<in> fragCoord : vec4<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		var index : u32 = 
			u32(fragCoord.x) +
			(u32(uniforms.screenHeight) - u32(fragCoord.y) - 1) * u32(uniforms.screenWidth);
		
		var c : u32 = ssbo_colors.values[4 * index + 3];
		var r : u32 = ssbo_colors.values[4 * index + 0] / c;
		var g : u32 = ssbo_colors.values[4 * index + 1] / c;
		var b : u32 = ssbo_colors.values[4 * index + 2] / c;

		outColor.r = f32(r) / 256.0;
		outColor.g = f32(g) / 256.0;
		outColor.b = f32(b) / 256.0;
		outColor.a = 1.0;

		#outColor.r = f32(r) / 10.0;
		#outColor.g = f32(g) / 10.0;
		#outColor.b = f32(b) / 10.0;

	}
`;




let _target_1 = null;

function getTarget1(renderer){
	if(_target_1 === null){

		let size = [128, 128, 1];
		_target_1 = new RenderTarget(renderer, {
			size: size,
			colorDescriptors: [{
				size: size,
				format: renderer.swapChainFormat,
				usage: GPUTextureUsage.SAMPLED 
					// | GPUTextureUsage.COPY_SRC 
					// | GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}],
			depthDescriptor: {
				size: size,
				format: "depth24plus-stencil8",
				usage: GPUTextureUsage.SAMPLED 
					// | GPUTextureUsage.COPY_SRC 
					// | GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}
		});
	}

	return _target_1;
}


let depthState = null;
let colorState = null;
let resetState = null;
let screenPassState = null;

function getDepthState(renderer){

	if(!depthState){

		let {device} = renderer;

		// let target = getTarget1(renderer);
		let ssboSize = 2560 * 1440 * 4 * 4;
		let ssbo = renderer.createBuffer(ssboSize);

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
			computeStage: {
				module: csModule,
				entryPoint: "main",
			}
		});

		depthState = {pipeline, ssbo, ssboSize, uniformBuffer};
	}

	return depthState;
}

function getColorState(renderer){

	if(!colorState){

		let {device} = renderer;

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
			computeStage: {
				module: csModule,
				entryPoint: "main",
			}
		});

		colorState = {pipeline, ssbo_colors, ssboSize, uniformBuffer};
	}

	return colorState;
}

function getResetState(renderer){

	if(!resetState){

		let {device} = renderer;

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
			computeStage: {
				module: csModule,
				entryPoint: "main",
			}
		});

		resetState = {pipeline, uniformBuffer};
	}

	return resetState;
}

function getScreenPassState(renderer){

	if(!screenPassState){
		let {device, swapChainFormat} = renderer;

		let bindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				type: "uniform-buffer"
			},{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				type: "storage-buffer"
			}]
		});

		let layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
		
		let pipeline = device.createRenderPipeline({
			layout: layout, 
			vertexStage: {
				module: device.createShaderModule({code: vs}),
				entryPoint: "main",
			},
			fragmentStage: {
				module: device.createShaderModule({code: fs}),
				entryPoint: "main",
			},
			primitiveTopology: "triangle-list",
			depthStencilState: {
					depthWriteEnabled: true,
					depthCompare: "less",
					format: "depth24plus-stencil8",
			},
			colorStates: [{
				format: swapChainFormat,
			}],
		});

		let uniformBufferSize = 32;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		screenPassState = {bindGroupLayout, pipeline, uniformBuffer}
	}

	return screenPassState;
}

let frame = 0;

function reset(renderer, ssbo, numUints, value){

	let {device} = renderer;
	let {pipeline, uniformBuffer} = getResetState(renderer);

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: ssbo,
					offset: 0,
					size: numUints * 4,
				}
			},{
				binding: 1,
				resource: {
					buffer: uniformBuffer,
				}
			}
		]
	});


	{ // uniform buffer
		let data = new Uint32Array([value]);
		device.defaultQueue.writeBuffer(
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

	device.defaultQueue.submit([commandEncoder.finish()]);
}

export function renderAtomic(renderer, octree, camera){

	if(!glslang){
		console.log("glslang not yet initialized");

		return;
	}

	let {device} = renderer;

	let nodes = octree.visibleNodes;

	if(nodes.length === 0){
		return getTarget1(renderer).colorAttachments[0].texture;
	}

	{
		let size = renderer.getSize();
		let target = getTarget1(renderer);
		target.setSize(size.width, size.height);
	}

	{ // update uniforms DEPTH
		let {uniformBuffer} = getDepthState(renderer);

		{ // transform
			let world = octree.world;
			let view = camera.view;
			let worldView = new Matrix4().multiplyMatrices(view, world);

			let tmp = new Float32Array(16);

			tmp.set(worldView.elements);
			device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				tmp.buffer, tmp.byteOffset, tmp.byteLength
			);

			tmp.set(camera.proj.elements);
			device.defaultQueue.writeBuffer(
				uniformBuffer, 64,
				tmp.buffer, tmp.byteOffset, tmp.byteLength
			);
		}

		{ // screen size
			let size = renderer.getSize();
			let data = new Uint32Array([size.width, size.height]);
			device.defaultQueue.writeBuffer(
				uniformBuffer,
				128,
				data.buffer,
				data.byteOffset,
				data.byteLength
			);
		}
	}

	{ // update uniforms COLOR
		let {uniformBuffer} = getColorState(renderer);

		{ // transform
			let world = octree.world;
			let view = camera.view;
			let worldView = new Matrix4().multiplyMatrices(view, world);

			let tmp = new Float32Array(16);

			tmp.set(worldView.elements);
			device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				tmp.buffer, tmp.byteOffset, tmp.byteLength
			);

			tmp.set(camera.proj.elements);
			device.defaultQueue.writeBuffer(
				uniformBuffer, 64,
				tmp.buffer, tmp.byteOffset, tmp.byteLength
			);
		}

		{ // screen size
			let size = renderer.getSize();
			let data = new Uint32Array([size.width, size.height]);
			device.defaultQueue.writeBuffer(
				uniformBuffer,
				128,
				data.buffer,
				data.byteOffset,
				data.byteLength
			);
		}
	}

	{ // RESET BUFFERS
		let size = renderer.getSize();
		let numUints = size.width * size.height;

		{
			let {ssbo} = getDepthState(renderer);

			reset(renderer, ssbo, numUints, 0xffffff);
		}

		{
			let {ssbo_colors} = getColorState(renderer);

			reset(renderer, ssbo_colors, 4 * numUints, 0);
		}
	}


	{ // DEPTH PASS
		let {pipeline, uniformBuffer, ssbo, ssboSize} = getDepthState(renderer);

		//let node = nodes[0];

		for(let node of nodes){
		let gpuBuffers = renderer.getGpuBuffers(node.geometry);

		let bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {
						buffer: uniformBuffer,
					}
				},{
					binding: 1,
					resource: {
						buffer: ssbo,
						offset: 0,
						size: ssboSize,
					}
				},{
					binding: 2,
					resource: {
						buffer: gpuBuffers[0].vbo,
						offset: 0,
						size: node.geometry.numElements * 12,
					}
				},{
					binding: 3,
					resource: {
						buffer: gpuBuffers[1].vbo,
						offset: 0,
						size: node.geometry.numElements * 4,
					}
				}
			]
		});

		const commandEncoder = device.createCommandEncoder();

		let passEncoder = commandEncoder.beginComputePass();

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);

		let groups = [
			Math.floor(node.geometry.numElements / 128),
			1, 1
		];
		passEncoder.dispatch(...groups);
		passEncoder.endPass();

		device.defaultQueue.submit([commandEncoder.finish()]);
		}

	}

	{ // COLOR PASS
		let {pipeline, uniformBuffer, ssboSize} = getColorState(renderer);
		let {ssbo_colors} = getColorState(renderer);
		let ssbo_depth = getDepthState(renderer).ssbo;


		for(let node of nodes){
			let gpuBuffers = renderer.getGpuBuffers(node.geometry);

			let bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{
						binding: 0,
						resource: {
							buffer: uniformBuffer,
						}
					},{
						binding: 1,
						resource: {
							buffer: ssbo_colors,
							offset: 0,
							size: ssboSize,
						}
					},{
						binding: 2,
						resource: {
							buffer: ssbo_depth,
							offset: 0,
							size: ssboSize / 4,
						}
					},{
						binding: 3,
						resource: {
							buffer: gpuBuffers[0].vbo,
							offset: 0,
							size: node.geometry.numElements * 12,
						}
					},{
						binding: 4,
						resource: {
							buffer: gpuBuffers[1].vbo,
							offset: 0,
							size: node.geometry.numElements * 4,
						}
					}
				]
			});

			const commandEncoder = device.createCommandEncoder();

			let passEncoder = commandEncoder.beginComputePass();

			passEncoder.setPipeline(pipeline);
			passEncoder.setBindGroup(0, bindGroup);

			let groups = [
				Math.floor(node.geometry.numElements / 128),
				1, 1
			];
			passEncoder.dispatch(...groups);
			passEncoder.endPass();

			device.defaultQueue.submit([commandEncoder.finish()]);
		}

	}

	{ // resolve
		let {ssbo} = getDepthState(renderer);
		let {ssbo_colors} = getColorState(renderer);
		let {pipeline, uniformBuffer} = getScreenPassState(renderer);
		let target = getTarget1(renderer);
		let size = renderer.getSize();

		let uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
					binding: 0,
					resource: {buffer: uniformBuffer}
				},{
					binding: 1,
					resource: {buffer: ssbo_colors},
				}],
		});



		let renderPassDescriptor = {
			colorAttachments: [{
				attachment: target.colorAttachments[0].texture.createView(),
				loadValue: { r: 0.4, g: 0.2, b: 0.3, a: 1.0 },
			}],
			depthStencilAttachment: {
				attachment: target.depth.texture.createView(),
				depthLoadValue: 1.0,
				depthStoreOp: "store",
				stencilLoadValue: 0,
				stencilStoreOp: "store",
			},
			sampleCount: 1,
		};

		const commandEncoder = renderer.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

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
			
			device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				source, 0, source.byteLength
			);

			passEncoder.setBindGroup(0, uniformBindGroup);
		}


		passEncoder.draw(6, 1, 0, 0);


		passEncoder.endPass();

		let commandBuffer = commandEncoder.finish();
		renderer.device.defaultQueue.submit([commandBuffer]);

	}

	frame++;

	return getTarget1(renderer).colorAttachments[0].texture;
}