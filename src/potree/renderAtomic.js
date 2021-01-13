
import {Vector3, Matrix4} from "../math/math.js";
import {SPECTRAL} from "../misc/Gradients.js";
import {RenderTarget} from "../core/RenderTarget.js";

import glslangModule from "../../libs/glslang/glslang.js";


let glslang = null;
glslangModule().then( result => {
	glslang = result;
});

let cs = `

#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform Uniforms {
	uint width;
	uint height;
} uniforms;


layout(std430, set = 0, binding = 1) buffer SSBO {
	uint ssbo[];
};



void main(){

	// uint globalID = gl_GlobalInvocationID.x;

	ivec2 pixelID = ivec2 (
		gl_GlobalInvocationID.x,
		gl_GlobalInvocationID.y);
	
	uint index = pixelID.y * uniforms.width + pixelID.x;

	// uint value = 255 * index / (uniforms.width * uniforms.height);
	uint value = index / uniforms.height;
	
	ssbo[index] = value;


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


	[[binding(1), set(0)]] var<storage_buffer> colors : Colors;

	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[builtin(frag_coord)]] var<in> fragCoord : vec4<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		var index : u32 = 256u * u32(fragCoord.y) + u32(fragCoord.x);

		var c : vec4<f32>;
		
		c.x = f32(colors.values[index]) / 255.0;
		c.y = f32(colors.values[index]) / 255.0;
		c.z = f32(colors.values[index]) / 255.0;

		c.w = 1.0;

		outColor = c;


		#outColor.x = 0.0;
		#outColor.y = 1.0;
		#outColor.z = 0.0;
		#outColor.w = 1.0;

		return;
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


let computeState = null;
let screenPassState = null;

function getComputeState(renderer){

	if(!computeState){

		let {device} = renderer;

		// let target = getTarget1(renderer);
		let ssboSize = 2560 * 1440 * 4 * 4;
		let ssbo = renderer.createBuffer(ssboSize);

		let csDescriptor = {
			code: glslang.compileGLSL(cs, "compute"),
			source: cs,
		};
		let csModule = device.createShaderModule(csDescriptor);

		let uniformBufferSize = 24;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let pipeline = device.createComputePipeline({
			// layout: layout, 
			computeStage: {
				module: csModule,
				entryPoint: "main",
			}
		});

		let bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
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
			}]
		});


		computeState = {pipeline, bindGroup, ssbo, uniformBuffer};
	}

	return computeState;
}

function getScreenPassState(renderer){

	if(!screenPassState){
		let {device, swapChainFormat} = renderer;

		let bindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
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

		let uniformBufferSize = 24;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		screenPassState = {bindGroupLayout, pipeline, uniformBuffer}
	}

	return screenPassState;
}

let frame = 0;

export function renderAtomic(renderer, octree, camera){

	if(!glslang){
		console.log("glslang not yet initialized");

		return;
	}

	let {device} = renderer;


	{ // COMPUTE SHADER
		let {pipeline, bindGroup, uniformBuffer} = getComputeState(renderer);
		let target = getTarget1(renderer);

		let width = target.size[0];
		let height = target.size[1];

		let source = new ArrayBuffer(24);
		let view = new DataView(source);

		view.setUint32(0, width, true);
		view.setUint32(4, height, true);
		
		device.defaultQueue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);

		const commandEncoder = device.createCommandEncoder();

		let passEncoder = commandEncoder.beginComputePass();

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);

		let groups = [
			target.size[0] / 8,
			target.size[1] / 8,
			1
		];
		passEncoder.dispatch(...groups);
		passEncoder.endPass();

		device.defaultQueue.submit([commandEncoder.finish()]);

		// renderer.readBuffer(ssbo, 0, 32).then(buffer => {
		// 	console.log(new Uint32Array(buffer));
		// });
	}

	{ // resolve
		let {ssbo} = getComputeState(renderer);
		let {pipeline, uniformBuffer} = getScreenPassState(renderer);

		let uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
					binding: 0,
					resource: {buffer: uniformBuffer}
				},{
					binding: 1,
					resource: {buffer: ssbo},
				}],
		});





		let size = renderer.getSize();
		let target = getTarget1(renderer);
		target.setSize(size.width, size.height);

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
			let source = new ArrayBuffer(24);
			let view = new DataView(source);

			let x = 0;
			let y = 0;
			let width = 1;
			let height = 1;

			view.setUint32(0, 5, true);
			view.setFloat32(4, x, true);
			view.setFloat32(8, y, true);
			view.setFloat32(12, width, true);
			view.setFloat32(16, height, true);
			
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