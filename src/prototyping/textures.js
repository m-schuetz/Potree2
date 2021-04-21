
import * as Timer from "../renderer/Timer.js";

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

	[[block]] struct Uniforms {
		[[size(4)]] uTest : u32;
		[[size(4)]] x : f32;
		[[size(4)]] y : f32;
		[[size(4)]] width : f32;
		[[size(4)]] height : f32;
	};
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	struct VertexInput {
		[[builtin(vertex_idx)]] index : i32;
	};

	struct VertexOutput {
		[[builtin(position)]] position : vec4<f32>;
		[[location(0)]] uv : vec2<f32>;
	};

	[[stage(vertex)]]
	fn main(vertex : VertexInput) -> VertexOutput {

		var output : VertexOutput;

		output.position = vec4<f32>(pos[vertex.index], 0.999, 1.0);
		output.uv = uv[vertex.index];

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

		return output;
	}
`;

let fs = `

	[[binding(1), set(0)]] var mySampler: sampler;
	[[binding(2), set(0)]] var myTexture: texture_2d<f32>;

	struct FragmentInput {
		[[location(0)]] uv: vec2<f32>;
	};

	[[stage(fragment)]]
	fn main(input : FragmentInput) -> [[location(0)]] vec4<f32> {

		var uv : vec2<f32> = input.uv;
		
		var outColor : vec4<f32> =  textureSample(myTexture, mySampler, uv);

		return outColor;
	}
`;


let pipeline = null;
let uniformBindGroup = null;
let uniformBuffer = null;

let state = new Map();

export async function loadImage(url){
	let img = document.createElement('img');
	
	img.src = url;
	await img.decode();

	let imageBitmap = await createImageBitmap(img);

	return imageBitmap;
}

function getGpuTexture(renderer, image){

	let gpuTexture = state.get(image);

	if(!gpuTexture){
		let {device} = renderer;

		gpuTexture = device.createTexture({
			size: [image.width, image.height, 1],
			format: "rgba8unorm",
			usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.COPY_DST,
		});

		device.queue.copyImageBitmapToTexture(
			{imageBitmap: image}, {texture: gpuTexture},
			[image.width, image.height, 1]
		);

		state.set(image, gpuTexture);
	}

	return gpuTexture;
}

function getPipeline(renderer, gpuTexture){

	if(pipeline){
		return pipeline;
	}

	let {device, swapChainFormat} = renderer;

	pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: "greater",
			format: "depth32float",
		},
	});

	let uniformBufferSize = 24;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	return pipeline;
}

export function drawImage(renderer, pass, image, x, y, width, height){
	let texture = getGpuTexture(renderer, image);

	drawTexture(renderer, pass, texture, x, y, width, height);
}

let states = new Map();
function getState(renderer, texture){

	let state = states.get(texture);

	if(!state){
		let {device} = renderer;

		let uniformBufferSize = 24;
		uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let sampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear",
		});

		uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
					binding: 0,
					resource: {
						buffer: uniformBuffer,
					}
				},{
					binding: 1,
					resource: sampler,
				},{
					binding: 2,
					resource: texture.createView(),
			}],
		});

		state = {uniformBuffer, uniformBindGroup};
		states.set(texture, state);
	}

	return state;
}

export function drawTexture(renderer, pass, texture, x, y, width, height){

	let {device} = renderer;

	let pipeline = getPipeline(renderer, texture);

	let {passEncoder} = pass;
	passEncoder.setPipeline(pipeline);

	Timer.timestamp(passEncoder, "texture-start");

	{
		let state = getState(renderer, texture);

		let source = new ArrayBuffer(24);
		let view = new DataView(source);

		view.setUint32(0, 5, true);
		view.setFloat32(4, x, true);
		view.setFloat32(8, y, true);
		view.setFloat32(12, width, true);
		view.setFloat32(16, height, true);
		
		device.queue.writeBuffer(
			state.uniformBuffer, 0,
			source, 0, source.byteLength
		);

		passEncoder.setBindGroup(0, state.uniformBindGroup);
	}


	passEncoder.draw(6, 1, 0, 0);

	Timer.timestamp(passEncoder, "texture-end");
}