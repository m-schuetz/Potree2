
import * as Timer from "../renderer/Timer.js";

const source = `

	struct Uniforms {
		uTest : u32;
		x : f32;
		y : f32;
		width : f32;
		height : f32;
	};
	@binding(0) @group(0) var<uniform> uniforms : Uniforms;
	@binding(1) @group(0) var mySampler: sampler;
	@binding(2) @group(0) var myTexture: texture_2d<f32>;

	struct VertexInput {
		@builtin(vertex_index) index : u32,
	};

	struct VertexOutput {
		@builtin(position) position : vec4<f32>,
		@location(0) uv : vec2<f32>,
	};

	struct FragmentInput {
		@location(0) uv: vec2<f32>,
	};

	@stage(vertex)
	fn main_vertex(vertex : VertexInput) -> VertexOutput {

		var pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
			vec2<f32>(0.0, 0.0),
			vec2<f32>(0.1, 0.0),
			vec2<f32>(0.1, 0.1),
			vec2<f32>(0.0, 0.0),
			vec2<f32>(0.1, 0.1),
			vec2<f32>(0.0, 0.1)
		);

		var uv : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
			vec2<f32>(0.0, 1.0),
			vec2<f32>(1.0, 1.0),
			vec2<f32>(1.0, 0.0),
			vec2<f32>(0.0, 1.0),
			vec2<f32>(1.0, 0.0),
			vec2<f32>(0.0, 0.0)
		);

		var output : VertexOutput;

		output.position = vec4<f32>(pos[vertex.index], 0.01, 1.0);
		output.uv = uv[vertex.index];

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		var vi = vertex.index;

		if(vi == 0u || vi == 3u || vi == 5u){
			output.position.x = x;
		}else{
			output.position.x = x + width;
		}

		if(vi == 0u || vi == 1u || vi == 3u){
			output.position.y = y;
		}else{
			output.position.y = y + height;
		}

		return output;
	}

	@stage(fragment)
	fn main_fragment(input : FragmentInput) -> @location(0) vec4<f32> {

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
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
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
			module: device.createShaderModule({code: source}),
			entryPoint: "main_vertex",
		},
		fragment: {
			module: device.createShaderModule({code: source}),
			entryPoint: "main_fragment",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: "always",
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