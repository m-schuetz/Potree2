
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
		Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
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

	[[binding(1), set(0)]] var<uniform_constant> mySampler: sampler;
	[[binding(2), set(0)]] var<uniform_constant> myTexture: texture_sampled_2d<f32>;

	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		outColor =  textureSample(myTexture, mySampler, fragUV);

		return;
	}
`;


let bindGroupLayout = null;
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

		device.defaultQueue.copyImageBitmapToTexture(
			{imageBitmap: image}, {texture: gpuTexture},
			[image.width, image.height, 1]
		);

		state.set(image, gpuTexture);
	}

	return gpuTexture;
}


function getBindGroupLayout(renderer){

	if(!bindGroupLayout){
		let {device} = renderer;

		bindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				type: "uniform-buffer"
			},{
				// Sampler
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				type: "sampler"
			},{
				// Texture view
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				type: "sampled-texture"
			}]
		});
	}

	return bindGroupLayout;
}

function getPipeline(renderer, gpuTexture){

	if(pipeline){
		return pipeline;
	}

	let {device, swapChainFormat} = renderer;

	let bindGroupLayout = getBindGroupLayout(renderer);
	let layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

	pipeline = device.createRenderPipeline({
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

	{
		let state = getState(renderer, texture);

		let source = new ArrayBuffer(24);
		let view = new DataView(source);

		view.setUint32(0, 5, true);
		view.setFloat32(4, x, true);
		view.setFloat32(8, y, true);
		view.setFloat32(12, width, true);
		view.setFloat32(16, height, true);
		
		device.defaultQueue.writeBuffer(
			state.uniformBuffer, 0,
			source, 0, source.byteLength
		);

		passEncoder.setBindGroup(0, state.uniformBindGroup);
	}


	passEncoder.draw(6, 1, 0, 0);

}