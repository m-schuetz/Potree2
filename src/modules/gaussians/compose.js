
let shaderCode = `
	@group(0) @binding(1) var mySampler   : sampler;
	@group(0) @binding(2) var myTexture   : texture_2d<f32>;

	@vertex
	fn main_vs(@builtin(vertex_index) index : u32) -> @builtin(position) vec4<f32> {

		var pos = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

		if(index == 0u){ pos = vec4f(-1.0f, -1.0f, 0.0f, 0.0f); }
		if(index == 1u){ pos = vec4f( 1.0f, -1.0f, 0.0f, 0.0f); }
		if(index == 2u){ pos = vec4f( 1.0f,  1.0f, 0.0f, 0.0f); }
		if(index == 3u){ pos = vec4f(-1.0f, -1.0f, 0.0f, 0.0f); }
		if(index == 4u){ pos = vec4f( 1.0f,  1.0f, 0.0f, 0.0f); }
		if(index == 5u){ pos = vec4f( 1.0f, -1.0f, 0.0f, 0.0f); }
		
		return pos;
	}

	@fragment
	fn main_fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {

		_ = mySampler;
		_= myTexture;

		var size = textureDimensions(myTexture);
		var uv = vec2f(
			pos.x / f32(size.x),
			pos.y / f32(size.y)
		);

		var sourceColor = textureSample(myTexture, mySampler, uv);

		return sourceColor;
	}
`;

let initialized = false;
let pipeline = null;

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	let module = device.createShaderModule({code: shaderCode});

	let blend = {
		color: {
			srcFactor: "one",
			dstFactor: "one-minus-src-alpha",
			operation: "add",
		},
		alpha: {
			srcFactor: "one",
			dstFactor: "one-minus-src-alpha",
			operation: "add",
		},
	};

	pipeline = device.createRenderPipeline({
		layout: "auto",
		vertex: {
			module,
			entryPoint: "main_vs",
		},
		fragment: {
			module,
			entryPoint: "main_fs",
			targets: [
				{format: "bgra8unorm", blend: blend},
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
	});


	// let uniformBufferSize = 256;
	// uniformBuffer = device.createBuffer({
	// 	size: uniformBufferSize,
	// 	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	// });

	initialized = true;
}

export function compose(renderer, source, target){

	init(renderer);

	let colorAttachments = [{
		view: target.colorAttachments[0].texture.createView(), 
		loadOp: "load", 
		storeOp: 'store',
	}];

	let renderPassDescriptor = {
		colorAttachments,
		sampleCount: 1,
	};

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	let sampler = renderer.device.createSampler({
		magFilter: "linear",
		minFilter: "linear",
	});

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 1, resource: sampler},
			{binding: 2, resource: source.colorAttachments[0].texture.createView()},
		],
	});

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	passEncoder.draw(6, 1, 0, 0);

	passEncoder.end();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);

}