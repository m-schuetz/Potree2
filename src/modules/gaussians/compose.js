
// import {SplatType, Timer} from "potree";

let shaderCode = `
	// struct Uniforms {
	// 	uTest     : u32,
	// 	x         : f32,
	// 	y         : f32,
	// 	width     : f32,
	// 	height    : f32,
	// 	near      : f32,
	// 	window    : i32,
	// };

	// @binding(0) @group(0) var<uniform> uniforms : Uniforms;
	@binding(1) @group(0) var mySampler   : sampler;
	@binding(2) @group(0) var myTexture   : texture_2d<f32>;
	// @binding(3) @group(0) var tex_pointID : texture_2d<u32>;
	// @binding(4) @group(0) var myDepth     : texture_depth_2d;

	@group(0) @binding(0) var accumTex : texture_2d<f32>;
	@group(0) @binding(1) var accumSampler : sampler;

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

		// let uv = pos.xy / vec2<f32>(outputResolution);
		// let acc = textureSample(accumTex, accumSampler, uv);
		// let color = acc.rgb / max(acc.a, 1e-6);

		var size = textureDimensions(myTexture);
		var uv = vec2f(
			pos.x / f32(size.x),
			pos.y / f32(size.y)
		);
		// var color = vec4f(0.0f, 1.0f, 0.0f, 1.0f);
		// color.x = uv.x;
		// color.y = uv.y;

		var sourceColor = textureSample(myTexture, mySampler, uv);
		var color = vec4f(sourceColor.xyz, 1.0f);
		// var c = sourceColor.rgb / max(sourceColor.a, 1e-6);
		// var color = vec4f(c, 1.0f);

		return color;
	}
`;

// let pipeline = null;
// let uniformBuffer = null;

let initialized = false;
let pipeline = null;

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	let module = device.createShaderModule({code: shaderCode});

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
				{format: "bgra8unorm"},
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
				depthWriteEnabled: false,
				depthCompare: "always",
				format: "depth32float",
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

	// let {renderer, camera, pass} = drawstate;
	// let {passEncoder} = pass;


	init(renderer);



	let colorAttachments = [{
		view: target.colorAttachments[0].texture.createView(), 
		loadOp: "clear", 
		clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
		storeOp: 'store',
	}];

	let renderPassDescriptor = {
		colorAttachments,
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadOp: "load",
			depthStoreOp: "store",
		},
		sampleCount: 1,
	};

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);




	// Timer.timestamp(passEncoder,"hqs-normalize-start");

	let sampler = renderer.device.createSampler({
		magFilter: "linear",
		minFilter: "linear",
	});

	// TODO: possible issue: re-creating bind group every frame
	// doing that because the render target attachments may change after resize

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			// {binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: sampler},
			{binding: 2, resource: source.colorAttachments[0].texture.createView()},
			// {binding: 3, resource: source.colorAttachments[1].texture.createView()},
			// {binding: 4, resource: source.depth.texture.createView({aspect: "depth-only"})}
		],
	});

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	// { // update uniforms
	// 	let source = new ArrayBuffer(32);
	// 	let view = new DataView(source);

	// 	let size = Potree.settings.pointSize;
	// 	let window = Math.round((size - 1) / 2);

			
	// 	view.setFloat32(4, 0, true);
	// 	view.setFloat32(8, 0, true);
	// 	view.setFloat32(12, 1, true);
	// 	view.setFloat32(16, 1, true);
	// 	view.setFloat32(20, camera.near, true);
	// 	view.setInt32(24, window, true);
		
	// 	renderer.device.queue.writeBuffer(
	// 		uniformBuffer, 0,
	// 		source, 0, source.byteLength
	// 	);

	// 	passEncoder.setBindGroup(0, uniformBindGroup);
	// }

	passEncoder.draw(6, 1, 0, 0);

	passEncoder.end();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);

	// Timer.timestamp(passEncoder,"hqs-normalize-end");

}