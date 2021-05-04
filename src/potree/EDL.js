
import {Timer} from "potree";

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
		uTest : u32;
		x : f32;
		y : f32;
		width : f32;
		height : f32;
		near : f32;
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
	[[binding(3), set(0)]] var myDepth: texture_2d<f32>;

	let sampleOffsets : array<vec2<f32>, 4> = array<vec2<f32>, 4>(
		vec2<f32>(-1.0,  0.0),
		vec2<f32>( 1.0,  0.0),
		vec2<f32>( 0.0, -1.0),
		vec2<f32>( 0.0,  1.0)
	);

	[[block]] struct Uniforms {
		uTest   : u32;
		x       : f32;
		y       : f32;
		width   : f32;
		height  : f32;
		near    : f32;
		window  : i32;
	};
	
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	struct FragmentInput {
		[[builtin(position)]] fragCoord : vec4<f32>;
		[[location(0)]] uv: vec2<f32>;
	};

	struct FragmentOutput{
		[[builtin(frag_depth)]] depth : f32;
		[[location(0)]] color : vec4<f32>;
	};

	var<private> fragXY : vec2<f32>;

	fn toLinear(depth: f32, near: f32) -> f32{
		return near / depth;
	}

	fn readLinearDepth(offsetX : f32, offsetY : f32, near : f32) -> f32 {

		var fCoord : vec2<f32> = vec2<f32>(fragXY.x + offsetX, fragXY.y + offsetY);
		var iCoord : vec2<i32> = vec2<i32>(fCoord);

		var d : f32 = textureLoad(myDepth, iCoord, 0).x;
		var dl : f32 = toLinear(d, uniforms.near);

		return dl;
	}

	fn getEdlResponse(input : FragmentInput) -> f32 {

		var depth : f32 = readLinearDepth(0.0, 0.0, uniforms.near);

		var sum : f32 = 0.0;
		
		for(var i : i32 = 0; i < 4; i = i + 1){
			var offset : vec2<f32> = sampleOffsets[i];
			var neighbourDepth : f32 = readLinearDepth(offset.x, offset.y, uniforms.near);

			sum = sum + min(log2(depth) - log2(neighbourDepth), 0.0);
		}
		
		var response : f32 = sum / 4.0;

		return response;
	}

	[[stage(fragment)]]
	fn main(input : FragmentInput) -> FragmentOutput {

		fragXY = input.fragCoord.xy;

		var coords : vec2<i32> = vec2<i32>(input.fragCoord.xy);

		var c : vec4<f32> = textureLoad(myTexture, coords, 0);
		var response : f32 = getEdlResponse(input);
		var shade : f32 = exp(response * 100.0);

		var output : FragmentOutput;
		output.color = vec4<f32>(
			c.r * shade, 
			c.g * shade, 
			c.b * shade, 
			1.0);

		var d : f32 = textureLoad(myDepth, coords, 0).x;
		output.depth = d;

		return output;
	}
`;

let pipeline = null;
// let uniformBindGroup = null;
let uniformBuffer = null;

function init(renderer){

	if(pipeline !== null){
		return;
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

	let uniformBufferSize = 256;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
}

export function EDL(args = {}){

	let source = args.in;
	let target = args.target;
	let drawstate = args.drawstate;
	let {renderer, camera} = drawstate;

	init(renderer);

	let firstDraw = target.version < renderer.frameCounter;
	let view = target.colorAttachments[0].texture.createView();
	let loadValue = firstDraw ? { r: 0.1, g: 0.2, b: 0.3, a: 1.0 } : "load";
	let depthLoadValue = firstDraw ? 0 : "load";
	let renderPassDescriptor = {
		colorAttachments: [{view, loadValue}],
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadValue: depthLoadValue,
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};
	target.version = renderer.frameCounter;

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	Timer.timestamp(passEncoder,"EDL-start");

	let sampler = renderer.device.createSampler({
		magFilter: "linear",
		minFilter: "linear",
	});

	// TODO: possible issue: re-creating bind group every frame
	// doing that because the render target attachments may change after resize
	let uniformBindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: sampler},
			{binding: 2, resource: source.colorAttachments[0].texture.createView()},
			{binding: 3, resource: source.depth.texture.createView({aspect: "depth-only"})}
		],
	});

	passEncoder.setPipeline(pipeline);

	{ // update uniforms
		let source = new ArrayBuffer(32);
		let view = new DataView(source);

		let size = Number(guiContent["point size"]);
		let window = Math.round((size - 1) / 2);

		view.setUint32(0, 5, true);
		view.setFloat32(4, 0, true);
		view.setFloat32(8, 0, true);
		view.setFloat32(12, 1, true);
		view.setFloat32(16, 1, true);
		view.setFloat32(20, camera.near, true);
		view.setInt32(24, window, true);
		
		renderer.device.queue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);

		passEncoder.setBindGroup(0, uniformBindGroup);
	}

	passEncoder.draw(6, 1, 0, 0);

	Timer.timestamp(passEncoder,"EDL-end");

	passEncoder.endPass();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);
	
}