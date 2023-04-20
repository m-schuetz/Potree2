
import {Timer} from "potree";

let vs = `

	struct Uniforms {
		uTest : u32,
		x : f32,
		y : f32,
		width : f32,
		height : f32,
		near : f32,
	};

	@binding(0) @group(0) var<uniform> uniforms : Uniforms;

	struct VertexInput {
		@builtin(vertex_index) index : u32,
	};

	struct VertexOutput {
		@builtin(position) position : vec4<f32>,
		@location(0) uv : vec2<f32>,
	};

	@vertex
	fn main(vertex : VertexInput) -> VertexOutput {

		var output : VertexOutput;

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

		output.position = vec4<f32>(pos[vertex.index], 0.999, 1.0);
		output.uv = uv[vertex.index];

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		var vi : u32 = vertex.index;
		
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
`;

let fs = `

	@binding(1) @group(0) var mySampler   : sampler;
	@binding(2) @group(0) var myTexture   : texture_2d<f32>;
	@binding(3) @group(0) var myDepth     : texture_depth_2d;

	struct Uniforms {
		uTest   : u32,
		x       : f32,
		y       : f32,
		width   : f32,
		height  : f32,
		near    : f32,
		window  : i32,
	};
	
	@binding(0) @group(0) var<uniform> uniforms : Uniforms;

	struct FragmentInput {
		@builtin(position) fragCoord : vec4<f32>,
		@location(0) uv: vec2<f32>,
	};

	struct FragmentOutput{
		@builtin(frag_depth) depth : f32,
		@location(0) color : vec4<f32>,
	};

	var<private> fragXY : vec2<f32>;

	fn toLinear(depth: f32, near: f32) -> f32{
		return near / depth;
	}

	fn readLinearDepth(offsetX : f32, offsetY : f32, near : f32) -> f32 {

		var fCoord : vec2<f32> = vec2<f32>(fragXY.x + offsetX, fragXY.y + offsetY);
		var iCoord : vec2<i32> = vec2<i32>(fCoord);

		var d : f32 = textureLoad(myDepth, iCoord, 0);
		// var d : f32 = textureLoad(myDepth, iCoord, 0).x;
		var dl : f32 = toLinear(d, uniforms.near);

		// Artificially reduce depth precision to visualize artifacts
		// var di : u32 = u32(10.0 * dl);
		// dl = f32(di);

		return dl;
	}

	fn getEdlResponse(input : FragmentInput) -> f32 {

		var depth : f32 = readLinearDepth(0.0, 0.0, uniforms.near);

		var sum : f32 = 0.0;

		// var sampleOffsets : array<vec2<f32>, 4> = array<vec2<f32>, 4>(
		// 	vec2<f32>(-1.0,  0.0),
		// 	vec2<f32>( 1.0,  0.0),
		// 	vec2<f32>( 0.0, -1.0),
		// 	vec2<f32>( 0.0,  1.0)
		// );

		var sampleOffsets : array<vec2<f32>, 8> = array<vec2<f32>, 8>(
			vec2<f32>(0.0, 1.0),
			vec2<f32>(0.7071067811865475, 0.70710678118654769),
			vec2<f32>(1.0, 0.0),
			vec2<f32>(0.7071067811865476, -0.7071067811865475),
			vec2<f32>(0.0, -1.0),
			vec2<f32>(-0.7071067811865475, -0.7071067811865477),
			vec2<f32>(-1.0, 0.0),
			vec2<f32>(-0.7071067811865477, 0.7071067811865474),
		);
		
		for(var i : i32 = 0; i < 8; i = i + 1){
			var offset : vec2<f32> = sampleOffsets[i];
			var neighbourDepth : f32 = readLinearDepth(offset.x, offset.y, uniforms.near);

			sum = sum + max(log2(depth) - log2(neighbourDepth), 0.0);
			// sum = sum + min(log2(depth) - log2(neighbourDepth), 0.0);
		}
		
		var response : f32 = sum / 8.0;

		return response;
	}

	@fragment
	fn main(input : FragmentInput) -> FragmentOutput {

		_ = mySampler;

		fragXY = input.fragCoord.xy;
		var coords : vec2<i32> = vec2<i32>(input.fragCoord.xy);

		var output : FragmentOutput;
		output.color = textureLoad(myTexture, coords, 0);
		output.depth = textureLoad(myDepth, coords, 0);


		var response = getEdlResponse(input);
		var w = exp(-response * 300.0f * 0.4f);
		output.color.r *= w;
		output.color.g *= w;
		output.color.b *= w;


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
		layout: "auto",
		vertex: {
			module: device.createShaderModule({code: vs, label: "vs_edl"}),
			entryPoint: "main",
		},
		fragment: {
			module: device.createShaderModule({code: fs, label: "fs_edl"}),
			entryPoint: "main",
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

	let uniformBufferSize = 256;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
}

let uniformBindGroupCache = new Map();
function getUniformBindGroup(renderer, source){

	let data = uniformBindGroupCache.get(source);

	if(data == null || data.version < source.version){

		let sampler = renderer.device.createSampler({
			magFilter: "linear",
			minFilter: "linear",
		});
		
		let uniformBindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: sampler},
				{binding: 2, resource: source.colorAttachments[0].texture.createView()},
				{binding: 3, resource: source.depth.texture.createView({aspect: "depth-only"})}
			],
		});

		let data = {
			version: source.version, 
			uniformBindGroup
		};

		uniformBindGroupCache.set(source, data);

	}

	return uniformBindGroupCache.get(source).uniformBindGroup;
}

export function EDL(source, drawstate){

	let {renderer, camera, pass} = drawstate;
	let {passEncoder} = pass;

	init(renderer);

	Timer.timestamp(passEncoder,"EDL-start");

	let uniformBindGroup = getUniformBindGroup(renderer, source);

	passEncoder.setPipeline(pipeline);

	{ // update uniforms
		let source = new ArrayBuffer(32);
		let view = new DataView(source);

		let size = Potree.settings.pointSize;
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
}