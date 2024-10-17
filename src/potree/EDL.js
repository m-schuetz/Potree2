
import {Timer} from "potree";

let shaderCode = `

	struct Uniforms {
		uTest : u32,
		x : f32,
		y : f32,
		width : f32,
		height : f32,
		near : f32,
	};

	@binding(0) @group(0) var<uniform> uniforms : Uniforms;
	@binding(2) @group(0) var myTexture         : texture_2d<f32>;
	@binding(3) @group(0) var myDepth           : texture_depth_2d;

	struct VertexInput {
		@builtin(vertex_index) index : u32,
	};

	struct VertexOutput {
		@builtin(position) position : vec4<f32>,
		@location(0) uv : vec2<f32>,
	};

	@vertex
	fn main_vertex(vertex : VertexInput) -> VertexOutput {

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

	fn readLinearDepth(offsetX : f32, offsetY : f32, near : f32, sampleIndex : u32) -> f32 {

		var fCoord : vec2<f32> = vec2<f32>(fragXY.x + offsetX, fragXY.y + offsetY);
		var iCoord : vec2<i32> = vec2<i32>(fCoord);

		var d : f32 = textureLoad(myDepth, iCoord, sampleIndex);
		var dl : f32 = toLinear(d, uniforms.near);

		return dl;
	}

	fn getEdlResponse(input : FragmentInput, sampleIndex: u32) -> f32 {

		var depth : f32 = readLinearDepth(0.0, 0.0, uniforms.near, sampleIndex);
		var sum : f32 = 0.0;

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
			var neighbourDepth : f32 = readLinearDepth(offset.x, offset.y, uniforms.near, sampleIndex);

			sum = sum + max(log2(depth) - log2(neighbourDepth), 0.0);
		}
		
		var response : f32 = sum / 8.0;

		return response;
	}

	@fragment
	fn main_fragment(input : FragmentInput) -> FragmentOutput {
		fragXY = input.fragCoord.xy;
		var coords : vec2<i32> = vec2<i32>(input.fragCoord.xy);

		var edlStrength = 0.2f;

		var output : FragmentOutput;
		var color = textureLoad(myTexture, coords, 0);
		output.depth = textureLoad(myDepth, coords, 0);
		var response = getEdlResponse(input, 0);
		var w = exp(-response * 300.0f * edlStrength);

		if(sampleCount == 4){
			color = color + textureLoad(myTexture, coords, 1);
			color = color + textureLoad(myTexture, coords, 2);
			color = color + textureLoad(myTexture, coords, 3);
			color = color * 0.25f;

			w = w + exp(-getEdlResponse(input, 1) * 300.0f * edlStrength);
			w = w + exp(-getEdlResponse(input, 2) * 300.0f * edlStrength);
			w = w + exp(-getEdlResponse(input, 3) * 300.0f * edlStrength);
			w = w * 0.25f;
		}

		color.r *= w;
		color.g *= w;
		color.b *= w;

		output.color = color;

		return output;
	}
`;

// let layout = null;
let initialized = false;
let pipelineCache = new Map();
let uniformBuffer = null;


function getLayout(renderer, renderTarget){

	let sampleCount = renderTarget.colorAttachments[0].texture.sampleCount;

	let layout = null;
	
	if(sampleCount === 1){
		layout = renderer.device.createBindGroupLayout({
			label: "EDL",
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'uniform'},
			},{
				binding: 2,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				texture: {multisampled: false, sampleType: "unfilterable-float"},
			},{
				binding: 3,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				texture: {multisampled: false, sampleType: "depth"},
			}]
		});
	}else{
		layout = renderer.device.createBindGroupLayout({
			label: "EDL",
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'uniform'},
			},{
				binding: 2,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				texture: {multisampled: true, sampleType: "unfilterable-float"},
			},{
				binding: 3,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				texture: {multisampled: true, sampleType: "depth"},
			}]
		});
	}

	return layout;
}

function getPipeline(renderer, renderTarget){

	let sampleCount = renderTarget.colorAttachments[0].texture.sampleCount;

	let key = `sampleCount=${sampleCount}`;

	if(!pipelineCache.has(key)){
		let {device} = renderer;

		let code = shaderCode;
		if(sampleCount > 1){
			code = code.replaceAll("texture_2d", "texture_multisampled_2d");
			code = code.replaceAll("texture_depth_2d", "texture_depth_multisampled_2d");
		}
		code = code.replaceAll("sampleCount", sampleCount);

		let module = device.createShaderModule({code, label: "EDL"});

		let layout = getLayout(renderer, renderTarget);
		
		let pipeline = device.createRenderPipeline({
			label: EDL,
			layout: device.createPipelineLayout({
				bindGroupLayouts: [layout],
			}),
			vertex: {module, entryPoint: "main_vertex"},
			fragment: {
				module,
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

		pipelineCache.set(key, pipeline);
	}

	return pipelineCache.get(key);
}

function init(renderer){

	if(initialized) return;

	let uniformBufferSize = 256;
	uniformBuffer = renderer.device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	initialized = true;
}

let uniformBindGroupCache = new Map();
function getUniformBindGroup(renderer, source){

	// let data = uniformBindGroupCache.get(source);

	// if(data == null || data.version < source.version){

	// 	let layout = getLayout(renderer, source);
		
	// 	let uniformBindGroup = renderer.device.createBindGroup({
	// 		label: "EDL",
	// 		layout,
	// 		entries: [
	// 			{binding: 0, resource: {buffer: uniformBuffer}},
	// 			{binding: 2, resource: source.colorAttachments[0].texture.createView()},
	// 			{binding: 3, resource: source.depth.texture.createView({aspect: "depth-only"})}
	// 		],
	// 	});

	// 	let data = {
	// 		version: source.version, 
	// 		uniformBindGroup
	// 	};

	// 	uniformBindGroupCache.set(source, data);

	// }

	// return uniformBindGroupCache.get(source).uniformBindGroup;

	let layout = getLayout(renderer, source);
	
	// TODO: cache this?
	let uniformBindGroup = renderer.device.createBindGroup({
		label: "EDL",
		layout,
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 2, resource: source.colorAttachments[0].texture.createView()},
			{binding: 3, resource: source.depth.texture.createView({aspect: "depth-only"})}
		],
	});

	return uniformBindGroup;
}

export function EDL(source, drawstate){

	let {renderer, camera, pass} = drawstate;
	let {passEncoder} = pass;

	init(renderer);

	Timer.timestamp(passEncoder,"EDL-start");

	let uniformBindGroup = getUniformBindGroup(renderer, source);

	passEncoder.setPipeline(getPipeline(renderer, source));

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