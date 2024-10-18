
let shaderCode = `

struct Uniforms {
	x            : u32,
	y            : u32,
	width        : u32,
	height       : u32,
	numChannels  : u32,
	sampleCount  : u32,
};

@binding(0) @group(0) var<uniform> uniforms              : Uniforms;
@binding(1) @group(0) var source_color                   : texture_2d<f32>;
@binding(2) @group(0) var source_depth                   : texture_depth_2d;
@binding(3) @group(0) var source_color_ms                : texture_multisampled_2d<f32>;
@binding(4) @group(0) var source_depth_ms                : texture_depth_multisampled_2d;
@binding(5) @group(0) var<storage, read_write> outbuffer : array<u32>;

@compute @workgroup_size(16, 16)
fn main_color(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {

	if(GlobalInvocationID.x >= uniforms.width){
		return;
	}
	if(GlobalInvocationID.y >= uniforms.height){
		return;
	}

	var coords : vec2<i32>;
	coords.x = i32(uniforms.x + GlobalInvocationID.x);
	coords.y = i32(uniforms.y + GlobalInvocationID.y);

	var index : u32 = uniforms.width * GlobalInvocationID.y + GlobalInvocationID.x;

	var color = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

	if(uniforms.sampleCount == 1u){
		color = textureLoad(source_color, coords, 0);
	}else{
		color = textureLoad(source_color_ms, coords, 0);
	}

	outbuffer[uniforms.numChannels * index + 0] = bitcast<u32>(color.r);
	if(uniforms.numChannels > 1){
		outbuffer[uniforms.numChannels * index + 1] = bitcast<u32>(color.g);
	}
	if(uniforms.numChannels > 2){
		outbuffer[uniforms.numChannels * index + 2] = bitcast<u32>(color.b);
	}
	if(uniforms.numChannels > 3){
		outbuffer[uniforms.numChannels * index + 3] = bitcast<u32>(color.a);
	}

}

@compute @workgroup_size(1)
fn main_depth(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {

	if(GlobalInvocationID.x > uniforms.width){
		return;
	}
	if(GlobalInvocationID.y > uniforms.height){
		return;
	}

	var coords : vec2<i32>;
	coords.x = i32(uniforms.x + GlobalInvocationID.x);
	coords.y = i32(uniforms.y + GlobalInvocationID.y);

	var depth = textureLoad(source_depth, coords, 0);

	var index = uniforms.width * GlobalInvocationID.y + GlobalInvocationID.x;

	outbuffer[index] = bitcast<u32>(depth);
}
`;


let pipeline_color = null;
let pipeline_depth = null;
let initialized = false;

let ssbo = null;
let uniformBuffer = null;

let dummyTexture = null;
let dummyTextureMS = null;

let layout_color = null;

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	let ssboSize = 128 * 128 * 4;
	ssbo = renderer.createBuffer({size: ssboSize});

	layout_color = renderer.device.createBindGroupLayout({
		label: "readPixels",
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.COMPUTE,
			buffer: {type: 'uniform'},
		},{
			binding: 1,
			visibility: GPUShaderStage.COMPUTE,
			texture: {multisampled: false, sampleType: "unfilterable-float"},
		},{
			binding: 3,
			visibility: GPUShaderStage.COMPUTE,
			texture: {multisampled: true, sampleType: "unfilterable-float"},
		},{
			binding: 5,
			visibility: GPUShaderStage.COMPUTE,
			buffer: {type: 'storage'},
		}]
	});

	let module = device.createShaderModule({code: shaderCode});

	pipeline_color =  device.createComputePipeline({
		layout: device.createPipelineLayout({bindGroupLayouts: [layout_color]}),
		compute: {
			module,
			entryPoint: 'main_color',
		},
	});

	pipeline_depth =  device.createComputePipeline({
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'main_depth',
		},
	});

	let uniformBufferSize = 256;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	
	dummyTexture = renderer.device.createTexture({
		label: "dummy, samplecount 1",
		size: [8, 8, 1],
		format: "r32float",
		arrayLayerCount: 1,
		mipLevelCount: 1,
		sampleCount: 1,
		dimension: "2d",
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
	});

	dummyTextureMS = renderer.device.createTexture({
		label: "dummy, samplecount 4",
		size: [8, 8, 1],
		format: "r32float",
		arrayLayerCount: 1,
		mipLevelCount: 1,
		sampleCount: 4,
		dimension: "2d",
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
	});
	

	initialized = true;
}

export async function readPixels(renderer, texture, x, y, width, height){
	init(renderer);

	let {device} = renderer;

	let texture_S1;
	let texture_S4;

	if(texture.sampleCount === 1){
		texture_S1 = texture;
		texture_S4 = dummyTextureMS;
	}else{
		texture_S1 = dummyTexture;
		texture_S4 = texture;
	}
	
	let bindGroup = renderer.device.createBindGroup({
		layout: layout_color,
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: texture_S1.createView()},
			{binding: 3, resource: texture_S4.createView()},
			{binding: 5, resource: {buffer: ssbo}}
		],
	});

	const commandEncoder = device.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();

	let numChannels = 1;
	if(texture.format === "bgra8unorm")     numChannels = 4;
	else if(texture.format === "r32float")  numChannels = 1;
	else{
		console.warn(`unhandled texture format: ${texture.format}`);
	}

	{ // update uniforms
		let source = new ArrayBuffer(256);
		let view = new DataView(source);

		view.setUint32(0, x, true);
		view.setUint32(4, y, true);
		view.setUint32(8, width, true);
		view.setUint32(12, height, true);
		view.setUint32(16, numChannels, true);
		view.setUint32(20, texture.sampleCount, true);
		// view.setUint32(16, format, true);
		
		renderer.device.queue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);
	}

	passEncoder.setPipeline(pipeline_color);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(1, 1);
	passEncoder.end();
	
	device.queue.submit([commandEncoder.finish()]);

	// renderer.readBuffer(ssbo, 0, width * height * 16).then(result => {
	// 	// let array = new Float32Array(result);
	// 	// let db = Math.max(...array);

	// 	// callback({d: db});

	// 	callback(result);
	// });

	let result = await renderer.readBuffer(ssbo, 0, width * height * 16);

	return result;
}

export function readDepth(renderer, texture, x, y, width, height, callback){
	init(renderer);

	let {device} = renderer;

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline_depth.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: texture.createView({aspect: "depth-only"})},
			{binding: 2, resource: {buffer: ssbo}}
		],
	});

	const commandEncoder = device.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();

	{ // update uniforms
		let source = new ArrayBuffer(256);
		let view = new DataView(source);

		view.setUint32(0, x, true);
		view.setUint32(4, y, true);
		view.setUint32(8, width, true);
		view.setUint32(12, height, true);
		
		renderer.device.queue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);
	}

	passEncoder.setPipeline(pipeline_depth);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatch(width, height);
	passEncoder.end();
	
	device.queue.submit([commandEncoder.finish()]);

	renderer.readBuffer(ssbo, 0, width * height * 4).then(result => {
		let array = new Float32Array(result);
		let db = Math.max(...array);

		callback({d: db});
	});
}