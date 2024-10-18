
let shaderCode = `

struct Uniforms {
	x            : u32,
	y            : u32,
	width        : u32,
	height       : u32,
	numChannels  : u32,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var source_color : texture_2d<f32>;
@binding(2) @group(0) var source_depth : texture_depth_2d;
@binding(3) @group(0) var<storage, read_write> outbuffer : array<u32>;

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

	var color : vec4<f32> = textureLoad(source_color, coords, 0);

	var index : u32 = uniforms.width * GlobalInvocationID.y + GlobalInvocationID.x;

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

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	let ssboSize = 128 * 128 * 4;
	ssbo = renderer.createBuffer({size: ssboSize});

	let module = device.createShaderModule({code: shaderCode});

	pipeline_color =  device.createComputePipeline({
		layout: 'auto',
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

	initialized = true;
}

export function readPixels(renderer, texture, x, y, width, height, callback){
	init(renderer);

	let {device} = renderer;

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline_color.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: texture.createView()},
			{binding: 3, resource: {buffer: ssbo}}
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

	renderer.readBuffer(ssbo, 0, width * height * 16).then(result => {
		let array = new Float32Array(result);
		let db = Math.max(...array);

		callback({d: db});
	});
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