
let csColor = `

[[block]] struct Uniforms {
	x : u32;
	y : u32;
	width: u32;
	height: u32;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var source : texture_2d<f32>;
[[binding(2), group(0)]] var<storage, read_write> target : U32s;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	if(GlobalInvocationID.x > uniforms.width){
		return;
	}
	if(GlobalInvocationID.y > uniforms.height){
		return;
	}

	var coords : vec2<i32>;
	coords.x = i32(uniforms.x + GlobalInvocationID.x);
	coords.y = i32(uniforms.y + GlobalInvocationID.y);

	var color : vec4<f32> = textureLoad(source, coords, 0);

	var index : u32 = uniforms.width * GlobalInvocationID.y + GlobalInvocationID.x;

	target.values[index] = u32(color.r * 256.0);
}
`;

let csDepth = `

[[block]] struct Uniforms {
	x : u32;
	y : u32;
	width: u32;
	height: u32;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var source : texture_depth_2d;
[[binding(2), group(0)]] var<storage, read_write> target : U32s;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	if(GlobalInvocationID.x > uniforms.width){
		return;
	}
	if(GlobalInvocationID.y > uniforms.height){
		return;
	}

	var coords : vec2<i32>;
	coords.x = i32(uniforms.x + GlobalInvocationID.x);
	coords.y = i32(uniforms.y + GlobalInvocationID.y);

	var depth = textureLoad(source, coords, 0);

	var index = uniforms.width * GlobalInvocationID.y + GlobalInvocationID.x;

	target.values[index] = bitcast<u32>(depth);
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
	ssbo = renderer.createBuffer(ssboSize);

	pipeline_color =  device.createComputePipeline({
		compute: {
			module: device.createShaderModule({code: csColor}),
			entryPoint: 'main',
		},
	});

	pipeline_depth =  device.createComputePipeline({
		compute: {
			module: device.createShaderModule({code: csDepth}),
			entryPoint: 'main',
		},
	});

	let uniformBufferSize = 256;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	initialized = true;
}

// function read(renderer, texture, x, y, width, height, callback, format){
// 	init(renderer);

// 	let {device} = renderer;

// 	let bindGroup = renderer.device.createBindGroup({
// 		layout: pipeline.getBindGroupLayout(0),
// 		entries: [
// 			{binding: 0, resource: {buffer: uniformBuffer}},
// 			{binding: 1, resource: texture.createView()},
// 			{binding: 2, resource: {buffer: ssbo}}
// 		],
// 	});

// 	const commandEncoder = device.createCommandEncoder();
// 	const passEncoder = commandEncoder.beginComputePass();

// 	{ // update uniforms
// 		let source = new ArrayBuffer(256);
// 		let view = new DataView(source);

// 		view.setUint32(0, x, true);
// 		view.setUint32(4, y, true);
// 		view.setUint32(8, width, true);
// 		view.setUint32(12, height, true);
// 		view.setUint32(16, format, true);
		
// 		renderer.device.queue.writeBuffer(
// 			uniformBuffer, 0,
// 			source, 0, source.byteLength
// 		);
// 	}

// 	passEncoder.setPipeline(pipeline);
// 	passEncoder.setBindGroup(0, bindGroup);
// 	passEncoder.dispatch(width, height);
// 	passEncoder.endPass();
	
// 	device.queue.submit([commandEncoder.finish()]);

// 	renderer.readBuffer(ssbo, 0, width * height * 4).then(result => {
// 		let array = new Float32Array(result);
// 		let db = Math.max(...array);

// 		callback({d: db});
// 	});
// }

export function readPixels(renderer, texture, x, y, width, height, callback){
	init(renderer);

	let {device} = renderer;

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline_color.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: texture.createView()},
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
		view.setUint32(16, format, true);
		
		renderer.device.queue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);
	}

	passEncoder.setPipeline(pipeline_color);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatch(width, height);
	passEncoder.endPass();
	
	device.queue.submit([commandEncoder.finish()]);

	renderer.readBuffer(ssbo, 0, width * height * 4).then(result => {
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
	passEncoder.endPass();
	
	device.queue.submit([commandEncoder.finish()]);

	renderer.readBuffer(ssbo, 0, width * height * 4).then(result => {
		let array = new Float32Array(result);
		let db = Math.max(...array);

		callback({d: db});
	});
}